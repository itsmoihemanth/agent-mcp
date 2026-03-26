from contextlib import asynccontextmanager
from pathlib import Path
import structlog
from fastapi import FastAPI, Request, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from src.core.config import get_settings
from src.core.logging import setup_logging
from src.agent.graph import create_agent
from src.agent.tools import get_native_tools
from src.api.routes_health import router as health_router
from src.api.routes_agent import router as agent_router
from src.api.routes_tools import router as tools_router
from src.api.routes_rag import router as rag_router
from src.api.routes_trace import router as trace_router
from src.rag.vectorstore import init_vectorstore
from src.rag.retriever_tool import make_retriever_tool

logger = structlog.stdlib.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    setup_logging(log_level=settings.log_level, json_logs=settings.log_json)

    # Validate required API key
    if not settings.openai_api_key:
        logger.error("openai_api_key_missing", message="OPENAI_API_KEY is required")
        raise RuntimeError("OPENAI_API_KEY must be set")

    model_kwargs = {"model": settings.model_name, "api_key": settings.openai_api_key}
    if settings.openai_api_base:
        model_kwargs["base_url"] = settings.openai_api_base
    model = ChatOpenAI(**model_kwargs)
    native_tools = get_native_tools(settings)

    # Initialize vectorstore and RAG tool (graceful degradation if unavailable)
    rag_tool = None
    app.state.vectorstore = None
    try:
        vectorstore = await init_vectorstore(
            settings.database_url, settings.openai_api_key, settings.openai_api_base
        )
        rag_tool = make_retriever_tool(vectorstore)
        app.state.vectorstore = vectorstore
        logger.info("vectorstore_ready")
    except Exception as e:
        logger.warning("vectorstore_init_failed", error=str(e),
                       message="Running without RAG capabilities")

    rag_tools = [rag_tool] if rag_tool else []

    # Initialize MCP tools (new API: no context manager)
    mcp_tools = []
    mcp_client = None
    try:
        mcp_client = MultiServerMCPClient(
            {
                "file_reader": {
                    "command": "python",
                    "args": [settings.mcp_file_reader_path],
                    "transport": "stdio",
                }
            }
        )
        mcp_tools = await mcp_client.get_tools()
        logger.info("mcp_tools_loaded", count=len(mcp_tools))
    except Exception as e:
        logger.warning("mcp_client_failed", error=str(e),
                       message="Running with native tools only")

    all_tools = mcp_tools + native_tools + rag_tools
    agent = create_agent(model, all_tools)
    app.state.agent = agent
    app.state.tools = all_tools
    app.state.mcp_client = mcp_client

    logger.info("agent_ready", tool_count=len(all_tools),
                native_count=len(native_tools), mcp_count=len(mcp_tools),
                rag_count=len(rag_tools))
    yield

    logger.info("app_shutting_down")


app = FastAPI(
    title="AgentMCP",
    description="LangGraph Agent with MCP Tools and RAG",
    version="0.1.0",
    lifespan=lifespan,
)


# Structured error handlers -- no raw tracebacks in responses


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "error": "validation_error",
            "detail": exc.errors(),
            "message": "Request validation failed",
        },
    )


@app.exception_handler(HTTPException)
async def http_error_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "http_error",
            "detail": exc.detail,
            "message": str(exc.detail),
        },
    )


@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception):
    logger.exception("unhandled_error", error=str(exc))
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            "detail": None,
            "message": "An internal error occurred",
        },
    )


app.include_router(health_router)
app.include_router(agent_router)
app.include_router(tools_router)
app.include_router(rag_router)
app.include_router(trace_router)

# Serve chat UI
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    async def serve_ui():
        return FileResponse(str(STATIC_DIR / "index.html"))
