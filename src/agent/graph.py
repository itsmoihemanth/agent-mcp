import structlog
from langgraph.prebuilt import create_react_agent  # noqa: deprecation warning expected in langgraph 1.x
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI

logger = structlog.get_logger()

SYSTEM_PROMPT = """You are a helpful AI assistant with access to tools.
Use the appropriate tool for each task:
- For math calculations, use the calculator tool
- For running Python code (data generation, text processing, algorithms, anything requiring execution), use the python_executor tool
- For fetching a specific URL or web page (reading articles, docs, READMEs), use the fetch_url tool
- For searching the web for current events or general information, use the web search tool
- For reading local files, use the file reader tools
- For questions about research papers, ingested documents, or technical concepts from the knowledge base, use the search_documents tool
If a tool returns an error, explain the issue to the user clearly.
Do not make up information - use tools to find answers.
When using python_executor, always use print() to output results."""

# In-memory checkpointer for conversation persistence per thread_id
checkpointer = MemorySaver()


def create_agent(model: ChatOpenAI, tools: list):
    """Create a ReAct agent with the given model and tools.

    Args:
        model: ChatOpenAI model instance
        tools: Flat list of LangChain-compatible tools (native + MCP)

    Returns:
        Compiled LangGraph agent with memory checkpointer
    """
    logger.info("creating_agent", tool_count=len(tools),
                tool_names=[t.name for t in tools])
    agent = create_react_agent(
        model=model, tools=tools, prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )
    return agent
