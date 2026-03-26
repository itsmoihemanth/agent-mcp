# AgentMCP

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.128-009688.svg)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.1-orange.svg)](https://github.com/langchain-ai/langgraph)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED.svg)](https://docs.docker.com/compose/)
[![MCP](https://img.shields.io/badge/MCP-Protocol-purple.svg)](https://modelcontextprotocol.io)

**A LangGraph ReAct agent with MCP tool integration, RAG retrieval, and structured observability -- deployable with a single command.**

---

## Architecture

```mermaid
graph TD
    User["User"] -->|"POST /agent/chat"| FastAPI["FastAPI"]
    FastAPI --> LangGraph["LangGraph ReAct Agent"]
    LangGraph -->|route| Calculator["Calculator"]
    LangGraph -->|route| WebSearch["Tavily Web Search"]
    LangGraph -->|route| FileReader["MCP File Reader"]
    LangGraph -->|route| RAG["RAG Retriever"]
    FileReader -->|stdio| MCPServer["MCP Server Process"]
    RAG -->|similarity search| pgvector[("PostgreSQL + pgvector")]
    LangGraph -->|response| FastAPI
    FastAPI -->|JSON| User
```

The agent receives a user message, reasons about which tools to invoke using the ReAct pattern, executes one or more tool calls, and synthesizes a final response. Tools are sourced from two integration patterns: **native LangGraph tools** (calculator, web search) and **MCP protocol tools** (file reader via stdio transport).

---

## Why I Built This

I have hands-on experience building AI agent orchestration systems and multi-model pipelines in production environments, along with published research (IEEE) on applied machine learning. This project distills those patterns into a clean, self-contained demonstration: a ReAct agent that routes between native tools and MCP-protocol tools, retrieves context from a vector store, and logs every decision with structured observability. It is designed to show production thinking -- graceful degradation when services are unavailable, clean separation of concerns between agent logic and API surface, and infrastructure that runs with a single command.

---

## Features

- **ReAct Agent** -- LangGraph state machine with explicit tool routing and multi-step reasoning
- **MCP Integration** -- File reader server using the Model Context Protocol (stdio transport)
- **Native Tools** -- Calculator (safe math eval) and Tavily web search
- **RAG Pipeline** -- Document ingestion, chunking, embedding, and similarity search via pgvector
- **Structured Observability** -- JSON-structured logging of every agent step, tool call, and latency
- **Graceful Degradation** -- Agent runs with reduced capabilities if MCP or vectorstore are unavailable
- **Docker Deployment** -- Full stack (FastAPI + PostgreSQL/pgvector) in one `docker-compose up`
- **Clean API** -- Typed request/response schemas, OpenAPI docs auto-generated at `/docs`

---

## Quick Start

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/)
- An [OpenAI API key](https://platform.openai.com/api-keys)
- (Optional) A [Tavily API key](https://tavily.com) for web search

### Setup

```bash
# Clone the repository
git clone https://github.com/your-username/agent-mcp.git
cd agent-mcp

# Configure environment
cp .env.example .env
# Edit .env and set your OPENAI_API_KEY (required)
# Optionally set TAVILY_API_KEY for web search

# Start everything
docker-compose up --build
```

### Verify

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{
  "status": "healthy",
  "database": { "connected": true, "error": null },
  "pgvector": { "installed": true }
}
```

API documentation is available at [http://localhost:8000/docs](http://localhost:8000/docs).

---

## Demo Scenario

Try these three requests to see the agent use different tools in sequence.

### 1. Calculator -- Math Reasoning

```bash
curl -s -X POST http://localhost:8000/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is (245 * 18) + (372 / 4)?"}' | jq
```

```json
{
  "response": "The result of (245 * 18) + (372 / 4) is 4503.0",
  "tools_used": ["calculator"],
  "metadata": { "duration_ms": 1200 }
}
```

### 2. Web Search -- Current Information

```bash
curl -s -X POST http://localhost:8000/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What are the latest developments in MCP protocol?"}' | jq
```

```json
{
  "response": "Based on my web search, the Model Context Protocol...",
  "tools_used": ["tavily_search"],
  "metadata": { "duration_ms": 3500 }
}
```

> Requires `TAVILY_API_KEY` to be set. Without it, the agent will respond using its own knowledge.

### 3. RAG -- Document Retrieval

First, ingest a document:

```bash
curl -s -X POST http://localhost:8000/rag/ingest \
  -F "file=@your-document.pdf" | jq
```

```json
{
  "document_name": "your-document.pdf",
  "chunks_ingested": 42,
  "message": "Document ingested successfully"
}
```

Then ask the agent about it:

```bash
curl -s -X POST http://localhost:8000/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What are the key findings from the ingested document?"}' | jq
```

The agent will use the `search_documents` tool to retrieve relevant chunks from the vector store and synthesize an answer.

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check -- database and pgvector status |
| `POST` | `/agent/chat` | Send a message to the agent (triggers tool use) |
| `GET` | `/tools` | List all available tools and descriptions |
| `POST` | `/rag/ingest` | Upload and ingest a document into the vector store |
| `POST` | `/rag/query` | Direct similarity search against the vector store |
| `GET` | `/agent/trace/{run_id}` | Retrieve execution trace for a completed run |

All endpoints return structured JSON. Error responses follow a consistent format:

```json
{
  "error": "error_type",
  "detail": "...",
  "message": "Human-readable description"
}
```

---

## Testing

```bash
# Run unit tests (inside container)
docker-compose exec api pytest

# Run unit tests (local Python env)
pytest

# Run E2E tests against running stack (requires docker-compose up)
bash scripts/test_e2e.sh
```

### Test Coverage

| Suite | Tests | What it covers |
|-------|-------|---------------|
| `tests/test_tools.py` | 6 | Calculator (valid/invalid/edge), Tavily loader |
| `tests/test_agent.py` | 2 | Agent creation, tool integration |
| `tests/test_rag.py` | 3 | Text ingestion, retriever tool creation |
| `scripts/test_e2e.sh` | 7+ | Health, tools, calculator, file reader, RAG, tracing |

---

## Deployment

### Production Configuration

| Variable | Production Value | Notes |
|----------|-----------------|-------|
| `OPENAI_API_KEY` | Your real key | Required |
| `DATABASE_URL` | Production Postgres URL | Use managed Postgres with pgvector |
| `LOG_JSON` | `true` | Structured logs for log aggregation |
| `LOG_LEVEL` | `INFO` or `WARNING` | Reduce noise in production |
| `TAVILY_API_KEY` | Your key | Optional -- agent degrades gracefully without it |

### Cloud Deployment

AgentMCP runs on any Docker-compatible platform:

1. **Build and push the image:**
   ```bash
   docker build -t agent-mcp .
   docker tag agent-mcp your-registry/agent-mcp:latest
   docker push your-registry/agent-mcp:latest
   ```

2. **Set environment variables** on your platform (Railway, Fly.io, AWS ECS, GCP Cloud Run)

3. **Provision PostgreSQL with pgvector** -- most managed Postgres providers support pgvector as an extension

4. **Expose port 8000** and point your domain/load balancer to it

---

## Tech Stack

| Technology | Purpose |
|------------|---------|
| Python 3.12 | Runtime |
| LangGraph | ReAct agent orchestration |
| FastAPI | REST API framework |
| PostgreSQL 17 + pgvector | Vector store and database |
| langchain-mcp-adapters | MCP-to-LangGraph tool bridge |
| MCP SDK (stdio) | File reader server |
| Tavily | Web search tool |
| structlog | Structured JSON logging |
| Docker Compose | Single-command deployment |

---

## Project Structure

```
agent-mcp/
├── src/
│   ├── main.py                  # FastAPI app, lifespan, error handlers
│   ├── agent/
│   │   ├── graph.py             # LangGraph ReAct agent factory
│   │   └── tools.py             # Native tools (calculator, web search)
│   ├── api/
│   │   ├── routes_agent.py      # POST /agent/chat, GET /agent/trace
│   │   ├── routes_health.py     # GET /health
│   │   ├── routes_rag.py        # POST /rag/ingest, POST /rag/query
│   │   └── routes_tools.py      # GET /tools
│   ├── core/
│   │   ├── config.py            # Pydantic settings (env var loading)
│   │   └── logging.py           # structlog configuration
│   ├── mcp_servers/
│   │   └── file_reader.py       # MCP file reader server (stdio)
│   ├── rag/
│   │   ├── ingestion.py         # Document chunking and embedding
│   │   ├── retriever_tool.py    # RAG as a LangGraph tool
│   │   └── vectorstore.py       # pgvector initialization
│   └── schemas/
│       └── api.py               # Pydantic request/response models
├── tests/
│   ├── conftest.py              # Shared test fixtures
│   ├── test_tools.py            # Calculator and tool loader tests
│   ├── test_agent.py            # Agent creation and routing tests
│   └── test_rag.py              # RAG ingestion and retrieval tests
├── scripts/
│   ├── init-db.sh               # PostgreSQL pgvector extension setup
│   └── test_e2e.sh              # curl-based E2E test suite
├── data/
│   ├── sample.txt               # Demo file for MCP file reader
│   └── ieee_paper.txt           # IEEE paper for RAG demo
├── docker-compose.yml           # FastAPI + PostgreSQL/pgvector
├── Dockerfile                   # Python 3.12-slim based image
├── .env.example                 # Environment variable template
└── pyproject.toml               # Dependencies and project metadata
```

---

## License

MIT
