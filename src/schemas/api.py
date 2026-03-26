from typing import Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., examples=["healthy"])
    database: dict = Field(..., examples=[{"connected": True, "error": None}])
    pgvector: dict = Field(..., examples=[{"installed": True}])


class ErrorResponse(BaseModel):
    error: str = Field(..., examples=["validation_error"])
    detail: object = Field(None, examples=["Invalid request body"])
    message: str = Field(..., examples=["Request validation failed"])


class AgentRequest(BaseModel):
    message: str = Field(..., min_length=1, examples=["What is 2 + 2?"])
    thread_id: Optional[str] = Field(None, examples=["thread-123"])


class AgentResponse(BaseModel):
    response: str = Field(..., examples=["The answer is 4."])
    tools_used: list[str] = Field(default_factory=list, examples=[["calculator"]])
    metadata: dict = Field(default_factory=dict, examples=[{"duration_ms": 150}])


class ToolInfo(BaseModel):
    name: str = Field(..., examples=["calculator"])
    description: str = Field(..., examples=["Evaluate a mathematical expression"])


class QueryRequest(BaseModel):
    query: str = Field(
        ..., min_length=1, examples=["What does the paper say about transformers?"]
    )
    top_k: int = Field(default=5, ge=1, le=20)


class ChunkResult(BaseModel):
    document_name: str = Field(..., examples=["ieee_paper.txt"])
    chunk_text: str = Field(..., examples=["Transformers use self-attention..."])
    similarity_score: float = Field(..., examples=[0.8723])


class QueryResponse(BaseModel):
    query: str
    results: list[ChunkResult]
    count: int


class IngestResponse(BaseModel):
    document_name: str
    chunks_ingested: int
    message: str = Field(default="Document ingested successfully")


class TraceStep(BaseModel):
    node: str = Field(..., examples=["agent"])
    tool_name: Optional[str] = Field(None, examples=["calculator"])
    input: str = Field(..., examples=["What is 2+2?"])
    output: str = Field(..., examples=["4"])
    latency_ms: int = Field(..., examples=[120])
    timestamp: str = Field(..., examples=["2026-03-25T12:00:00Z"])


class TraceResponse(BaseModel):
    run_id: str = Field(..., examples=["550e8400-e29b-41d4-a716-446655440000"])
    steps: list[TraceStep] = Field(default_factory=list)
    total_duration_ms: int = Field(..., examples=[1500])
    token_usage: dict = Field(default_factory=dict, examples=[{"prompt_tokens": 100, "completion_tokens": 50}])
    cost_estimate: float = Field(..., examples=[0.000045])
