import structlog
from fastapi import APIRouter, HTTPException

from src.agent.tracing import trace_store
from src.schemas.api import TraceResponse

router = APIRouter(tags=["trace"])
logger = structlog.stdlib.get_logger()


@router.get("/agent/trace/{run_id}", response_model=TraceResponse)
async def get_trace(run_id: str) -> TraceResponse:
    """Retrieve the full execution trace for a previous agent run."""
    trace = trace_store.get(run_id)
    if trace is None:
        raise HTTPException(status_code=404, detail=f"Trace not found for run_id: {run_id}")

    logger.info("trace_retrieved", run_id=run_id, steps=len(trace.get("steps", [])))
    return TraceResponse(**trace)
