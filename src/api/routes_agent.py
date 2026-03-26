import time
from uuid import uuid4

import structlog
from fastapi import APIRouter, HTTPException, Request
from langchain_core.messages import HumanMessage

from src.agent.tracing import AgentTracer
from src.schemas.api import AgentRequest, AgentResponse

router = APIRouter(tags=["agent"])
logger = structlog.stdlib.get_logger()


@router.post("/agent/chat", response_model=AgentResponse)
async def agent_chat(request: Request, body: AgentRequest) -> AgentResponse:
    agent = getattr(request.app.state, "agent", None)
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    run_id = str(uuid4())
    tracer = AgentTracer(run_id)

    start = time.time()
    try:
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=body.message)]},
            config={
                "callbacks": [tracer],
                "configurable": {"thread_id": body.thread_id or "default"},
            },
        )

        # Finalize trace (stores in trace_store)
        tracer.finalize()

        # Extract final AI message
        ai_message = result["messages"][-1]

        # Extract tools used from intermediate messages
        tools_used = []
        for msg in result["messages"]:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                tools_used.extend([tc["name"] for tc in msg.tool_calls])

        duration_ms = int((time.time() - start) * 1000)
        logger.info("agent_chat_completed", duration_ms=duration_ms,
                     tools_used=tools_used, run_id=run_id)

        return AgentResponse(
            response=ai_message.content,
            tools_used=tools_used,
            metadata={
                "duration_ms": duration_ms,
                "run_id": run_id,
                "token_usage": tracer.token_usage,
                "cost_estimate": tracer.cost_estimate,
            },
        )
    except Exception as e:
        logger.exception("agent_chat_error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")
