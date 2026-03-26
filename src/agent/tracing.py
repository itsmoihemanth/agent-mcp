"""Agent execution tracing via LangGraph callbacks.

Captures node execution, tool calls, and token usage for each agent run.
Stores traces in-memory keyed by run_id for retrieval via the trace endpoint.
"""

import time
from datetime import datetime, timezone
from typing import Any

import structlog
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

logger = structlog.stdlib.get_logger()

# In-memory trace storage keyed by run_id (no persistence needed for demo)
trace_store: dict[str, dict] = {}

# GPT-4o-mini pricing per 1M tokens
_PROMPT_COST_PER_M = 0.15
_COMPLETION_COST_PER_M = 0.60


class AgentTracer(BaseCallbackHandler):
    """LangGraph callback handler for step tracing and token tracking."""

    def __init__(self, run_id: str) -> None:
        super().__init__()
        self.run_id = run_id
        self.steps: list[dict] = []
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self._start_time = time.time()
        self._node_starts: dict[str, float] = {}
        self._tool_starts: dict[str, dict] = {}

    def on_chain_start(
        self, serialized: dict[str, Any], inputs: dict[str, Any], **kwargs: Any
    ) -> None:
        run_id = str(kwargs.get("run_id", ""))
        name = (kwargs.get("name") or serialized.get("name", "")).lower()
        # Only track the "agent" node (LLM reasoning step) — skip wrappers
        if name == "agent":
            self._node_starts[run_id] = time.time()

    def on_chain_end(self, outputs: dict[str, Any], **kwargs: Any) -> None:
        run_id = str(kwargs.get("run_id", ""))
        start = self._node_starts.pop(run_id, None)
        if start is None:
            return
        latency_ms = int((time.time() - start) * 1000)

        # Extract what the LLM decided from the output
        output_str = ""
        messages = outputs.get("messages", [])
        if messages:
            last = messages[-1] if isinstance(messages, list) else messages
            if hasattr(last, "content") and last.content:
                output_str = last.content
            elif hasattr(last, "tool_calls") and last.tool_calls:
                output_str = "-> " + ", ".join(tc["name"] for tc in last.tool_calls)

        step = {
            "node": "llm",
            "tool_name": None,
            "input": "",
            "output": _truncate(output_str),
            "latency_ms": latency_ms,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.steps.append(step)

    def on_tool_start(
        self, serialized: dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        run_id = str(kwargs.get("run_id", ""))
        name = kwargs.get("name") or serialized.get("name", "tool")
        self._tool_starts[run_id] = {"name": name, "input": input_str, "start": time.time()}
        logger.debug("trace_tool_start", run_id=self.run_id, tool=name)

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        run_id = str(kwargs.get("run_id", ""))
        tool_info = self._tool_starts.pop(run_id, None)
        if tool_info is None:
            return
        latency_ms = int((time.time() - tool_info["start"]) * 1000)
        step = {
            "node": "tool",
            "tool_name": tool_info["name"],
            "input": _truncate(tool_info["input"]),
            "output": _truncate(str(output)),
            "latency_ms": latency_ms,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.steps.append(step)
        logger.info(
            "trace_tool_step",
            run_id=self.run_id,
            tool=tool_info["name"],
            latency_ms=latency_ms,
        )

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Extract and accumulate token usage from LLM response."""
        llm_output = response.llm_output or {}
        usage = llm_output.get("token_usage", {})
        self.prompt_tokens += usage.get("prompt_tokens", 0)
        self.completion_tokens += usage.get("completion_tokens", 0)
        logger.debug(
            "trace_llm_tokens",
            run_id=self.run_id,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
        )

    @property
    def token_usage(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
        }

    @property
    def cost_estimate(self) -> float:
        return (
            self.prompt_tokens * _PROMPT_COST_PER_M / 1_000_000
            + self.completion_tokens * _COMPLETION_COST_PER_M / 1_000_000
        )

    def finalize(self) -> dict:
        """Store completed trace in trace_store and return trace data."""
        total_duration_ms = int((time.time() - self._start_time) * 1000)
        trace = {
            "run_id": self.run_id,
            "steps": self.steps,
            "total_duration_ms": total_duration_ms,
            "token_usage": self.token_usage,
            "cost_estimate": self.cost_estimate,
        }
        trace_store[self.run_id] = trace
        logger.info(
            "trace_finalized",
            run_id=self.run_id,
            steps=len(self.steps),
            total_duration_ms=total_duration_ms,
            token_usage=self.token_usage,
            cost_estimate=self.cost_estimate,
        )
        return trace


def _truncate(text: str, max_len: int = 500) -> str:
    """Truncate long strings for trace storage."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."
