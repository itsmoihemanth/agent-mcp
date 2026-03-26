from fastapi import APIRouter, Request

from src.schemas.api import ToolInfo

router = APIRouter(tags=["tools"])


@router.get("/tools", response_model=list[ToolInfo])
async def list_tools(request: Request) -> list[ToolInfo]:
    tools = getattr(request.app.state, "tools", None)
    if not tools:
        return []
    return [ToolInfo(name=t.name, description=t.description) for t in tools]
