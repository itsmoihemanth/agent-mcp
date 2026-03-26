import psycopg
import structlog
from fastapi import APIRouter
from src.core.config import get_settings
from src.schemas.api import HealthResponse

router = APIRouter()
logger = structlog.stdlib.get_logger()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    settings = get_settings()
    db_ok = False
    pgvector_ok = False
    db_error = None

    try:
        # Use raw psycopg3 async connection -- strip +psycopg from URL
        conninfo = settings.database_url.replace("postgresql+psycopg://", "postgresql://")
        async with await psycopg.AsyncConnection.connect(conninfo) as conn:
            result = await conn.execute("SELECT 1")
            db_ok = True

            result = await conn.execute(
                "SELECT installed_version FROM pg_available_extensions WHERE name = 'vector'"
            )
            row = await result.fetchone()
            pgvector_ok = row is not None and row[0] is not None
    except Exception as e:
        db_error = str(e)
        logger.error("health_check_failed", error=db_error)

    status = "healthy" if (db_ok and pgvector_ok) else "unhealthy"
    return HealthResponse(
        status=status,
        database={"connected": db_ok, "error": db_error},
        pgvector={"installed": pgvector_ok},
    )
