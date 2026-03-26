import structlog
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGEngine, PGVectorStore
from langchain_postgres.v2.async_vectorstore import DistanceStrategy

logger = structlog.get_logger()

COLLECTION_NAME = "documents"
VECTOR_SIZE = 1536  # text-embedding-3-small


async def init_vectorstore(database_url: str, openai_api_key: str, openai_api_base: str = "") -> PGVectorStore:
    """Initialize PGEngine, create table if needed, return PGVectorStore."""
    logger.info("vectorstore_init_start", collection=COLLECTION_NAME)

    engine = PGEngine.from_connection_string(url=database_url)

    try:
        await engine.ainit_vectorstore_table(
            table_name=COLLECTION_NAME,
            vector_size=VECTOR_SIZE,
        )
        logger.info("vectorstore_table_created", table=COLLECTION_NAME)
    except Exception:
        logger.info("vectorstore_table_exists", table=COLLECTION_NAME)

    embed_kwargs = {"model": "text-embedding-3-small", "api_key": openai_api_key}
    if openai_api_base:
        embed_kwargs["base_url"] = openai_api_base
    embeddings = OpenAIEmbeddings(**embed_kwargs)

    store = await PGVectorStore.create(
        engine=engine,
        table_name=COLLECTION_NAME,
        embedding_service=embeddings,
        distance_strategy=DistanceStrategy.COSINE_DISTANCE,
    )

    logger.info("vectorstore_init_complete", collection=COLLECTION_NAME)
    return store
