import uuid

import structlog
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = structlog.get_logger()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)


async def ingest_text(vectorstore, text: str, document_name: str) -> int:
    """Chunk text and store embeddings. Returns chunk count."""
    chunks = text_splitter.split_text(text)
    docs = [
        Document(
            id=str(uuid.uuid4()),
            page_content=chunk,
            metadata={
                "document_name": document_name,
                "chunk_index": i,
            },
        )
        for i, chunk in enumerate(chunks)
    ]

    await vectorstore.aadd_documents(docs)

    logger.info(
        "ingestion_complete",
        document_name=document_name,
        chunk_count=len(docs),
    )
    return len(docs)
