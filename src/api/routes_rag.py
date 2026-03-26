import io

import structlog
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from pypdf import PdfReader

from src.rag.ingestion import ingest_text
from src.schemas.api import (
    ChunkResult,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)

router = APIRouter(prefix="/rag", tags=["rag"])
logger = structlog.stdlib.get_logger()

ALLOWED_EXTENSIONS = {".txt", ".pdf"}


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(request: Request, file: UploadFile = File(...)):
    vectorstore = getattr(request.app.state, "vectorstore", None)
    if vectorstore is None:
        raise HTTPException(status_code=503, detail="Vectorstore not initialized")

    # Validate file extension
    filename = file.filename or "unknown"
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Allowed: .txt, .pdf",
        )

    content = await file.read()
    file_size = len(content)

    if ext == ".txt":
        text = content.decode("utf-8")
    elif ext == ".pdf":
        reader = PdfReader(io.BytesIO(content))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)

    count = await ingest_text(vectorstore, text, filename)

    logger.info(
        "document_ingested",
        document_name=filename,
        file_size=file_size,
        chunks_ingested=count,
    )

    return IngestResponse(document_name=filename, chunks_ingested=count)


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: Request, body: QueryRequest):
    vectorstore = getattr(request.app.state, "vectorstore", None)
    if vectorstore is None:
        raise HTTPException(status_code=503, detail="Vectorstore not initialized")

    results = await vectorstore.asimilarity_search_with_score(body.query, k=body.top_k)

    chunks = [
        ChunkResult(
            document_name=doc.metadata.get("document_name", "unknown"),
            chunk_text=doc.page_content,
            similarity_score=round(score, 4),
        )
        for doc, score in results
    ]

    logger.info(
        "rag_query_completed",
        query=body.query[:100],
        result_count=len(chunks),
    )

    return QueryResponse(query=body.query, results=chunks, count=len(chunks))
