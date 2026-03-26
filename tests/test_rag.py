"""Unit tests for RAG ingestion and retriever tool."""

import pytest

from src.rag.ingestion import ingest_text
from src.rag.retriever_tool import make_retriever_tool


class TestIngestText:
    @pytest.mark.asyncio
    async def test_ingest_text_chunks(self, mock_vectorstore):
        """ingest_text splits text and calls aadd_documents."""
        text = "Hello world. " * 200  # Enough text for multiple chunks
        count = await ingest_text(mock_vectorstore, text, "test_doc.txt")

        assert count > 0
        mock_vectorstore.aadd_documents.assert_called_once()
        docs = mock_vectorstore.aadd_documents.call_args[0][0]
        assert len(docs) == count

    @pytest.mark.asyncio
    async def test_ingest_text_metadata(self, mock_vectorstore):
        """Ingested docs have document_name and chunk_index in metadata."""
        text = "Some content for testing purposes. " * 100
        await ingest_text(mock_vectorstore, text, "paper.pdf")

        docs = mock_vectorstore.aadd_documents.call_args[0][0]
        for i, doc in enumerate(docs):
            assert doc.metadata["document_name"] == "paper.pdf"
            assert doc.metadata["chunk_index"] == i


class TestMakeRetrieverTool:
    def test_make_retriever_tool(self, mock_vectorstore):
        """make_retriever_tool returns a tool named search_documents."""
        tool = make_retriever_tool(mock_vectorstore)
        assert tool.name == "search_documents"
