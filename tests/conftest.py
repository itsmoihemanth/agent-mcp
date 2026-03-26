"""Shared test fixtures for agent-mcp test suite."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.config import Settings


@pytest.fixture
def mock_settings():
    """Settings with dummy keys -- no Tavily."""
    return Settings(
        openai_api_key="test-openai-key",
        tavily_api_key="",
        database_url="postgresql+psycopg://test:test@localhost/test",
    )


@pytest.fixture
def mock_settings_with_tavily():
    """Settings with Tavily key set."""
    return Settings(
        openai_api_key="test-openai-key",
        tavily_api_key="test-tavily-key",
        database_url="postgresql+psycopg://test:test@localhost/test",
    )


@pytest.fixture
def mock_vectorstore():
    """AsyncMock vectorstore with aadd_documents and as_retriever."""
    store = AsyncMock()
    store.aadd_documents = AsyncMock()

    # as_retriever returns a sync mock retriever
    mock_retriever = MagicMock()
    store.as_retriever = MagicMock(return_value=mock_retriever)

    return store
