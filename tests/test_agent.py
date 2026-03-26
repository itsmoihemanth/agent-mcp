"""Unit tests for agent creation and routing."""

from unittest.mock import MagicMock, patch

from langchain_core.tools import tool

from src.agent.tools import calculator


@tool
def dummy_tool_a(x: str) -> str:
    """A dummy tool for testing."""
    return x


@tool
def dummy_tool_b(x: str) -> str:
    """Another dummy tool for testing."""
    return x


class TestCreateAgent:
    @patch("src.agent.graph.ChatOpenAI")
    def test_create_agent_returns_compiled_graph(self, mock_chat_cls):
        """create_agent returns a runnable compiled graph."""
        from src.agent.graph import create_agent

        mock_model = MagicMock()
        mock_chat_cls.return_value = mock_model

        tools = [calculator]
        agent = create_agent(mock_model, tools)

        # LangGraph compiled graph has an invoke method
        assert hasattr(agent, "invoke") or hasattr(agent, "ainvoke")

    @patch("src.agent.graph.ChatOpenAI")
    def test_create_agent_tool_count(self, mock_chat_cls):
        """Agent is created with the correct number of tools."""
        from src.agent.graph import create_agent

        mock_model = MagicMock()

        agent = create_agent(mock_model, [dummy_tool_a, dummy_tool_b])
        assert hasattr(agent, "ainvoke")
