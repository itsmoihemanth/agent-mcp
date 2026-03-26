"""Unit tests for agent tools: calculator and get_native_tools."""

from unittest.mock import patch, MagicMock

from src.agent.tools import calculator, get_native_tools


class TestCalculator:
    def test_calculator_addition(self):
        result = calculator.invoke("2 + 2")
        assert result == "4"

    def test_calculator_complex(self):
        result = calculator.invoke("(100 - 32) * 5/9")
        expected = (100 - 32) * 5 / 9
        assert float(result) == pytest.approx(expected)

    def test_calculator_invalid(self):
        result = calculator.invoke("invalid")
        assert "Error" in result

    def test_calculator_division(self):
        result = calculator.invoke("10 / 3")
        assert float(result) == pytest.approx(10 / 3)


class TestGetNativeTools:
    def test_get_native_tools_no_tavily(self, mock_settings):
        tools = get_native_tools(mock_settings)
        assert len(tools) == 1
        assert tools[0].name == "calculator"

    def test_get_native_tools_with_tavily(self, mock_settings_with_tavily):
        mock_tavily_cls = MagicMock()
        mock_tavily_instance = MagicMock()
        mock_tavily_cls.return_value = mock_tavily_instance

        with patch.dict(
            "sys.modules",
            {"langchain_tavily": MagicMock(TavilySearch=mock_tavily_cls)},
        ):
            tools = get_native_tools(mock_settings_with_tavily)
        assert len(tools) == 2


# Need pytest import for approx
import pytest
