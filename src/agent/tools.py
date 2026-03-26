import io
import builtins
import threading
import traceback
from contextlib import redirect_stdout, redirect_stderr

import structlog
from langchain_core.tools import tool
from simpleeval import simple_eval

from src.core.config import Settings

logger = structlog.get_logger()


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression and return the numeric result. Use this tool for any math calculations, arithmetic, unit conversions, or numeric computations. Examples: '2 + 2', '15 * 3.14', '(100 - 32) * 5/9', '2 ** 10'"""
    try:
        result = simple_eval(expression)
        return str(result)
    except Exception as e:
        return f"Error evaluating expression '{expression}': {e}"


# Sandboxed Python executor - allowed standard library modules only
_ALLOWED_MODULES = frozenset({
    "math", "random", "statistics", "datetime", "json", "re", "string",
    "collections", "itertools", "functools", "operator", "textwrap",
    "hashlib", "hmac", "base64", "binascii", "struct", "uuid",
    "decimal", "fractions", "csv", "io", "urllib.parse",
})

_SAFE_BUILTIN_NAMES = [
    "abs", "all", "any", "bin", "bool", "bytes", "callable", "chr",
    "complex", "dict", "dir", "divmod", "enumerate", "filter", "float",
    "format", "frozenset", "getattr", "hasattr", "hash", "hex", "id",
    "int", "isinstance", "issubclass", "iter", "len", "list", "map",
    "max", "min", "next", "object", "oct", "ord", "pow", "print",
    "range", "repr", "reversed", "round", "set", "slice", "sorted",
    "str", "sum", "tuple", "type", "vars", "zip",
]

_original_import = builtins.__import__


def _safe_import(name, *args, **kwargs):
    top_level = name.split(".")[0]
    if top_level not in _ALLOWED_MODULES and name not in _ALLOWED_MODULES:
        raise ImportError(
            f"Module '{name}' is not allowed. "
            f"Allowed: {', '.join(sorted(_ALLOWED_MODULES))}"
        )
    return _original_import(name, *args, **kwargs)


def _build_restricted_globals():
    safe_builtins = {k: getattr(builtins, k) for k in _SAFE_BUILTIN_NAMES}
    safe_builtins["__import__"] = _safe_import
    return {"__builtins__": safe_builtins}


@tool
def python_executor(code: str) -> str:
    """Run Python code in a sandboxed environment and return stdout output.
Use this tool when you need to:
- Run calculations too complex for the calculator (statistics, data processing)
- Generate data (random passwords, UUIDs, dates, sequences)
- Process or transform text (regex, parsing, encoding/decoding)
- Work with data structures (sorting, filtering, aggregation)
- Any task requiring actual code execution to produce a result

Only standard library modules are available (math, random, datetime, json, re, hashlib, uuid, etc.).
Dangerous modules (os, subprocess, sys, shutil, socket) are blocked.
10-second timeout. Use print() to produce output.
Example: 'import random; print([random.randint(1,100) for _ in range(5)])'"""

    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    result = {"output": "", "error": ""}
    timed_out = False

    restricted_globals = _build_restricted_globals()

    def _run():
        try:
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                # compile first for better error messages
                compiled = compile(code, "<agent_code>", "exec")
                exec(compiled, restricted_globals)  # noqa: S102 - intentional sandboxed exec

            result["output"] = stdout_buf.getvalue()
            result["error"] = stderr_buf.getvalue()

            # REPL-style: if no stdout, try evaluating last line as expression
            if not result["output"].strip() and not result["error"]:
                lines = code.strip().splitlines()
                if lines:
                    try:
                        val = eval(lines[-1], restricted_globals)  # noqa: S307
                        if val is not None:
                            result["output"] = repr(val)
                    except Exception:
                        pass
        except Exception:
            result["error"] = traceback.format_exc()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=10)

    if thread.is_alive():
        return "Error: Code execution timed out after 10 seconds."

    output = ""
    if result["output"]:
        output += result["output"].strip()
    if result["error"]:
        err = result["error"].strip()
        output = f"{output}\n\nSTDERR:\n{err}" if output else f"Error:\n{err}"
    if not output:
        output = "(No output produced. Use print() to see results.)"

    # Truncate very long output
    if len(output) > 5000:
        output = output[:5000] + "\n... (output truncated at 5000 chars)"

    logger.info("python_executor_run", code_length=len(code),
                output_length=len(output), had_error=bool(result["error"]))
    return output


@tool
def fetch_url(url: str) -> str:
    """Fetch the text content of a web page at the given URL. Use this tool when the user asks you to:
- Read, summarize, or analyze a specific web page or article
- Check what's on a particular URL (GitHub README, blog post, news article, documentation)
- Extract information from a specific link the user provides

This is different from web search - use this when you have an exact URL to fetch.
Returns the page text content (HTML tags stripped), truncated to 10000 chars.
Example: fetch_url('https://example.com/article')"""
    import re
    import urllib.request
    import urllib.error

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "AgentMCP/0.1"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read(500_000)  # cap at 500KB
            html = raw.decode("utf-8", errors="replace")

        # Strip script/style tags and their content
        html = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
        # Strip all remaining HTML tags
        text = re.sub(r"<[^>]+>", " ", html)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()

        if not text:
            return "Page fetched but no text content found."

        if len(text) > 10000:
            text = text[:10000] + "\n... (truncated at 10000 chars)"

        logger.info("url_fetched", url=url, content_length=len(text))
        return text

    except urllib.error.HTTPError as e:
        return f"HTTP error {e.code}: {e.reason} for URL: {url}"
    except urllib.error.URLError as e:
        return f"URL error: {e.reason} for URL: {url}"
    except Exception as e:
        return f"Error fetching URL: {e}"


def get_native_tools(settings: Settings) -> list:
    """Return list of native LangGraph tools based on available configuration."""
    tools = [calculator, python_executor, fetch_url]

    if settings.tavily_api_key:
        try:
            from langchain_tavily import TavilySearch

            tavily = TavilySearch(max_results=3, topic="general")
            tools.append(tavily)
            logger.info("tavily_enabled")
        except Exception as e:
            logger.warning("tavily_init_failed", error=str(e))
    else:
        logger.warning("tavily_disabled", reason="tavily_api_key not set")

    return tools
