import sys
import logging
from pathlib import Path

from mcp.server.fastmcp import FastMCP

logging.basicConfig(stream=sys.stderr, level=logging.INFO)

mcp = FastMCP("file_reader")

ALLOWED_DIR = Path(__file__).resolve().parent.parent.parent / "data"


@mcp.tool()
def read_file(file_path: str) -> str:
    """Read the contents of a text file from the data directory. Use this tool when the user wants to read, view, or examine the contents of a local file. The file must be within the allowed data directory."""
    try:
        target = (ALLOWED_DIR / file_path).resolve()
        if not str(target).startswith(str(ALLOWED_DIR)):
            return f"Error: Access denied. Path traversal detected. Files must be within {ALLOWED_DIR}"
        if not target.exists():
            return f"Error: File not found: {file_path}"
        if not target.is_file():
            return f"Error: Not a file: {file_path}"
        return target.read_text(encoding="utf-8")
    except Exception as e:
        return f"Error reading file: {e}"


@mcp.tool()
def list_files() -> str:
    """List all available files in the data directory. Use this tool to see what files are available before reading them."""
    try:
        if not ALLOWED_DIR.exists():
            return "Error: Data directory does not exist."
        files = sorted(f.name for f in ALLOWED_DIR.iterdir() if f.is_file())
        if not files:
            return "No files found in data directory."
        return "\n".join(files)
    except Exception as e:
        return f"Error listing files: {e}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
