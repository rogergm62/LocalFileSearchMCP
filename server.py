from pathlib import Path
from mcp.server import MCPServer

# Create our MCP server
mcp = MCPServer("Local File Search")

# Define the folder we are allowed to search
BASE_DIR = Path(__file__).resolve().parent / "test_files"

@mcp.tool()
def list_files() -> list[str]:
    """List all filenames in the test_files directory.
Use this tool when the user asks which files exist
or requests a complete directory listing.
"""

    files = []

    for file in BASE_DIR.iterdir():
        if file.is_file():
            files.append(file.name)

    return files

@mcp.tool()
def search_files(keyword: str) -> list[str]:

    """Search filenames for a keyword.
Use this tool only when searching file NAMES,
not when searching the contents of files.
"""
    matches = []

    for file in BASE_DIR.iterdir():
        if file.is_file() and keyword.lower() in file.name.lower():
            matches.append(file.name)

    return matches

@mcp.tool()
def search_content(keyword: str) -> list[str]:
    """Search inside text files for a keyword.
Returns matching filenames, line numbers, and text.
Use when the user asks which files contain a word
or phrase in their contents.
"""
    matches = []

    for file in BASE_DIR.iterdir():
        if file.is_file() and file.suffix in [".txt", ".py", ".md"]:

            try:
                content = file.read_text(encoding="utf-8")

                for line_number, line in enumerate(content.splitlines(), start=1):
                    if keyword.lower() in line.lower():
                        matches.append(f"{file.name} (Line {line_number}): {line.strip()}")

            except (UnicodeDecodeError, OSError):
                matches.append(f"Could not read: {file.name}")

    return matches

if __name__ == "__main__":
    mcp.run()
