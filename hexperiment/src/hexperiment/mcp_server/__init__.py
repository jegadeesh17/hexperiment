"""The MCP server of Hexperiment (an optional part of the package).

``main`` is the console command ``hexperiment-mcp``. It imports the server module
only when it runs, so ``import hexperiment`` never loads the MCP SDK (Requirement
4.1). The tool logic is in ``service`` and needs no SDK.
"""

import importlib
import sys

from hexperiment.console import print_text
from hexperiment.errors import HexInputError

SERVER_MODULE = "hexperiment.mcp_server.server"
SDK_MODULES = frozenset({"mcp", "anyio"})
EXIT_FAILED = 2
MISSING_SDK_MESSAGE = (
    "hexperiment-mcp: the MCP SDK is not installed. "
    'Install it with: pip install "hexperiment[mcp]"'
)


def main() -> None:
    """Serve the workspace over stdio; exit with code 2 when that is impossible."""
    try:
        server = importlib.import_module(SERVER_MODULE)
    except ModuleNotFoundError as error:
        if (error.name or "").split(".")[0] not in SDK_MODULES:
            raise
        print_text(MISSING_SDK_MESSAGE, sys.stderr)
        raise SystemExit(EXIT_FAILED) from None
    try:
        server.serve()
    except HexInputError as error:
        print_text(f"hexperiment-mcp: {error}", sys.stderr)
        raise SystemExit(EXIT_FAILED) from None
