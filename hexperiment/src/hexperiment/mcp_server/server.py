"""The MCP server: five tools and the model cards as resources (Requirement 4).

This is the only module of the package that imports the MCP SDK and ``anyio``
(Requirement 4.1). It adds no logic: every tool calls a function of ``service`` and
turns ``HexInputError`` into the SDK's error type, so that the client sees an error
result that names the problem (Requirement 4.5). Standard output carries the
protocol, so nothing here prints.
"""

from collections.abc import Callable
from importlib.metadata import version as installed_version
from pathlib import Path
from typing import Any

import anyio
import anyio.to_thread
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ResourceError, ToolError

from hexperiment.cli import run_app
from hexperiment.errors import HexInputError
from hexperiment.mcp_server import service
from hexperiment.workspace import RunApp, discover_apps, resolve_workspace

SERVER_NAME = "hexperiment"
PACKAGE_NAME = "hexperiment"
MARKDOWN = "text/markdown"
INSTRUCTIONS = (
    "Hexperiment audits ML training scripts for data leakage. Apps are the folders "
    "of the workspace that hold hex.toml and train.py; call list_apps first, it "
    "reports the workspace this server uses. run_audit is the only tool that runs "
    "code; the others only read. A verdict is APPROVED only after a full run: "
    "mode 'fast' skips the canary test and can never approve. get_verdict shows "
    "the current verdict, get_history the earlier runs, and explain_ghost says "
    "what a finding means and how to fix it. Never edit verdict.json, a model "
    "card, hex.toml thresholds or waivers by hand."
)


def build_server(workspace: Path, *, run: RunApp = run_app) -> MCPServer:
    """A server for ``workspace`` whose audits go through ``run``.

    The server, the lock and the registrations are created here, so the module
    holds no state. At most one audit runs at a time (Requirement 4.6).
    """
    root = workspace.resolve()
    server = MCPServer(
        name=SERVER_NAME, instructions=INSTRUCTIONS, version=_package_version()
    )
    lock = anyio.Lock()

    @server.tool()
    def list_apps() -> dict[str, Any]:
        """List every app in the workspace with its name, path and current status.

        The status is the status in the app's verdict.json (APPROVED, REJECTED or
        UNVERIFIED), or "none" when the app has no current verdict. The result
        also names the workspace this server uses.
        """
        return _refuse(lambda: service.list_apps(root))

    @server.tool()
    async def run_audit(app: str, mode: str = "full") -> dict[str, Any]:
        """Run `hexperiment run` on an app and return its status and summary.

        This is the only tool that executes code (the app's train.py). Mode
        "full" (the default) includes the canary test and is the only mode that
        can give APPROVED; mode "fast" skips the canary and can give at best
        UNVERIFIED. `app` is the app folder relative to the workspace, for
        example apps/churn. A run that fails is a normal result with status
        "none", exit_code 2 and the failure message in summary.
        """
        _refuse(lambda: service.check_audit_request(root, app, mode))
        async with lock:
            return await anyio.to_thread.run_sync(
                lambda: _refuse(lambda: service.run_audit(root, app, mode, run))
            )

    @server.tool()
    def get_verdict(app: str) -> dict[str, Any]:
        """Return the app's current verdict.json, or say that it has none.

        The file exists only after a run has finished; every run deletes it
        first. `app` is the app folder relative to the workspace.
        """
        return _refuse(lambda: service.get_verdict(root, app))

    @server.tool()
    def get_history(app: str) -> dict[str, Any]:
        """List the earlier runs of an app, oldest first.

        Each entry has the run id, the status, the metric value and the names of
        the ghosts found. `app` is the app folder relative to the workspace.
        """
        return _refuse(lambda: service.get_history(root, app))

    @server.tool()
    def explain_ghost(name: str) -> dict[str, Any]:
        """Explain a ghost or a check: what it means, why it matters, how to fix it.

        `name` is a ghost name such as "The Possession" or a check name such as
        "target_proxy", as they appear in the findings of a verdict.
        """
        return _refuse(lambda: service.explain_ghost(name))

    @server.resource(
        service.RESOURCE_PREFIX + "{app}",
        name="model-card",
        description="The model card of the newest run of an app. The app is the "
        "folder relative to the workspace, with each / written as %2F.",
        mime_type=MARKDOWN,
    )
    def model_card_of(app: str) -> str:
        # The SDK has already decoded the percent escapes of the URI.
        return _card(root, app)

    for entry in discover_apps(root):
        server.resource(
            service.resource_uri(entry.relative),
            name=f"model card of {entry.relative}",
            description=f"The model card of the newest run of {entry.name}.",
            mime_type=MARKDOWN,
        )(_card_reader(root, entry.relative))

    return server


def serve() -> None:
    """Serve the workspace over standard input and output until the client leaves."""
    build_server(resolve_workspace()).run()


# --- helpers -------------------------------------------------------------------------


def _refuse[T](call: Callable[[], T]) -> T:
    """Run ``call``; a ``HexInputError`` becomes a ``ToolError`` (Requirement 4.5)."""
    try:
        return call()
    except HexInputError as error:
        raise ToolError(str(error)) from None


def _card(root: Path, app: str) -> str:
    try:
        return service.model_card(root, app)
    except HexInputError as error:
        raise ResourceError(str(error)) from None


def _card_reader(root: Path, app: str) -> Callable[[], str]:
    def read() -> str:
        return _card(root, app)

    return read


def _package_version() -> str:
    return installed_version(PACKAGE_NAME)
