# Decisions

Dependencies and other choices that the code alone does not explain. The core package's runtime dependencies are `pandas`, `numpy` and `scikit-learn`; anything else is recorded here.

## ADR-0001: Run the test suite in parallel with pytest-xdist

- **Status:** accepted, 2026-11-11.
- **Context:** the canary scenario tests start real `train.py` executions. Each one costs 8 to 9 seconds of interpreter and scikit-learn start-up on the reference laptop, so the suite takes about 330 seconds when run serially, against a 90-second budget.
- **Decision:** add `pytest-xdist==3.8.0` to the `dev` extra and run the suite with `-n auto --dist loadscope` (in `pyproject.toml`). Each scenario is one test class, so `loadscope` runs its fixture once on one worker while the classes spread across CPU cores.
- **Consequences:** about 70 seconds on 8 logical cores. Debugging one test needs `-n 0`. Machines with few cores may exceed the budget. Dev-only; no runtime effect.

## ADR-0002: Optional MCP server dependency, pinned

- **Status:** accepted, 2026-11-11.
- **Context:** the Kiro power exposes the kit to agents through an MCP server over stdio.
- **Decision:** use the official MCP Python SDK, `mcp==2.3.0`, as the optional extra `hexperiment[mcp]`. Only the server module imports it, and `import hexperiment` never loads it.
- **Why pinned:** version 2.x renamed `FastMCP` to `MCPServer` and changed other APIs, so a floating version could break the server.
- **Consequences:** the core keeps three runtime dependencies; users who do not need the MCP server never install the SDK.
