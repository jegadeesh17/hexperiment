# Kiro formats and behaviour we relied on (checked 2026-11-11 and 2026-11-12)

Notes taken while building Hexperiment's Kiro layer: the documented formats (from kiro.dev, summarised in our words) and what live probes with Kiro CLI 3.2.0 and Kiro IDE 1.4.154 on Windows 11 showed where the docs were silent or differed. Paths are shortened to the project folder.

## Hooks — https://kiro.dev/docs/hooks/ , /docs/hooks/types/ , /docs/hooks/actions/
- File: `.kiro/hooks/<id>.json`, top level `{"version": "v1", "hooks": [ ... ]}`; several hooks per file. CLI V3 also loads `~/.kiro/hooks/`.
- Hook keys: `name` (req), `description`, `trigger` (req, PascalCase event), `matcher` (regex on file path for file events), `action` {`type`: `command` | `agent`, `command` | `prompt`}, `timeout` (s, default 60, 0 = off; ignored for agent), `enabled`, `confirm` (Stop command hooks only).
- Triggers: Prompt Submit, Agent Stop (`Stop`), Session Start/End, Pre/Post Compaction, Pre/Post Tool Use, File Create, File Save, File Delete, Pre/Post Task Execution, Manual (CLI V3). Literal seen: `PostFileSave` (overview example). File triggers fire only on agent changes, not manual editor edits.
- Command action: runs in project root, gets session JSON on STDIN; exit 0 adds stdout to agent context; non-zero sends stderr to the agent. Agent actions cost credits.
- Available in IDE and CLI (file and task triggers: IDE and CLI V3).

## Powers — https://kiro.dev/docs/powers/ , /docs/powers/create/ , /docs/powers/installation/ , /docs/mcp/configuration/
- Current format: Agent Plugins `plugin.json` ($schema https://agent-plugins.org/schemas/1.0.0/plugin.schema.json; `name` kebab-case, `version` semver, `description`, `author` {name, email?, url?}, `keywords` [activation]; optional homepage, repository, license). Legacy `POWER.md` still works (fallback).
- Contents: `plugin.json`, optional `mcp.json`, `skills/<name>/SKILL.md` (+ scripts/, references/). Bundled hooks/steering paths: not documented.
- mcp.json: $schema https://agent-plugins.org/schemas/1.0.0/mcp.schema.json, `mcpServers` {name: {type: "stdio", command, args[], env{}}}; secrets as `${VAR}`. Server names namespaced on install; plugin-format servers are not written to ~/.kiro/settings/mcp.json.
- Install: IDE Powers panel → Add Custom Power → Import from GitHub (public repo) or from a folder. CLI: `kiro-cli powers install <name-or-local-path>` (checked locally, CLI 3.2.0), `kiro-cli powers uninstall <name>`.

## Workflows — https://kiro.dev/docs/workflows/ , /docs/workflows/authoring/
- Enable: Workspace Configuration → Workflows (`kiroAgent.workflows.enabled`).
- Files: `.kiro/workflows/<name>.workflow.json|yaml|yml`; personal in `~/.kiro/workflows/`.
- Root: `name` (req), `description`, `inputs` {name: type hint}, `modelId`, `effortLevel`, `steps` (req).
- Nodes: `step` (id, agent, prompt), `sequence`, `repeat` (id, steps, maxIterations, onMaxIterations abort|continue|pause), `parallel` (branches, joinPolicy all|allSettled|any), `watch`. Limits 50 steps, depth 8.
- Repeat stop: body runs at least once; `stopCondition.fileCheck` {path, jsonPath, value} (missing/invalid file = no match, loop continues; array value = any-of), or `containsText`, `completionSignal`, or `stopWhen` expression. Not both.
- Start: IDE `/workflow-run` or ask in chat; CLI `/workflow run <name> --task "..."`. Validate with the `validate_workflow` tool.

## MCP SDK
- PyPI `mcp` latest 2.3.0 (pip index, 2026-11-11).

## MCP Python SDK 2.3.0 (installed in .venv, read from source 2026-11-11)
- Released 2026-10-02. Pin `mcp==2.3.0` (pulls `mcp-types==2.3.0`, pydantic, anyio, starlette, httpx2, ...; pywin32 on Windows).
- `FastMCP` was renamed in 2.x: `from mcp.server.mcpserver import MCPServer` (`mcp.server.fastmcp` raises ModuleNotFoundError with a migration hint).
- `MCPServer(name=..., instructions=..., version=...)`; `@server.tool(name=None, title=None, description=None, annotations=None, structured_output=None)`; `@server.resource("scheme://{param}", name=..., mime_type=...)` (template when the URI has `{param}`; str return = text); `server.run()` defaults to stdio (`run_stdio_async` exists).
- Tool errors: raise `mcp.server.mcpserver.exceptions.ToolError` → result with `is_error=True` (mcp_types uses snake_case `is_error`).
- Tests: `from mcp import Client`; `async with Client(server) as client:` connects in-process to an `MCPServer`; `await client.call_tool(name, args)`, `list_tools()`, `list_resources()`, `read_resource(uri)`. anyio is a dependency (its pytest plugin gives `@pytest.mark.anyio`).

## Hook events captured live (Kiro CLI 3.2.0, headless V3, 2026-11-11)
- Hooks hot-reload: a hook file added mid-session fired in already-running sessions.
- `PostFileCreate` and `PostFileSave` both exist and both fire when the agent creates a file with fs_write (create first, then save). STDIN:
  `{"session_id": "sess_...", "hook_event_name": "PostFileSave", "cwd": "C:\Users\...\KiroweenHackathonSubmission", "file_path": "c:\Users\...\KiroweenHackathonSubmission\probe-hook\probe.py"}`
  (absolute path, lower-case drive letter, backslashes; `cwd` has an upper-case drive letter).
- `PostToolUse` STDIN: `{"session_id", "hook_event_name": "PostToolUse", "cwd", "tool_name": "fs_write" | "str_replace" | "execute_pwsh", "tool_input": {"path", "text"} | {"path", "oldStr", "newStr", "replace_all"} | {"command", ...}, "tool_response": "..."}`; it fires even when the tool call was refused.
- Environment: `KIRO_SESSION_ID` and other `KIRO_*` variables; the command runs with the project root as working directory.

## Workflow probe (Kiro CLI 3.2.0, headless V3, 2026-11-11)
- Inputs are referenced as bare `{{name}}` (docs: "Use template variables"); `{{inputs.app}}` stays literal. `{{app}}` expands in step prompts AND in `stopCondition.fileCheck.path`; a 2-iteration repeat stopped after iteration 0 (`loop_iteration ... stopConditionMet: true`), run status `completed`, exit 0.
- Needs `chat.enableWorkflows true` (removed again after the probe). Launch: `MSYS_NO_PATHCONV=1 kiro-cli chat --v3 --no-interactive --trust-workspace --model auto --trust-tools=run_workflow,inspect_workflow,read_file,list_directory,fs_write,fs_append,str_replace --output-format stream-json "/workflow run <name> --app <path>"`. Without MSYS_NO_PATHCONV, Git Bash turns `/workflow` into `C:/Program Files/Git/workflow`. The custom builder agent has no `run_workflow` tool; the default agent does.
- Step agents in headless mode only get tools trusted on the CLI; untrusted fs_write → "[denied] tool permission approval is not supported in non-interactive mode".
- A failed step (agent reports it could not do the task) → node `failed` → loop `failed` → run `failed`; no next iteration. onMaxIterations `abort` → run `aborted`, CLI exit 1.
- stream-json emits `workflowEvent` lines (`_kiro/workflow/run_start|node_start|node_complete|loop_iteration|run_complete`) with `capturedOutput` per step. Cost: about 1 credit per probe run on auto.

## Agent permission rules, checked live (Kiro CLI 3.2.0, headless V3, 2026-11-11, throwaway app)
- fs_write globs: case-insensitive; `\` and `/` treated alike (`apps/x\hex.toml`, `HEX.TOML`, `Fetch_Data.py` refused by `**/hex.toml` / `**/fetch_data.py`). `..` NOT resolved: `apps/x/../y.py` matched `apps/**/*.py` and was written. Absolute backslash path inside an app matched `**/apps/**/*.py`. Refusal text: "Blocked by a permission rule (deny fs_write matching "..."); the user did not reject it. Source: agent-profile."
- shell: allow match + no deny → runs. Deny match → "Blocked by a permission rule (deny shell matching ...)". No allow match → approval prompt → in headless "The permission request failed before it was answered, so this call did not run" (ls, whoami, `cmd\nwhoami`, `cmd ; whoami` all ended this way).
- A deny pattern containing a real LF or CR (`"*\n*"`, `"*\r*"`) makes EVERY shell call fail with "permission request failed", allowed ones included. Do not put control characters in patterns.
- `--trust-tools` does not bypass agent permission rules (denies still applied).
- The custom agent needs the CLI's `--trust-tools` for its tools in headless; the probe ran with read_file,list_directory,file_search,grep_search,fs_write,str_replace,fs_append,execute_pwsh.

## Live checks, 2026-11-12 (Kiro CLI 3.2.0, headless V3)

- **Hook output never reaches the agent.** Probe hooks on `PostFileSave` (print at once, print after 12 s, a wrapper logging `hexperiment hook` stdout) and on `PostToolUse` with matcher `write` all ran and completed; `hexperiment hook` printed the right summary (exit 0, 17.7 s). Kiro waits for the hooks before the next model turn, but the stored `ContextualHookInvoked` records carry no output, no `HOOK_INSTRUCTION` block appeared, and the agent saw nothing. The docs say exit-0 stdout is added to the agent's context for every trigger (IDE and CLI). The same was then seen in the Kiro IDE 1.4.154 (see `hook-output-issue.md`).
- **PostToolUse event shape:** `{"session_id", "hook_event_name": "PostToolUse", "cwd", "tool_name": "fs_write", "tool_input": {"path": "c:\...\file.py", "text": "..."}, "tool_response": "Created the ... file."}`.
- **Powers:** `kiro-cli powers install <path>` copies the power to `~/.kiro/powers/installed/<name>`; the server is named `power-<power>-<server>` and starts at session start with the working directory set to the install folder, so `list_apps` reported `workspace` `~/.kiro/powers/installed/hexperiment` and no apps. The agent reaches power tools through the `kiro_powers` tool (`{"action": "activate", "powerName"}`, then `{"action": "use", "powerName", "serverName", "toolName"}`); `--trust-tools=kiro_powers` is needed headless. Activation lists "(No tools available)" while the server is still starting. `hexperiment-mcp` took 21 to 23 s to connect (budget `eraProbeBudgetMs` 30000); `import hexperiment.mcp_server.server` alone is 10.9 s (scikit-learn through `ghosts` -> `checks.metric`, pandas through `canary`). Without `.venv/Scripts` on PATH the server fails with "Connection closed".
- **Workflows:** the CLI has `run_workflow`, `inspect_workflow`, `update_workflow`; there is no `validate_workflow` tool.
- **Ghost hunter:** `ls` fails with "The permission request failed before it was answered" (no allow rule), `git status` is blocked by the deny rule `git*`.
- **Workflow steps and shell permissions (security).** In a workflow step the CLI shell tool is `run_command` (not `execute_pwsh`). Without `run_command` in `--trust-tools`, every shell call in the step is evaluated `ask` (no capability matched), even `hexperiment run apps/x`, which the ghost hunter's allow rule matches; headless it fails "permission request failed" and the workflow pauses. With `run_command` trusted, deny rules still apply (`git status` blocked, fs_write deny on `hex.toml` held) but a command that matches no rule runs (`ls`, `Get-Content ...` ran), and a command chained with `;` is split and each part judged alone (`hexperiment run --fast apps/x ; whoami` ran, `*;*` never matched). So in a headless workflow the shell allow list is not a boundary; only the deny list is. Interactive runs ask the user for each command instead.
- **Parallel tool calls.** A step agent may issue several tool calls at once; two `hexperiment run` on the same app then collide on `.hex/` ("snapshot.npz cannot be used"). The CLI has no per-app lock (the MCP server's lock is per process).
- **Power `mcp.json` with a `cwd` field holding `${workspaceFolder}`:** the server was never started (no log line). Placeholders in `env` untested because of that.
- **Recording (2026-11-12):** both recordings ran in a detached git worktree outside the project folder with `run_command` trusted, and every command and write was audited afterwards (hotels: `hexperiment run` twice, one `Get-Content` of `verdict.json`, five edits to `train.py`).
- **Shell rule matching (2026-11-12, chat session, run_command not trusted so nothing ran):** `*` matches `/` (`git*` refused `git log -1 apps/x`; `Write-Output*` refused `Write-Output apps/x`). Matching is case-sensitive: `GIT log -1` and `write-output x` matched no deny rule ("permission request failed"), while PowerShell ignores case, so in a headless workflow with `run_command` trusted they would run. `ls; Get-Date` was not caught by `*;*` in chat either (split). Conclusion: the deny list stops the usual spellings only; moving the ghost hunter to MCP `run_audit` is the real fix.
- **MCP start-up after the lazy imports (2026-11-12, Power saver):** workspace server connected in 13.1 s, the power's in 12.6 s, started together (was 21 to 23 s). The agent's first turn still came before the tools were listed, so a prompt that calls a tool at once races the server; reading a few files first avoids it. `hexperiment-mcp` must be on PATH (activate `.venv` before starting Kiro), or both servers fail with "Connection closed".

## Agent preToolUse hooks, checked live (Kiro CLI 3.5.0, headless V3, 2026-11-22)
- An agent file accepts `"hooks": {"preToolUse": [{"matcher": "<tool name>", "command": "<command>"}]}` (`kiro-cli agent validate` passes); `PreToolUse` or an unknown event name as the key fails validation.
- The hook runs before the tool with STDIN `{"session_id", "hook_event_name": "PreToolUse", "cwd": "C:\...\KiroweenHackathonSubmission", "tool_name": "fs_write" | "str_replace" | "fs_append", "tool_input": {"path", "text"} | {"path", "oldStr", "newStr", "replace_all"}}`. The path is given as the agent wrote it (relative here).
- Exit code 2 blocks the call: the agent receives "Tool "fs_write" was intercepted by PreToolUse hooks before execution. ... <stderr> ... The tool was NOT executed." No file was created or changed for any of the three tools. Unlike PostToolUse output, the hook's standard error does reach the agent.
- Matchers overlap: with entries for `fs_write`, `str_replace`, `fs_append` and `write`, the hook ran twice per call (the tool name and the `write` alias both matched).
- A missing hook command does not block: with `preToolUse` set to a command that does not exist, an `fs_write` the deny rules allow was carried out ("Created the ... file."). Only exit code 2 blocks, so the write guard fails open when `hexperiment` is not on `PATH`; the deny rules still apply.
