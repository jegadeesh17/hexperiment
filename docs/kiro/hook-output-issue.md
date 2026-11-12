# Command hook output is not added to the agent's context (CLI 3.2.0 and IDE 1.4.154)

## Summary

The hooks documentation (https://kiro.dev/docs/hooks/actions/) says that when a command hook exits with code 0, its standard output is added to the agent's context. In Kiro CLI 3.2.0 and Kiro IDE 1.4.154 on Windows 11, a command hook on `PostFileSave` runs and completes, but its output never reaches the agent.

## Environment

- Windows 11, x64
- Kiro IDE 1.4.154 (Vibe session, trusted workspace)
- Kiro CLI 3.2.0 (`kiro-cli chat --v3 --no-interactive`)

## Steps to reproduce

1. In a workspace, create `.kiro/hooks/probe.json`:

   ```json
   {
     "version": "v1",
     "hooks": [
       {
         "name": "probe",
         "trigger": "PostFileSave",
         "matcher": ".*\\.py$",
         "action": { "type": "command", "command": "echo HOOK-PROBE-OUTPUT" },
         "timeout": 30
       }
     ]
   }
   ```

2. Trust the workspace (IDE) and start a new session.
3. Ask the agent: "Append the line `# probe` to `example.py` and save it. Then wait 30 seconds and quote, word for word, any hook output you received after the save. If you received none, answer 'none'."

## Expected

The agent's context contains `HOOK-PROBE-OUTPUT` after the save, and the agent quotes it.

## Actual

- The hook runs: its command's side effects happen, and the session records a `ContextualHookInvoked` entry with `"status": "completed"`.
- That entry has no output field, the text `HOOK-PROBE-OUTPUT` appears nowhere in the session's `messages.jsonl`, and the agent answers "none".
- Same result in the CLI (headless) and the IDE. In the CLI, a `PostToolUse` hook with matcher `write` behaves the same.
- Kiro does wait for the hook to finish before the next model turn, so the output is produced in time; it is just not passed on.

## Impact

Hooks that report a check result to the agent (linters, test runners, audits) have no effect on the agent's behaviour, so tools built on the documented behaviour silently degrade.

## How this was found

The probes behind this report ran `hexperiment hook` (Hexperiment's save hook) and small logging wrappers instead of `echo`; the minimal reproduction above is assembled from them. The Hexperiment README lists this as a known limit: the save hook records a fast run and clears an old approval, but its summary does not reach the agent.
