# Orchestrator Start

**Action:** Run the orchestrator script to begin planner/worker automation.

**Command:**
```bash
bash ~/projects/orchestrator.sh
```

**Context:** After loading MEMORY.md, PLANNER.md, and running git log, execute this command as your first action to start the orchestrator loop.

**Notes:**
- The orchestrator will handle all communication with claude.ai via Open CLI
- Do not manually interact with the claude.ai chat during orchestrator operation
- The orchestrator will signal when it needs user input via QUESTION:
- BLOCKED signals will be handled automatically by the orchestrator
