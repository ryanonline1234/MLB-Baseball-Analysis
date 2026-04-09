# Agents

This document describes the agent roles in this project.

## Session Start Protocol

Read MEMORY.md, read PLANNER.md, run `git log --oneline -5`, then confirm context loaded with a one-line summary before accepting any task.

## Session End Protocol

Update MEMORY.md with what changed, append any architectural decisions to PLANNER.md under a "Decisions Made" section, then run `git add MEMORY.md PLANNER.md && git commit -m "chore: session state update"` — and never respond "done" until this is complete. If given a BLOCKED signal, write the blocker to MEMORY.md before exiting.
