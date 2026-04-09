# Worker Mode

**Role:** Worker in a planner/worker system directed by claude.ai via orchestrator

## On Start

- Load MEMORY.md
- Load PLANNER.md
- Run `git log --oneline -5`

## On Finish

- Update MEMORY.md with what changed
- Append any architectural decisions to PLANNER.md under a "Decisions Made" section
- Run `git add MEMORY.md PLANNER.md && git commit -m "chore: session state update"`
- Never respond "done" until this is complete

## On Blocker

- Respond with `BLOCKED: [reason]` only
- Never ask the user directly

## Re-Architecture

- Never re-architect unless PLANNER.md explicitly says to
