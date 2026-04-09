# Planner

This document describes the high-level plan for this project.

## Decisions Made

**Memory and Context Architecture (2026-04-09)**

Universal files (live in ~/, apply to all projects):
- `~/AGENTS.md` — worker behavior rules and session protocols, never project-specific
- `~/.claude/commands/worker-mode.md` — slash command definition

Project-specific files (live in `~/projects/PROJECT_NAME/`):
- `MEMORY.md` — what was built, current state, blockers
- `PLANNER.md` — goal, decisions made, next steps
- `AGENTS.md` (optional) — copy of global AGENTS.md with project-specific gotchas appended

Session start convention: Worker is always started by `cd ~/projects/PROJECT_NAME` then `claude-code` then `/worker-mode`. Working directory determines which project's MEMORY.md and PLANNER.md are loaded. No explicit folder path needs to be passed.

**new-project.sh (2026-04-09)**
- Create `~/projects/$PROJECT/`
- Copy `~/AGENTS.md` into it as a base
- Create empty `MEMORY.md` and `PLANNER.md`
- Create `.claude/commands/` directory inside the project
- Print "✅ $PROJECT ready. cd into it and run /worker-mode"
