#!/bin/bash

# Create a new project directory with worker-mode support
# Usage: ./new-project.sh PROJECT_NAME

PROJECT_NAME="$1"

if [ -z "$PROJECT_NAME" ]; then
  echo "Usage: $0 PROJECT_NAME"
  exit 1
fi

# Create project directory
mkdir -p ~/projects/"$PROJECT_NAME"

# Copy global AGENTS.md as base
cp ~/AGENTS.md ~/projects/"$PROJECT_NAME"/AGENTS.md

# Create empty MEMORY.md and PLANNER.md
touch ~/projects/"$PROJECT_NAME"/MEMORY.md
touch ~/projects/"$PROJECT_NAME"/PLANNER.md

# Create .claude/commands directory inside the project
mkdir -p ~/projects/"$PROJECT_NAME"/.claude/commands

echo "✅ $PROJECT_NAME ready. cd into it and run /worker-mode"
