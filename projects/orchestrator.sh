#!/bin/bash

# Orchestrator Script for Planner/Worker Architecture
# Manages the planner/worker loop via Open CLI browser automation of claude.ai GUI

set -e

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

# Wait for streaming to complete by polling for stop button disappearance
wait_for_streaming_complete() {
    local timeout=120
    local poll_interval=2
    local elapsed=0

    echo "Waiting for streaming to complete..."

    while [ $elapsed -lt $timeout ]; do
        # Check if stop button (cancel button) has disappeared from DOM
        # The stop button typically has specific classes or text
        if open-cli find 'button[aria-label="Stop"]' | grep -q "not found" 2>/dev/null || \
           open-cli find 'div[role="button"][aria-label="Stop"]' | grep -q "not found" 2>/dev/null; then
            echo "Streaming complete - stop button disappeared"
            return 0
        fi

        sleep $poll_interval
        elapsed=$((elapsed + poll_interval))
        echo "Streaming in progress... (${elapsed}s elapsed)"
    done

    echo "ERROR: Timeout waiting for streaming to complete"
    return 1
}

# Inject content into claude.ai input box
inject_content() {
    local content="$1"
    echo "Injecting content into claude.ai input box"
    open-cli form_input --selector 'div[contenteditable="true"]' --value "$content"
}

# Get the latest planner response
get_planner_response() {
    # Read the page content around the chat area to get the latest response
    # This is a simplified version - may need adjustment based on actual DOM structure
    open-cli read_page --all 2>/dev/null | tail -100
}

# ============================================================================
# STEP 1: CONTEXT INJECTION ON START
# ============================================================================

context_injection() {
    echo "=== Context Injection Phase ==="

    # Read project files
    local planner_content
    local memory_content
    local agents_content
    local git_log

    if [ -f "PLANNER.md" ]; then
        planner_content=$(cat PLANNER.md)
    else
        planner_content="# Planner\n\nNo planner content found."
    fi

    if [ -f "MEMORY.md" ]; then
        memory_content=$(cat MEMORY.md)
    else
        memory_content="# Memory\n\nNo memory content found."
    fi

    if [ -f "AGENTS.md" ]; then
        agents_content=$(cat AGENTS.md)
    else
        agents_content="# Agents\n\nNo agents content found."
    fi

    git_log=$(git log --oneline -5 2>/dev/null || echo "No git history available")

    # Compose briefing message
    local briefing="--- WORKER MODE BRIEFING ---

[PLANNER]
$planner_content

[MEMORY]
$memory_content

[AGENTS]
$agents_content

[GIT LOG -5]
$git_log

Please confirm context loaded with a one-line summary, then I will be ready to receive tasks."

    echo "Composing briefing message..."
    echo "$briefing"

    # Inject context into claude.ai
    inject_content "$briefing"

    # Wait for streaming to complete
    wait_for_streaming_complete

    echo "Context injection complete."
}

# ============================================================================
# STEP 2: MAIN LOOP
# ============================================================================

main_loop() {
    echo "=== Main Loop Started ==="

    while true; do
        echo "Waiting for planner response..."

        # Wait for streaming to complete
        if ! wait_for_streaming_complete; then
            echo "ERROR: Failed to receive planner response"
            exit 1
        fi

        # Get the latest planner response
        local response
        response=$(get_planner_response)

        echo "=== Received Planner Response ==="
        echo "$response"
        echo "==============================="

        # Parse for signals
        # Check for BLOCKED signal first
        if echo "$response" | grep -q "^BLOCKED:"; then
            local reason
            reason=$(echo "$response" | sed 's/^BLOCKED: *//')
            handle_blocked "$reason"
        elif echo "$response" | grep -q "<exec>PROMPT</exec>"; then
            # Extract prompt and execute as task
            local prompt
            prompt=$(echo "$response" | grep -oP '(?<=<exec>PROMPT>).*?(?=</exec>)' | head -1)
            handle_exec_prompt "$prompt"
        elif echo "$response" | grep -q "^QUESTION:"; then
            # Handle question
            local question
            question=$(echo "$response" | sed 's/^QUESTION: *//')
            handle_question "$question"
        elif echo "$response" | grep -q "^DONE:"; then
            # Handle done signal
            local summary
            summary=$(echo "$response" | sed 's/^DONE: *//')
            handle_done "$summary"
            break
        else
            echo "No recognized signal found in response. Continuing..."
            # Continue loop to wait for next response
        fi
    done
}

# ============================================================================
# STEP 3: WORKER RESULT INJECTION
# ============================================================================

handle_exec_prompt() {
    local prompt="$1"
    echo "=== EXEC PROMPT DETECTED ==="
    echo "Prompt: $prompt"
    echo ""

    # Execute the prompt as a task using Claude Code CLI
    # Note: This would typically use the Claude Code CLI or API
    local output
    output=$(claude --prompt "$prompt" 2>&1) || {
        echo "ERROR: Task execution failed"
        output="ERROR: Task execution failed - no output captured"
    }

    echo "=== TASK OUTPUT ==="
    echo "$output"
    echo "==================="

    # Inject result back to claude.ai
    local result_injection="[WORKER RESULT]
$output"

    echo ""
    echo "Injecting worker result..."
    inject_content "$result_injection"

    # Wait for streaming to complete
    wait_for_streaming_complete
}

handle_question() {
    local question="$1"
    echo "=== QUESTION DETECTED ==="
    echo "Question: $question"
    echo ""

    # Wait for user keyboard input
    read -p "Your answer: " answer

    # Inject answer back to claude.ai
    local answer_injection="[USER ANSWER]
$answer"

    echo ""
    echo "Injecting user answer..."
    inject_content "$answer_injection"

    # Wait for streaming to complete
    wait_for_streaming_complete
}

handle_done() {
    local summary="$1"
    echo "=== DONE SIGNAL DETECTED ==="
    echo "Summary: $summary"

    # Run session end protocol
    run_session_end_protocol
}

# ============================================================================
# STEP 4: SESSION END PROTOCOL
# ============================================================================

run_session_end_protocol() {
    echo ""
    echo "=== Running Session End Protocol ==="

    # Update MEMORY.md with what changed
    echo "Updating MEMORY.md with session changes..."

    # Append to MEMORY.md (this is a simplified approach - actual implementation may need diff/summary)
    echo ""
    echo "## Session $(date '+%Y-%m-%d %H:%M:%S')" >> MEMORY.md
    echo "Session completed. Summary: $summary" >> MEMORY.md

    # Commit changes
    echo ""
    echo "Committing changes..."
    git add MEMORY.md PLANNER.md 2>/dev/null || true
    git commit -m "chore: session state update" -m "Summary: $summary" 2>/dev/null || {
        echo "No changes to commit or git error"
    }

    echo "Session end protocol complete."
}

# ============================================================================
# STEP 5: BLOCKED HANDLING
# ============================================================================

handle_blocked() {
    local reason="$1"
    echo "=== BLOCKED SIGNAL ==="
    echo "Reason: $reason"

    # Write blocker to MEMORY.md
    echo "" >> MEMORY.md
    echo "## BLOCKED: $(date '+%Y-%m-%d %H:%M:%S')" >> MEMORY.md
    echo "Blocker: $reason" >> MEMORY.md

    # Inject BLOCKED signal into claude.ai
    inject_content "BLOCKED: $reason"

    # Wait for streaming to complete
    wait_for_streaming_complete

    # Check for resolution or QUESTION
}

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

main() {
    echo "=========================================="
    echo "Claude.ai Orchestrator - Starting"
    echo "=========================================="
    echo "Working directory: $(pwd)"
    echo ""

    # Verify required tools
    if ! command -v open-cli &> /dev/null; then
        echo "ERROR: open-cli is not installed or not in PATH"
        exit 1
    fi

    # Check we're in a project directory
    if [ ! -f "PLANNER.md" ] || [ ! -f "MEMORY.md" ]; then
        echo "ERROR: Not in a valid project directory (missing PLANNER.md or MEMORY.md)"
        echo "Current directory: $(pwd)"
        exit 1
    fi

    # Step 1: Context injection
    context_injection

    # Step 2: Main loop
    main_loop

    echo ""
    echo "=========================================="
    echo "Orchestrator completed successfully"
    echo "=========================================="
}

# Run main function
main "$@"
