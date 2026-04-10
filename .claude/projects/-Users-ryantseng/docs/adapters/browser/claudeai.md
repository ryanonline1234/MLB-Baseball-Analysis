# Claude.ai Browser Adapter

This adapter allows you to control a Claude.ai browser tab through the OpenCLI framework.

## Prerequisites

- **OpenCLI** must be installed and running (`opencli doctor` should show all checks passing)
- A **Claude.ai tab must be open and logged in** before running any commands

## Setup

1. Install OpenCLI:
   ```bash
   npm install -g @jackwener/opencli
   opencli doctor
   ```

2. Open Claude.ai in your browser and log in:
   ```
   https://claude.ai
   ```

3. The adapter will automatically connect to the active Claude.ai tab.

## Commands

### `opencli claudeai status`

Verify the Claude.ai tab is reachable and ready.

**Returns:**
- `success`: Whether the tab is ready
- `url`: Current page URL
- `title`: Page title
- `hasChatInterface`: Whether chat interface elements are present
- `hasContent`: Whether page content has loaded

### `opencli claudeai dump`

Dump the current document body HTML and accessibility snapshot for analysis.

**Returns:**
- `bodyHTML`: Full innerHTML of document.body
- `bodyText`: Visible text content
- `interactiveElements`: List of interactive elements with their selectors
- `totalInteractiveElements`: Count of interactive elements

Use this to discover the correct selectors for the input box, send button, and response area.

### `opencli claudeai read`

Extract the latest assistant response from the current conversation.

**Returns:**
- `success`: Whether a response was found
- `message`: The text content of the latest assistant message
- `elementInfo`: Details about the response element (tagName, className, etc.)

### `opencli claudeai send "message"`

Send a message to the Claude.ai chat interface.

**Arguments:**
- `message` (required): The text to send

**Returns:**
- `success`: Whether the message was sent
- `action`: Either "clicked_send_button" or "pressed_enter"
- `composerInfo`: Details about the composer element

### `opencli claudeai new`

Start a fresh conversation in Claude.ai.

**Returns:**
- `success`: Whether a new conversation was started
- `details`: Information about the action taken

### `opencli claudeai ask "prompt"`

One-shot command that sends a prompt and waits for the response.

**Arguments:**
- `prompt` (required): The prompt to send
- `timeout` (optional, default: 120000): Timeout in milliseconds

**Returns:**
- `success`: Whether the operation completed
- `message`: The assistant's response
- `elapsed`: Time taken in milliseconds

If the response takes longer than the timeout, returns an error with details.

## Troubleshooting

### "Not on claude.ai" error
Make sure you are on the claude.ai website before running commands.

### "Chat interface not found"
The page may not be fully loaded. Wait a few seconds and try again.

### "Composer input not found"
Claude.ai's UI may have changed. Run `opencli claudeai dump` to see the current structure and update the selector patterns in the adapter code.

### Response timeout
The 120-second timeout may be exceeded for long conversations. Use the `timeout` option to increase it:

```bash
opencli claudeai ask "Your prompt" --timeout 180000
```

## Requirements

- **The Claude.ai tab must be open** before running any commands
- **You must be logged in** to Claude.ai
- The tab should be **active/visible** for best results
- JavaScript must be enabled in your browser
