import { defineCommand } from '@jackwener/opencli';

export default defineCommand({
  description: 'Verify the claude.ai tab is reachable and ready',
  handler: async ({ adapter }) => {
    const result = await adapter.evaluate(async () => {
      const url = window.location.href;
      const title = document.title;
      const body = document.body;

      // Check if we're on claude.ai
      const isClaudeUrl = url.includes('claude.ai');

      // Check for main chat interface elements
      const chatContainer = document.querySelector('[data-testid="chat-area"]') ||
                            document.querySelector('.chat-container') ||
                            document.querySelector('#chat-container') ||
                            document.querySelector('.chat-interface');

      // Check for React root or app container
      const appContainer = document.querySelector('[data-testid="app-container"]') ||
                          document.querySelector('.app-container') ||
                          document.querySelector('#app') ||
                          document.querySelector('main');

      // Check if page has loaded (body should have content)
      const hasContent = body.innerText.length > 0 || body.innerHTML.length > 100;

      return {
        url,
        title,
        isClaudeUrl,
        hasChatInterface: !!chatContainer,
        hasAppContainer: !!appContainer,
        hasContent,
        ready: isClaudeUrl && hasChatInterface && hasContent
      };
    });

    if (!result.isClaudeUrl) {
      return {
        success: false,
        error: `Not on claude.ai. Current URL: ${result.url}`,
        details: result
      };
    }

    if (!result.hasChatInterface) {
      return {
        success: false,
        error: 'Chat interface not found. Make sure you are on the chat page.',
        details: result
      };
    }

    if (!result.hasContent) {
      return {
        success: false,
        error: 'Page content not loaded yet. Please wait and try again.',
        details: result
      };
    }

    return {
      success: true,
      message: 'claude.ai is ready',
      details: result
    };
  }
});
