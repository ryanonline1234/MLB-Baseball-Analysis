import { defineCommand } from '@jackwener/opencli';

export default defineCommand({
  description: 'Start a fresh conversation in claude.ai',
  handler: async ({ adapter }) => {
    const result = await adapter.evaluate(async () => {
      // Method 1: Check for Cmd+N or new conversation button
      // Method 2: Navigate to claude.ai/new

      // Try to find a new conversation button or menu item
      const findNewButton = () => {
        const selectors = [
          '[data-testid*="new-chat"]',
          '[data-testid*="new-conversation"]',
          '[aria-label*="new chat"]',
          'button[aria-label*="new chat"]',
          '.new-chat-button',
          '#new-chat-button'
        ];

        for (const selector of selectors) {
          const el = document.querySelector(selector);
          if (el) return el;
        }

        // Look for any button with "new" text
        const buttons = Array.from(document.querySelectorAll('button'));
        for (const btn of buttons) {
          const btnText = btn.innerText.toLowerCase();
          if (btnText.includes('new chat') || btnText.includes('new conversation')) {
            return btn;
          }
        }

        return null;
      };

      const newButton = findNewButton();

      if (newButton) {
        newButton.click();
        return { action: 'clicked_new_button', success: true, urlAfter: window.location.href };
      }

      // Method 2: Navigate to new conversation
      const originalUrl = window.location.href;
      window.location.href = 'https://claude.ai/new';

      return { action: 'navigated_to_new', success: true, originalUrl, newUrl: 'https://claude.ai/new' };
    });

    // Wait a moment for navigation to complete
    await new Promise(resolve => setTimeout(resolve, 2000));

    return {
      success: true,
      message: 'Started new conversation',
      details: result
    };
  }
});
