import { defineCommand } from '@jackwener/opencli';

export default defineCommand({
  description: 'Send a message to the claude.ai chat interface',
  options: [
    {
      name: 'message',
      type: 'string',
      required: true,
      description: 'The message to send'
    }
  ],
  handler: async ({ adapter, options }) => {
    const text = options.message;

    const result = await adapter.evaluate(async (inputText) => {
      // Find the composer/input area
      // Claude.ai uses a contenteditable div for the editor
      const findComposer = () => {
        // Try multiple selector patterns
        const selectors = [
          '[contenteditable="true"]',
          '[data-testid="chat-input"]',
          '#chat-input',
          '.chat-input',
          '[placeholder*="message"]',
          '[placeholder*="type"]',
          '[role="textbox"]',
          'div[contenteditable="true"][data-placeholder]'
        ];

        for (const selector of selectors) {
          const el = document.querySelector(selector);
          if (el) return el;
        }

        // Fallback: find any contenteditable that's not hidden
        const allContentEditable = Array.from(document.querySelectorAll('[contenteditable="true"]'));
        for (const el of allContentEditable) {
          const style = window.getComputedStyle(el);
          if (style.display !== 'none' && style.visibility !== 'hidden') {
            return el;
          }
        }

        return null;
      };

      const composer = findComposer();

      if (!composer) {
        return {
          success: false,
          error: 'Composer input not found. Make sure you are on the chat page and the page is fully loaded.',
          candidates: Array.from(document.querySelectorAll('[contenteditable="true"]')).map(el => ({
            tagName: el.tagName,
            className: el.className,
            placeholder: el.getAttribute('placeholder'),
            id: el.id
          }))
        };
      }

      // Focus and insert text using execCommand for React controlled editor
      composer.focus();

      // Wait for focus to be established
      await new Promise(resolve => setTimeout(resolve, 100));

      // Clear existing text first by selecting all and deleting
      document.execCommand('selectAll', false, null);
      document.execCommand('delete', false, null);

      // Insert the new text
      document.execCommand('insertText', false, inputText);

      // Wait for React to update
      await new Promise(resolve => setTimeout(resolve, 200));

      // Find and click the send button, or press Enter/Meta+Enter
      const findSendButton = () => {
        const selectors = [
          '[data-testid="send-button"]',
          '[aria-label*="send"]',
          'button[aria-label*="send"]',
          '.send-button',
          '#send-button'
        ];

        for (const selector of selectors) {
          const el = document.querySelector(selector);
          if (el) return el;
        }

        // Look for any button that might be send
        const buttons = Array.from(document.querySelectorAll('button'));
        for (const btn of buttons) {
          const btnText = btn.innerText.toLowerCase();
          if (btnText.includes('send') || btnText.includes('submit')) {
            return btn;
          }
        }

        return null;
      };

      const sendButton = findSendButton();

      if (sendButton) {
        sendButton.click();
        return { success: true, action: 'clicked_send_button', composerInfo: { id: composer.id, className: composer.className } };
      }

      // Press Enter if no send button found
      const event = new KeyboardEvent('keydown', {
        key: 'Enter',
        code: 'Enter',
        keyCode: 13,
        bubbles: true,
        cancelable: true
      });
      composer.dispatchEvent(event);

      return { success: true, action: 'pressed_enter', composerInfo: { id: composer.id, className: composer.className } };
    }, text);

    if (!result.success) {
      return {
        success: false,
        error: result.error,
        composerCandidates: result.candidates
      };
    }

    return {
      success: true,
      message: 'Message sent',
      action: result.action,
      composerInfo: result.composerInfo
    };
  }
});
