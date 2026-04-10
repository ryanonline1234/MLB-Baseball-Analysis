import { defineCommand } from '@jackwener/opencli';

export default defineCommand({
  description: 'Send a message and wait for the response (one-shot command)',
  options: [
    {
      name: 'prompt',
      type: 'string',
      required: true,
      description: 'The prompt to send'
    },
    {
      name: 'timeout',
      type: 'number',
      required: false,
      defaultValue: 120000,
      description: 'Timeout in milliseconds (default: 120000)'
    }
  ],
  handler: async ({ adapter, options }) => {
    const prompt = options.prompt;
    const timeout = options.timeout;
    const startTime = Date.now();

    // Step 1: Send the message
    const sendResult = await adapter.evaluate(async (inputText) => {
      const findComposer = () => {
        const selectors = [
          '[contenteditable="true"]',
          '[data-testid="chat-input"]',
          '#chat-input',
          '.chat-input',
          '[placeholder*="message"]',
          '[placeholder*="type"]',
          '[role="textbox"]'
        ];

        for (const selector of selectors) {
          const el = document.querySelector(selector);
          if (el) return el;
        }

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
        return { success: false, error: 'Composer not found' };
      }

      composer.focus();
      await new Promise(resolve => setTimeout(resolve, 100));
      document.execCommand('selectAll', false, null);
      document.execCommand('delete', false, null);
      document.execCommand('insertText', false, inputText);
      await new Promise(resolve => setTimeout(resolve, 200));

      // Try to find send button
      const buttons = Array.from(document.querySelectorAll('button'));
      let sentViaButton = false;
      for (const btn of buttons) {
        const btnText = btn.innerText.toLowerCase();
        if (btnText.includes('send') || btnText.includes('submit')) {
          btn.click();
          sentViaButton = true;
          break;
        }
      }

      if (!sentViaButton) {
        composer.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', code: 'Enter', bubbles: true }));
      }

      return { success: true, sentViaButton };
    }, prompt);

    if (!sendResult.success) {
      return { success: false, error: sendResult.error };
    }

    // Step 2: Poll for response completion
    const response = await adapter.evaluate(async (timeoutMs) => {
      const startTime = Date.now();

      // Wait for the stop/generating button to appear (indicates response has started)
      const waitForStart = async () => {
        const check = () => {
          const stopButtons = document.querySelectorAll(
            '[data-testid*="stop"], button[aria-label*="stop"], .stop-button'
          );
          return stopButtons.length > 0;
        };

        while (!check() && Date.now() - startTime < timeoutMs) {
          await new Promise(resolve => setTimeout(resolve, 200));
        }

        if (Date.now() - startTime >= timeoutMs) {
          return { timedOut: true, message: 'Timeout waiting for response to start' };
        }

        return { timedOut: false };
      };

      const startResult = await waitForStart();
      if (startResult.timedOut) {
        return startResult;
      }

      // Now wait for the stop button to disappear (indicates response is complete)
      const waitForComplete = async () => {
        const check = () => {
          const stopButtons = document.querySelectorAll(
            '[data-testid*="stop"], button[aria-label*="stop"], .stop-button'
          );
          return stopButtons.length === 0;
        };

        while (!check() && Date.now() - startTime < timeoutMs) {
          await new Promise(resolve => setTimeout(resolve, 500));
        }

        if (Date.now() - startTime >= timeoutMs) {
          return { timedOut: true, message: 'Timeout waiting for response to complete' };
        }

        return { timedOut: false };
      };

      const completeResult = await waitForComplete();
      if (completeResult.timedOut) {
        return completeResult;
      }

      // Get the latest assistant message
      const getLatestMessage = () => {
        const allMessages = Array.from(
          document.querySelectorAll(
            '[class*="message"], [class*="response"], [class*="chat-message"], [class*="assistant-message"]'
          )
        );

        const assistantMessages = allMessages.filter(msg => {
          const hasAssistantClass = msg.className.toLowerCase().includes('assistant') ||
                                    msg.className.toLowerCase().includes('claude');
          const hasAiAttribute = msg.getAttribute('data-role') === 'assistant';

          return hasAssistantClass || hasAiAttribute || allMessages.length - allMessages.indexOf(msg) <= 2;
        });

        if (assistantMessages.length === 0) {
          return null;
        }

        return assistantMessages[assistantMessages.length - 1].innerText;
      };

      return { timedOut: false, message: getLatestMessage() };
    }, timeout);

    const elapsed = Date.now() - startTime;

    if (response.timedOut) {
      return {
        success: false,
        error: response.message,
        elapsed
      };
    }

    return {
      success: true,
      message: response.message,
      elapsed
    };
  }
});
