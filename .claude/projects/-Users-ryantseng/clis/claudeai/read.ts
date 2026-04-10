import { defineCommand } from '@jackwener/opencli';

export default defineCommand({
  description: 'Extract the latest assistant response from the conversation',
  handler: async ({ adapter }) => {
    const result = await adapter.evaluate(async () => {
      // Look for message bubbles/containers - prioritize the last one
      // Claude.ai uses various selectors, try multiple patterns
      const allMessages = Array.from(
        document.querySelectorAll(
          '[class*="message"], [class*="response"], [class*="chat-message"], [class*="assistant-message"], ' +
          '[data-testid*="message"], [data-testid*="response"], ' +
          '.message-content, .response-content, .chat-message-content'
        )
      );

      // Filter out user messages and focus on assistant responses
      // Claude.ai assistant messages typically have specific patterns
      const assistantMessages = allMessages.filter(msg => {
        const msgText = msg.innerText.toLowerCase();
        const hasAssistantClass = msg.className.toLowerCase().includes('assistant') ||
                                  msg.className.toLowerCase().includes('claude') ||
                                  msg.className.toLowerCase().includes('ai');
        const hasAssistantId = msg.id.toLowerCase().includes('assistant') ||
                               msg.id.toLowerCase().includes('claude');
        const hasAiAttribute = msg.getAttribute('data-ai') === 'true' ||
                               msg.getAttribute('data-role') === 'assistant';

        // Must have some assistant indicator OR be among the last few messages
        return hasAssistantClass || hasAssistantId || hasAiAttribute || allMessages.length - allMessages.indexOf(msg) <= 2;
      });

      if (assistantMessages.length === 0) {
        return {
          found: false,
          message: 'No assistant response found',
          candidates: allMessages.slice(-5).map(m => ({
            tagName: m.tagName,
            className: m.className,
            id: m.id,
            textPreview: m.innerText?.substring(0, 100) || ''
          }))
        };
      }

      // Get the last assistant message
      const lastMessage = assistantMessages[assistantMessages.length - 1];

      return {
        found: true,
        message: lastMessage.innerText || '',
        elementInfo: {
          tagName: lastMessage.tagName,
          id: lastMessage.id,
          className: lastMessage.className,
          outerHTML: lastMessage.outerHTML.substring(0, 500)
        }
      };
    });

    if (!result.found) {
      return {
        success: false,
        error: result.message,
        candidates: result.candidates
      };
    }

    return {
      success: true,
      message: result.message,
      elementInfo: result.elementInfo
    };
  }
});
