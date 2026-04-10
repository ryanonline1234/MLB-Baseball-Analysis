import { defineCommand } from '@jackwener/opencli';

export default defineCommand({
  description: 'Dump document.body.innerHTML and accessibility snapshot for analysis',
  handler: async ({ adapter }) => {
    // First dump the full HTML body
    const htmlResult = await adapter.evaluate(async () => {
      return {
        bodyHTML: document.body.innerHTML,
        bodyText: document.body.innerText,
        documentURL: window.location.href,
        title: document.title
      };
    });

    // Then get accessibility tree for better element identification
    const accessibilityResult = await adapter.evaluate(async () => {
      try {
        // Try to get accessibility info using the browser API
        const getAccessibilityInfo = (element: HTMLElement) => {
          return {
            tagName: element.tagName,
            id: element.id || null,
            className: element.className || null,
            role: (element as HTMLElement & { role?: string }).role || null,
            ariaLabel: element.getAttribute('aria-label') || null,
            ariaRoleDescription: element.getAttribute('aria-roledescription') || null,
            placeholder: element.getAttribute('placeholder') || null,
            innerText: element.innerText?.substring(0, 200) || null,
            hasAttributeDataTestid: element.hasAttribute('data-testid'),
            dataTestid: element.getAttribute('data-testid') || null
          };
        };

        // Get all interactive elements
        const interactiveElements: Array<HTMLElement> = Array.from(
          document.querySelectorAll('button, input, textarea, [contenteditable="true"], [tabindex]:not([tabindex="-1"])')
        );

        // Limit to first 50 to avoid huge response
        const elementsInfo = interactiveElements.slice(0, 50).map(getAccessibilityInfo);

        return {
          interactiveElements: elementsInfo,
          totalInteractiveElements: interactiveElements.length
        };
      } catch (e) {
        return { error: (e as Error).message };
      }
    });

    return {
      success: true,
      htmlDump: {
        bodyHTML: htmlResult.bodyHTML,
        bodyText: htmlResult.bodyText,
        documentURL: htmlResult.documentURL,
        title: htmlResult.title
      },
      accessibilityDump: accessibilityResult,
      message: 'HTML and accessibility snapshot captured'
    };
  }
});
