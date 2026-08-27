import '@testing-library/jest-dom/vitest'

// jsdom 29.1.1 ships an HTMLDialogElement whose implementation is a bare subclass with no
// showModal/close -- verified against node_modules/jsdom/lib/jsdom/living/nodes/
// HTMLDialogElement-impl.js, which is `class HTMLDialogElementImpl extends HTMLElementImpl {}`.
// Dialog.tsx uses the native <dialog>, so any test that opens it needs this shim.
//
// This does NOT emulate focus trapping, the ::backdrop pseudo-element, or top-layer stacking --
// those are real browser behaviour with no jsdom equivalent, and are verified by hand during
// the responsive/keyboard-focus pass, not by this test suite.
if (typeof HTMLDialogElement !== 'undefined' && !HTMLDialogElement.prototype.showModal) {
  HTMLDialogElement.prototype.showModal = function (this: HTMLDialogElement) {
    this.setAttribute('open', '')
  }
  HTMLDialogElement.prototype.show = function (this: HTMLDialogElement) {
    this.setAttribute('open', '')
  }
  HTMLDialogElement.prototype.close = function (this: HTMLDialogElement, returnValue?: string) {
    this.removeAttribute('open')
    if (returnValue !== undefined) {
      this.returnValue = returnValue
    }
    this.dispatchEvent(new Event('close'))
  }
}

// jsdom does not implement window.matchMedia at all -- useTheme.ts reads
// `prefers-color-scheme` through it to resolve the effective theme when no explicit choice is
// stored, and any component using that hook throws in every test without this. `matches:
// false` is an arbitrary default (jsdom has no real viewport media state to report); a test
// asserting behaviour under a specific system preference should override `window.matchMedia`
// itself rather than rely on this stub's default.
if (typeof window !== 'undefined' && !window.matchMedia) {
  window.matchMedia = (query: string) =>
    ({
      matches: false,
      media: query,
      onchange: null,
      addListener: () => {},
      removeListener: () => {},
      addEventListener: () => {},
      removeEventListener: () => {},
      dispatchEvent: () => false,
    }) as MediaQueryList
}
