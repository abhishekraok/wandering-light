import "@testing-library/jest-dom/vitest";

// React Flow measures the DOM; jsdom reports zero-sized elements, so the graph
// renders nothing without these. The graph itself is covered by layout tests.
class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}
globalThis.ResizeObserver = globalThis.ResizeObserver ?? (ResizeObserverStub as never);
globalThis.DOMMatrixReadOnly =
  globalThis.DOMMatrixReadOnly ??
  (class {
    m22 = 1;
  } as never);
