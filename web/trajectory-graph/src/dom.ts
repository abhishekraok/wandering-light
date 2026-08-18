export function requiredElement<T extends HTMLElement>(selector: string): T {
  const element = document.querySelector<T>(selector);
  if (!element) throw new Error(`Missing required element: ${selector}`);
  return element;
}

export function clear(element: Element): void {
  element.replaceChildren();
}

export function textElement<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  text: string,
  className?: string,
): HTMLElementTagNameMap[K] {
  const element = document.createElement(tag);
  element.textContent = text;
  if (className) element.className = className;
  return element;
}

export function setHidden(element: HTMLElement, hidden: boolean): void {
  element.classList.toggle("hidden", hidden);
}

export function compactType(typeName: string): string {
  return typeName.split(".").at(-1) ?? typeName;
}

export function compactId(value: string, length = 10): string {
  return value.length <= length ? value : `${value.slice(0, length)}…`;
}

export function previewSerialized(value: string | null, limit = 88): string {
  if (value === null) return "—";
  let preview = value;
  try {
    const payload = JSON.parse(value) as { type?: unknown; items?: unknown };
    if (typeof payload.type === "string" && Array.isArray(payload.items)) {
      preview = `TL<${compactType(payload.type)}>(${JSON.stringify(payload.items)})`;
    }
  } catch {
    // Legacy records may use a non-JSON preview; show it as inert text.
  }
  return preview.length <= limit ? preview : `${preview.slice(0, limit - 1)}…`;
}

export function formatCount(value: number | null | undefined): string {
  return value === null || value === undefined ? "—" : new Intl.NumberFormat().format(value);
}

export function clampNumber(input: HTMLInputElement, fallback: number): number {
  const parsed = Number(input.value);
  const minimum = Number(input.min);
  const maximum = Number(input.max);
  if (!Number.isFinite(parsed)) return fallback;
  return Math.min(maximum, Math.max(minimum, parsed));
}

export function debounce<T extends unknown[]>(
  callback: (...args: T) => void,
  delay: number,
): (...args: T) => void {
  let timer: number | undefined;
  return (...args: T) => {
    window.clearTimeout(timer);
    timer = window.setTimeout(() => callback(...args), delay);
  };
}
