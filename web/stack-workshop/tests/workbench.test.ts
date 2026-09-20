import { beforeAll, afterAll, describe, expect, it, vi } from 'vitest';
import { spawn, spawnSync, type ChildProcess } from 'node:child_process';
import { existsSync } from 'node:fs';
import { resolve } from 'node:path';

const root = resolve(process.cwd(), '../..');
let server: ChildProcess;
const tools: Record<string, any> = {};
const nativeFetch = globalThis.fetch;
const base = 'http://127.0.0.1:18018';
const waitFor = async (predicate: () => boolean) => { const end = Date.now() + 12000; while (!predicate()) { if (Date.now() > end) throw Error('Timed out waiting for UI state'); await new Promise(r => setTimeout(r, 35)); } };
const click = (query: string) => { const el = document.querySelector<HTMLButtonElement>(query); expect(el, query).toBeTruthy(); el!.click(); };

beforeAll(async () => {
  const python = process.env.STACK_WORKSHOP_PYTHON || (existsSync(resolve(root, '.venv-workshop312/Scripts/python.exe')) ? resolve(root, '.venv-workshop312/Scripts/python.exe') : 'python');
  server = spawn(python, ['-m', 'wandering_light.stack_workshop.server', '--port', '18018'], { cwd: root, windowsHide: true, stdio: 'pipe' });
  await new Promise<void>((resolve, reject) => { server.stdout!.on('data', chunk => { if (chunk.toString().includes('Signal House is ready')) resolve(); }); server.once('error', reject); server.once('exit', code => { if (code) reject(Error(`Server exited ${code}`)); }); });
  vi.stubGlobal('fetch', (url: string, init?: RequestInit) => nativeFetch(new URL(url, base), init));
  Element.prototype.animate = vi.fn(() => ({ finished: Promise.resolve(), cancel() {} })) as any;
  Element.prototype.scrollIntoView = vi.fn();
  document.body.innerHTML = '<div id="app"></div>';
  Object.defineProperty(document, 'modelContext', { configurable: true, value: { registerTool: (tool: any) => { tools[tool.name] = tool; } } });
  localStorage.clear();
  await import('../src/main');
  await waitFor(() => !!tools.stage_stack_program && document.querySelectorAll('.lesson-link').length === 14);
});
afterAll(() => { if (server?.pid && process.platform === 'win32') spawnSync('taskkill', ['/pid', String(server.pid), '/t', '/f'], { windowsHide: true, stdio: 'ignore' }); else server?.kill(); vi.unstubAllGlobals(); });

describe('real app against the local Python machine (DOM, not layout)', () => {
  it('opens on the first repair and input state, with all lessons keyboard reachable', async () => {
    expect(document.querySelector('h1')!.textContent).toBe('First light');
    expect(document.querySelector('#frame-label')!.textContent).toBe('INPUT');
    expect(document.querySelectorAll('.example-row[data-example]')).toHaveLength(3);
    expect(document.querySelector('[data-word="add"]')).toBeTruthy();
  });
  it('stages a real program, steps backward by keyboard, and checks extra tests', async () => {
    const state = await tools.stage_stack_program.execute({ program: '1 add' });
    expect(state.matches).toBe(3);
    expect(state.step).toBe(2);
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowLeft', altKey: true, bubbles: true }));
    expect(tools.read_stack_workshop.execute().step).toBe(1);
    click('[data-action="verify"]');
    await waitFor(() => !!document.querySelector('.verification.success'));
    expect(document.querySelector('#progress-count')!.textContent).toBe('1 / 14');
  });
  it('rejects invalid staged programs without overwriting the editor', async () => {
    await expect(tools.stage_stack_program.execute({ program: 'not-an-instruction' })).rejects.toThrow('Unknown word');
    await expect(tools.stage_stack_program.execute({ program: 5 })).rejects.toThrow('Expected a program');
    expect(tools.read_stack_workshop.execute().program).toBe('1 add');
  });
  it('switches lessons without carrying old traces or erasing saved code', async () => {
    click('[data-lesson="echo"]');
    await waitFor(() => document.querySelector('#busy-label')!.textContent === 'Live');
    expect(tools.read_stack_workshop.execute().step).toBe(0);
    await tools.stage_stack_program.execute({ program: 'dup add' });
    expect(tools.read_stack_workshop.execute().matches).toBe(3);
    click('[data-lesson="first-light"]');
    await waitFor(() => document.querySelector('#busy-label')!.textContent === 'Live');
    expect(tools.read_stack_workshop.execute().program).toBe('1 add');
  });
  it('operates the command palette by keyboard', async () => {
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'k', ctrlKey: true, bubbles: true }));
    const input = document.querySelector<HTMLInputElement>('#command-query')!;
    expect(input).toBeTruthy();
    input.value = 'Step forward'; input.dispatchEvent(new Event('input', { bubbles: true }));
    input.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
    expect((document.querySelector('#modal') as HTMLDialogElement).open).toBe(false);
    expect(tools.read_stack_workshop.execute().step).toBe(1);
  });
  it('edits custom examples and distinguishes a fit from a reference-backed repair', async () => {
    click('[data-action="sandbox"]');
    await waitFor(() => document.querySelector('#busy-label')!.textContent === 'Live');
    click('[data-action="edit-examples"]');
    const json = document.querySelector<HTMLTextAreaElement>('#example-json')!;
    json.value = JSON.stringify([{ input: [[3, 5]], output: [8] }, { input: [[]], output: [0] }]);
    click('[data-action="apply-examples"]');
    await waitFor(() => !(document.querySelector('#modal') as HTMLDialogElement).open && document.querySelector('#busy-label')!.textContent === 'Live');
    const state = await tools.stage_stack_program.execute({ program: 'sum' });
    expect(state.matches).toBe(2);
    click('[data-action="verify"]');
    await waitFor(() => document.querySelector('#verification')!.textContent!.includes('These examples agree.'));
    expect(document.querySelector('#test-description')!.textContent).toContain('Matches establish a fit');
  });
  it('opens real nested execution after cached live evaluation', async () => {
    await tools.stage_stack_program.execute({ program: 'sum' });
    click('[data-action="inspect"]');
    await waitFor(() => document.querySelectorAll('.inner-frame').length > 1);
    expect(document.querySelector('#modal-title')!.textContent).toBe('Inside fold');
    expect(document.querySelector('.inner-frame code')!.textContent).toBe('add');
    click('[data-action="close-modal"]');
  });
  it('provides authored hints, macro expansion, and persistent notes', async () => {
    click('[data-lesson="vocabulary"]');
    await waitFor(() => document.querySelector('#busy-label')!.textContent === 'Live');
    expect(document.querySelector('#word-detail')!.textContent).toContain('0 [ add ] fold');
    click('[data-tab="guide"]');
    click('[data-action="hint"]');
    await waitFor(() => document.querySelector('#hint-text')!.textContent!.includes('sq names'));
    click('[data-tab="notebook"]');
    const note = document.querySelector<HTMLTextAreaElement>('#lesson-note')!;
    note.value = 'Macros preserve the primitive execution.'; note.dispatchEvent(new Event('input', { bubbles: true }));
    expect(JSON.parse(localStorage.getItem('signal-house.v1')!).lessons.vocabulary.note).toContain('Macros preserve');
  });
});
