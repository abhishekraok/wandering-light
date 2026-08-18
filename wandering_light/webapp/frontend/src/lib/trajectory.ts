import type { Step } from "../types";

/** Replace one step, dropping everything after it. */
export function replaceStep(steps: string[], index: number, name: string): string[] {
  return [...steps.slice(0, index), name];
}

/** Drop one step and everything downstream of it. */
export function truncateAt(steps: string[], index: number): string[] {
  return steps.slice(0, index);
}

/** The state a trajectory ends on, or null when a step failed. */
export function finalState(steps: Step[]): Step | null {
  if (steps.length === 0) return null;
  const last = steps[steps.length - 1];
  return last.ok ? last : null;
}

/** Whether the executed trajectory lands on the target's wire form. */
export function reachesTarget(steps: Step[], rootWire: string, targetWire: string): boolean {
  const last = finalState(steps);
  if (last === null) return steps.length === 0 && rootWire === targetWire;
  return last.state?.wire === targetWire;
}
