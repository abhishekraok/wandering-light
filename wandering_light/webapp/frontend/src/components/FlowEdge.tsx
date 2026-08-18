import { BaseEdge, EdgeLabelRenderer, getBezierPath, type EdgeProps } from "@xyflow/react";

const LOOP_HEIGHT = 58;
const RECIPROCAL_OFFSET = 46;

/**
 * One edge, in the three shapes this graph actually produces.
 *
 * A default bezier draws nothing for a self-loop and draws the two halves of an
 * involution on top of each other -- which would hide exactly the structure the
 * graph exists to show. Loops get an arc over the node, reciprocal pairs get
 * pushed to opposite sides, everything else stays a plain curve.
 */
export function FlowEdge({
  id,
  source,
  target,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  label,
  data,
  selected,
}: EdgeProps) {
  const highlighted = Boolean(data?.highlighted);
  const reciprocal = Boolean(data?.reciprocal);
  const selfLoop = source === target;

  let path: string;
  let labelX: number;
  let labelY: number;

  if (selfLoop) {
    // Out of the right handle, over the top, back into the left handle.
    const left = targetX;
    const right = sourceX;
    const top = sourceY - LOOP_HEIGHT;
    path = `M ${right} ${sourceY} C ${right + 60} ${top}, ${left - 60} ${top}, ${left} ${targetY}`;
    labelX = (left + right) / 2;
    labelY = top + 6;
  } else if (reciprocal) {
    // Bow the two directions apart so both remain readable.
    const midX = (sourceX + targetX) / 2;
    const midY = (sourceY + targetY) / 2;
    const dx = targetX - sourceX;
    const dy = targetY - sourceY;
    const length = Math.hypot(dx, dy) || 1;
    const offsetX = (-dy / length) * RECIPROCAL_OFFSET;
    const offsetY = (dx / length) * RECIPROCAL_OFFSET;
    path = `M ${sourceX} ${sourceY} Q ${midX + offsetX} ${midY + offsetY} ${targetX} ${targetY}`;
    labelX = midX + offsetX * 0.75;
    labelY = midY + offsetY * 0.75;
  } else {
    const [bezier, centerX, centerY] = getBezierPath({
      sourceX,
      sourceY,
      sourcePosition,
      targetX,
      targetY,
      targetPosition,
    });
    path = bezier;
    labelX = centerX;
    labelY = centerY;
  }

  const stroke = highlighted ? "var(--accent)" : selfLoop ? "var(--warn)" : "var(--line)";
  return (
    <>
      <BaseEdge
        id={id}
        path={path}
        style={{ stroke, strokeWidth: highlighted || selected ? 2 : 1 }}
      />
      <EdgeLabelRenderer>
        <div
          className={`edge-label${highlighted ? " highlighted" : ""}${selfLoop ? " loop" : ""}`}
          style={{ transform: `translate(-50%, -50%) translate(${labelX}px, ${labelY}px)` }}
        >
          {label}
        </div>
      </EdgeLabelRenderer>
    </>
  );
}
