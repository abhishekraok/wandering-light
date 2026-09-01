interface StartupSource {
  id: string;
  name: string;
  ready: boolean;
  payload_bytes: number;
  index: { status: string };
}

/** Choose a source that can render quickly instead of opening on missing data. */
export function chooseInitialSource<T extends StartupSource>(
  sources: readonly T[],
  rememberedId: string | null,
): T | undefined {
  const remembered = sources.find((source) => source.id === rememberedId);
  if (remembered?.ready) return remembered;

  const ready = sources.filter((source) => source.ready);
  ready.sort(
    (left, right) =>
      Number(right.index.status === "ready") - Number(left.index.status === "ready") ||
      left.payload_bytes - right.payload_bytes ||
      left.name.localeCompare(right.name),
  );
  return ready[0] ?? remembered ?? sources[0];
}
