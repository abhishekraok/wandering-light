import { describe, expect, it } from "vitest";

import { chooseInitialSource } from "./startup";

const source = (
  id: string,
  ready: boolean,
  payloadBytes: number,
  status = "idle",
) => ({ id, name: id, ready, payload_bytes: payloadBytes, index: { status } });

describe("chooseInitialSource", () => {
  it("does not restore a remembered source whose payload is missing", () => {
    const missing = source("missing", false, 0);
    const small = source("small", true, 10);
    const large = source("large", true, 1_000);

    expect(chooseInitialSource([missing, large, small], missing.id)?.id).toBe("small");
  });

  it("prefers an already indexed source and preserves a ready remembered source", () => {
    const remembered = source("remembered", true, 1_000);
    const indexed = source("indexed", true, 10_000, "ready");

    expect(chooseInitialSource([remembered, indexed], null)?.id).toBe("indexed");
    expect(chooseInitialSource([remembered, indexed], remembered.id)?.id).toBe("remembered");
  });
});
