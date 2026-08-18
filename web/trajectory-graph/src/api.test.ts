import { describe, expect, it } from "vitest";

import { apiErrorMessage } from "./api";

describe("apiErrorMessage", () => {
  it("uses FastAPI detail when it is a string", () => {
    expect(apiErrorMessage(422, "Unprocessable Entity", { detail: "invalid task cursor" })).toBe(
      "invalid task cursor",
    );
  });

  it("falls back to the HTTP status for structured or invalid detail", () => {
    expect(apiErrorMessage(500, "Internal Server Error", { detail: [{ msg: "bad" }] })).toBe(
      "500 Internal Server Error",
    );
    expect(apiErrorMessage(429, "Too Many Requests", undefined)).toBe("429 Too Many Requests");
  });
});
