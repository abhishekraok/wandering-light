import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Built assets land in ../static, which the FastAPI app serves directly, so a
// production run is `npm run build` once and then `python -m wandering_light.webapp`.
export default defineConfig({
  plugins: [react()],
  build: { outDir: "../static", emptyOutDir: true },
  server: {
    port: 5173,
    proxy: { "/api": "http://127.0.0.1:8765" },
  },
  test: {
    environment: "jsdom",
    environmentMatchGlobs: [["src/lib/**", "node"]],
    globals: true,
    setupFiles: ["./src/test-setup.ts"],
  },
});
