import { defineConfig } from "vite";

export default defineConfig({
  base: "/",
  server: {
    host: "127.0.0.1",
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:8765",
    },
  },
  build: {
    outDir: "../../wandering_light/evals/trajectory_graph_web",
    emptyOutDir: true,
    target: "es2022",
  },
});
