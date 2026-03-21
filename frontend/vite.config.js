import { defineConfig } from "vite";

export default defineConfig({
  base: "/projector-static/",
  build: {
    outDir: "dist",
    emptyOutDir: true,
    sourcemap: false,
  },
});
