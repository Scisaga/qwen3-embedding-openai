import { defineConfig } from "vite";

export default defineConfig({
  base: "/projector-static/",
  build: {
    outDir: "dist",
    emptyOutDir: true,
    sourcemap: false,
    rollupOptions: {
      output: {
        entryFileNames: "projector.js",
        chunkFileNames: "chunks/[name]-[hash].js",
        assetFileNames: (assetInfo) =>
          assetInfo.name?.endsWith(".css")
            ? "projector.css"
            : "assets/[name]-[hash][extname]",
      },
    },
  },
});
