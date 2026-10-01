import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const target = (globalThis as any).process?.env?.VITE_API_PROXY || "http://127.0.0.1:8100";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5174,
    strictPort: true,
    proxy: {
      "/api": { target, changeOrigin: true },
      "/health": { target, changeOrigin: true },
      "/metrics": { target, changeOrigin: true },
    },
  },
  preview: {
    port: 5174,
    strictPort: true,
    proxy: {
      "/api": { target, changeOrigin: true },
      "/health": { target, changeOrigin: true },
      "/metrics": { target, changeOrigin: true },
    },
  },
});
