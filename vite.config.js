import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "tailwindcss";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": "http://127.0.0.1:8000",
      "/ws": {
        target: "ws://127.0.0.1:8000",
        ws: true,
      },
    },
    watch: {
      ignored: ["**/public/demo/frauddetect_demo.mp4"],
    },
  },
  css: {
    postcss: {
      plugins: [tailwindcss(), (await import("autoprefixer")).default()],
    },
  },
});
