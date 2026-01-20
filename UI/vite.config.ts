import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// 说明：后端是 Cloudflare URL（跨域一般没问题）。
// 如果你未来改成本地后端（http://localhost:8000）且遇到 CORS，
// 可以再考虑用 Vite proxy 或后端加 CORS。
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: true
  }
});
