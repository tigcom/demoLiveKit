import { defineConfig } from 'vite'

// Vite config with dev server proxy to backend token server
export default defineConfig({
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: process.env.VITE_TOKEN_SERVER_URL || 'http://localhost:8000',
        changeOrigin: true,
        secure: false,
      }
    }
  }
})
