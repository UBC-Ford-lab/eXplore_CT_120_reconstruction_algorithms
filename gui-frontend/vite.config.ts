import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Built into the Python package so `pip install` ships the page.
export default defineConfig({
  plugins: [react()],
  base: '/',
  build: { outDir: '../src/explore_ct120_recon/gui/static', emptyOutDir: true },
  server: { proxy: { '/api': 'http://127.0.0.1:8120' } },
})
