import { defineConfig } from 'vitest/config';
export default defineConfig({ test: { environment: 'happy-dom', environmentOptions: { happyDOM: { url: 'http://127.0.0.1:18018/' } }, include: ['tests/**/*.test.ts'], testTimeout: 15000, hookTimeout: 30000, fileParallelism: false } });
