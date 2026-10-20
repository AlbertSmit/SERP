#!/usr/bin/env node
// Compatibility entry point for existing MCP client configurations and imports.
import { pathToFileURL } from 'node:url';
import { startMcpServer } from './src/mcp/server.js';

export { SQLiteContextEngine } from './src/engine/engine.js';
export { tokenize, gramTokens, countTokens } from './src/engine/tokenize.js';

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  await startMcpServer();
}
