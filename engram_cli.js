#!/usr/bin/env node
/**
 * Engram CLI helper for research_memory.py
 * 
 * Bridges Python → Engram Trace (Node.js)
 * 
 * Usage:
 *   node engram_cli.js remember '{"content":"...", "tags":[...], "metadata":{...}}'
 *   node engram_cli.js recall '{"query":"...", "limit":5}'
 */

import { EngramTrace } from '@terronex/engram-trace';
import { join } from 'node:path';
import { homedir } from 'node:os';
import { mkdirSync } from 'node:fs';

const RESEARCH_DIR = join(homedir(), '.karpathy-problem');
const ENGRAM_FILE = join(RESEARCH_DIR, 'research-brain.engram');

mkdirSync(RESEARCH_DIR, { recursive: true });

const trace = new EngramTrace({
    file: ENGRAM_FILE,
    embedder: { provider: 'local' },
    autoConsolidate: false,
    debug: false,
});

async function main() {
    const [action, argsJson] = process.argv.slice(2);
    const args = JSON.parse(argsJson || '{}');

    await trace.init();

    try {
        switch (action) {
            case 'remember': {
                const memory = await trace.remember(args.content, {
                    tags: args.tags || [],
                    metadata: args.metadata || {},
                    importance: args.importance || 0.5,
                });
                console.log(JSON.stringify({ success: true, id: memory.id }));
                break;
            }

            case 'recall': {
                const results = await trace.recall(args.query, {
                    limit: args.limit || 5,
                    tags: args.tags,
                });
                const memories = results.map(r => ({
                    content: r.memory.content,
                    score: r.score,
                    metadata: r.memory.metadata,
                    tags: r.memory.tags,
                }));
                console.log(JSON.stringify({ memories }));
                break;
            }

            case 'stats': {
                const stats = trace.stats();
                console.log(JSON.stringify(stats));
                break;
            }

            default:
                console.error(`Unknown action: ${action}`);
                process.exit(1);
        }
    } finally {
        await trace.save();
        await trace.close();
    }
}

main().catch(err => {
    console.error(err.message);
    process.exit(1);
});
