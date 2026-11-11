// Integration test for the worker *assembly* path: the main-thread sorters turn the worker entry
// and the shared sort primitives into source text and run it in a real worker. The golden replay
// test calls the worker functions directly; this one builds the exact workerSource string the
// sorters build and runs it in a Node worker_threads eval worker, proving the stringified shared
// core resolves inside the worker and the outputs stay byte-identical.
//
// The browser path differs only in Blob/URL creation around the same source string, so this covers
// the stringification contract (no imports, no module-scope references) for both environments.

import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { Worker } from 'node:worker_threads';

import { expect } from 'chai';

import { readGolden } from './gsplat-sort-golden/golden-codec.mjs';
import { buildLegacyScenarios, buildUnifiedScenarios } from './gsplat-sort-golden/scenarios.mjs';
import { normalizeRecord } from './gsplat-sort-golden/worker-driver.mjs';
import { sortCoreSource } from '../../src/scene/gsplat/gsplat-sort-core.js';
import { SortWorker } from '../../src/scene/gsplat/gsplat-sort-worker.js';
import { GSplatSortBinWeights } from '../../src/scene/gsplat-unified/gsplat-sort-bin-weights.js';
import { UnifiedSortWorker } from '../../src/scene/gsplat-unified/gsplat-unified-sort-worker.js';

const here = dirname(fileURLToPath(import.meta.url));
const workerPort = 'require("node:worker_threads").parentPort';
const goldenDir = join(here, 'gsplat-sort-golden');

const onceMessage = worker => new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error('timed out waiting for worker message')), 10000);
    worker.once('message', (message) => {
        clearTimeout(timer);
        resolve(message);
    });
    worker.once('error', (error) => {
        clearTimeout(timer);
        reject(error);
    });
});

// Posts a message, transferring any ArrayBuffers exactly like the main-thread sorters do.
const post = (worker, message) => {
    const transfer = [];
    for (const value of Object.values(message)) {
        if (value instanceof ArrayBuffer) transfer.push(value);
    }
    worker.postMessage(message, transfer);
};

const expectRecord = (actual, expected, label) => {
    expect(actual.count, `${label} count`).to.equal(expected.count);
    if (expected.version !== undefined) {
        expect(actual.version, `${label} version`).to.equal(expected.version);
    }
    expect(Array.from(actual.order), `${label} order`).to.deep.equal(Array.from(expected.order));
};

describe('sort worker source assembly in a real worker_threads eval worker', function () {
    this.timeout(30000);

    it('UnifiedSortWorker runs the stringified shared core and matches the golden output', async function () {
        // Mirrors GSplatUnifiedSorter's workerSource assembly.
        const workerSource = `
            ${sortCoreSource}
            const gsplatSortCore = createGSplatSortCore();
            const GSplatSortBinWeights = ${GSplatSortBinWeights.toString()};
            (${UnifiedSortWorker.toString()})(${workerPort})
        `;

        const golden = readGolden(readFileSync(join(goldenDir, 'unified-sort-golden.bin')));
        const scenarios = buildUnifiedScenarios().filter(s => ['linear-bins-100', 'log-depth-100', 'radial-100'].includes(s.name));

        const worker = new Worker(workerSource, { eval: true });
        try {
            for (const scenario of scenarios) {
                const baseline = golden.find(s => s.name === scenario.name);
                let recordIdx = 0;

                for (const message of scenario.messages) {
                    post(worker, structuredClone(message));
                    if (message.command === 'sort') {
                        // messages must stay in order and each sort result is awaited before the
                        // next job, so the awaits in this loop are intentional
                        // eslint-disable-next-line no-await-in-loop
                        const result = await onceMessage(worker);
                        expectRecord(
                            normalizeRecord(result),
                            baseline.records[recordIdx++],
                            `${scenario.name} record ${recordIdx - 1}`
                        );
                    }
                }

                expect(recordIdx, `${scenario.name} record count`).to.equal(baseline.records.length);
            }
        } finally {
            await worker.terminate();
        }
    });

    it('SortWorker runs the stringified shared core and matches the golden output', async function () {
        // Mirrors GSplatSorter's workerSource assembly.
        const workerSource =
            `${sortCoreSource}\nconst gsplatSortCore = createGSplatSortCore();\n` +
            `(${SortWorker.toString()})(${workerPort})`;

        const golden = readGolden(readFileSync(join(goldenDir, 'legacy-sort-golden.bin')));
        // derived-chunks-1300: an init message then two camera moves, all moving enough to sort.
        const scenario = buildLegacyScenarios().find(s => s.name === 'derived-chunks-1300');
        const baseline = golden.find(s => s.name === scenario.name);

        const worker = new Worker(workerSource, { eval: true });
        try {
            let recordIdx = 0;

            for (const message of scenario.messages) {
                post(worker, structuredClone(message));

                // the init (centers) message has no camera yet, so update() returns without a
                // result; only camera messages that pass the movement gate produce a post here
                if (message.cameraPosition) {
                    // each sort is awaited and its order buffer returned before the next message,
                    // mirroring the main-thread handoff, so the await here is intentional
                    // eslint-disable-next-line no-await-in-loop
                    const result = await onceMessage(worker);
                    expectRecord(
                        normalizeRecord(result),
                        baseline.records[recordIdx],
                        `${scenario.name} record ${recordIdx}`
                    );
                    recordIdx++;

                    // hand the order buffer back, as GSplatSorter does after each result
                    post(worker, { order: new ArrayBuffer(result.order.byteLength) });
                }
            }

            expect(recordIdx, 'record count').to.equal(baseline.records.length);
        } finally {
            await worker.terminate();
        }
    });
});
