// Records the current sort workers' outputs for the deterministic scenarios into binary golden
// files. Run before the sort-core refactor, then re-run the replay test after it.
//
// Run with: node test/scene/gsplat-sort-golden/generate-golden.mjs
//
// The script exits non-zero if the golden files already exist, so an accidental regeneration
// overwriting a recorded baseline does not go unnoticed. Delete the files deliberately first.

import { writeFileSync, existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';


import { GoldenWriter } from './golden-codec.mjs';
import { buildLegacyScenarios, buildUnifiedScenarios } from './scenarios.mjs';
import { normalizeRecord, runScenario } from './worker-driver.mjs';
import { SortWorker } from '../../../src/scene/gsplat/gsplat-sort-worker.js';
import { GSplatSortBinWeights } from '../../../src/scene/gsplat-unified/gsplat-sort-bin-weights.js';
import { UnifiedSortWorker } from '../../../src/scene/gsplat-unified/gsplat-unified-sort-worker.js';

const here = dirname(fileURLToPath(import.meta.url));

// The worker's creator injects the bin weights class into the worker's scope as source text.
globalThis.GSplatSortBinWeights = GSplatSortBinWeights;

const writeGolden = (file, scenarios, workerFactory, options) => {
    const target = join(here, file);
    if (existsSync(target)) {
        console.error(`Refusing to overwrite existing golden file: ${target}`);
        console.error('Delete it deliberately if a new baseline is intended.');
        process.exit(1);
    }

    const writer = new GoldenWriter();
    let totalRecords = 0;
    for (const scenario of scenarios) {
        const posted = runScenario(workerFactory, scenario, options);
        writer.beginScenario(scenario.name);
        for (const message of posted) {
            writer.writeRecord(normalizeRecord(message));
            totalRecords++;
        }
        writer.endScenario();
    }

    writeFileSync(target, writer.toBuffer());
    console.log(`${file}: ${scenarios.length} scenarios, ${totalRecords} recorded worker messages`);
};

const main = () => {
    writeGolden(
        'legacy-sort-golden.bin',
        buildLegacyScenarios(),
        port => SortWorker(port),
        { returnOrders: true }
    );

    writeGolden(
        'unified-sort-golden.bin',
        buildUnifiedScenarios(),
        port => UnifiedSortWorker(port),
        { returnOrders: false }
    );
};

main();
