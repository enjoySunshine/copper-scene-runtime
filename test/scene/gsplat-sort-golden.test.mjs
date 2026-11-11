// Replays the deterministic sort scenarios against the current workers and compares every emitted
// message against the recorded golden baseline, element by element:
//   - the full order array
//   - count
//   - version (unified worker only)
//
// The golden files were recorded by generate-golden.mjs before the shared sort-core refactor.
// Behavior must stay identical, so this suite must pass unchanged afterwards.

import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { expect } from 'chai';


import { readGolden } from './gsplat-sort-golden/golden-codec.mjs';
import { buildLegacyScenarios, buildUnifiedScenarios } from './gsplat-sort-golden/scenarios.mjs';
import { normalizeRecord, runScenario } from './gsplat-sort-golden/worker-driver.mjs';
import { createGSplatSortCore } from '../../src/scene/gsplat/gsplat-sort-core.js';
import { SortWorker } from '../../src/scene/gsplat/gsplat-sort-worker.js';
import { GSplatSortBinWeights } from '../../src/scene/gsplat-unified/gsplat-sort-bin-weights.js';
import { UnifiedSortWorker } from '../../src/scene/gsplat-unified/gsplat-unified-sort-worker.js';

const here = dirname(fileURLToPath(import.meta.url));

const loadGolden = file => readGolden(readFileSync(join(here, 'gsplat-sort-golden', file)));

// Compares one replay record against the golden baseline, reporting the first differing entry.
const expectRecordEqual = (actual, expected, label) => {
    expect(actual.count, `${label}: count`).to.equal(expected.count);

    if (expected.version === undefined) {
        expect(actual.version, `${label}: version`).to.be.undefined;
    } else {
        expect(actual.version, `${label}: version`).to.equal(expected.version);
    }

    const actualOrder = actual.order;
    const expectedOrder = expected.order;
    expect(actualOrder.length, `${label}: order length`).to.equal(expectedOrder.length);

    for (let i = 0; i < expectedOrder.length; i++) {
        if (actualOrder[i] !== expectedOrder[i]) {
            expect.fail(`${label}: order differs first at index ${i}: ` +
                `actual ${actualOrder[i]} !== expected ${expectedOrder[i]}`);
        }
    }
};

describe('sort worker golden outputs', function () {

    // The workers' creators inject the bin weights class and the shared sort core into worker
    // scope as source text; emulate those globals for the direct function calls here.
    before(function () {
        globalThis.GSplatSortBinWeights = GSplatSortBinWeights;
        globalThis.gsplatSortCore = createGSplatSortCore();
    });

    after(function () {
        delete globalThis.GSplatSortBinWeights;
        delete globalThis.gsplatSortCore;
    });

    describe('legacy SortWorker (GSplatSorter)', function () {
        const scenarios = buildLegacyScenarios();
        const golden = loadGolden('legacy-sort-golden.bin');

        it('has a golden baseline for every scenario', function () {
            expect(golden.map(s => s.name)).to.deep.equal(scenarios.map(s => s.name));
        });

        for (let i = 0; i < scenarios.length; i++) {
            const scenario = scenarios[i];
            const baseline = golden[i];

            it(`reproduces the recorded order/count for "${scenario.name}"`, function () {
                const posted = runScenario(port => SortWorker(port), scenario, { returnOrders: true });

                expect(posted.length, 'number of worker messages').to.equal(baseline.records.length);
                for (let r = 0; r < baseline.records.length; r++) {
                    expectRecordEqual(
                        normalizeRecord(posted[r]),
                        baseline.records[r],
                        `${scenario.name} record ${r}`
                    );
                }
            });
        }
    });

    describe('UnifiedSortWorker (GSplatUnifiedSorter)', function () {
        const scenarios = buildUnifiedScenarios();
        const golden = loadGolden('unified-sort-golden.bin');

        it('has a golden baseline for every scenario', function () {
            expect(golden.map(s => s.name)).to.deep.equal(scenarios.map(s => s.name));
        });

        for (let i = 0; i < scenarios.length; i++) {
            const scenario = scenarios[i];
            const baseline = golden[i];

            it(`reproduces the recorded order/count/version for "${scenario.name}"`, function () {
                const posted = runScenario(port => UnifiedSortWorker(port), scenario);

                expect(posted.length, 'number of worker messages').to.equal(baseline.records.length);
                for (let r = 0; r < baseline.records.length; r++) {
                    expectRecordEqual(
                        normalizeRecord(posted[r]),
                        baseline.records[r],
                        `${scenario.name} record ${r}`
                    );
                }
            });
        }
    });
});
