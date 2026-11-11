// Deterministic input scenarios for the two Gaussian splatting sort workers.
//
// This module is the single source of inputs for both:
//   - generate-golden.mjs, which records the workers' outputs into binary golden files, and
//   - gsplat-sort-golden.test.mjs, which replays the same inputs after the sort-core refactor.
//
// Everything here is seeded PRNG data, so the scenarios reproduce bit-for-bit on any machine.

// Deterministic PRNG so the generated inputs are reproducible across machines and runs.
export const mulberry32 = (seed) => {
    let a = seed >>> 0;
    return () => {
        a |= 0;
        a = (a + 0x6D2B79F5) | 0;
        let t = Math.imul(a ^ (a >>> 15), 1 | a);
        // mulberry32's published mixing step; keep the exact expression so generated inputs stay
        // byte-identical to the recorded golden fixtures
        // eslint-disable-next-line operator-assignment
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
};

// ---------------------------------------------------------------------------
// Legacy SortWorker (GSplatSorter) scenarios
// ---------------------------------------------------------------------------

export const buildLegacyScenarios = () => {
    const rand = mulberry32(0x4753504C); // 'GSPl'
    const scenarios = [];

    const makeCenters = (count, spread) => {
        const centers = new Float32Array(count * 3);
        for (let i = 0; i < centers.length; i++) {
            centers[i] = (rand() - 0.5) * spread;
        }
        return centers;
    };

    // Scenario 1: random splats with explicit chunk bounds, several camera poses, including a
    // second camera update that is close enough to skip sorting.
    {
        const numSplats = 5000;
        const centers = makeCenters(numSplats, 40);
        const numChunks = Math.ceil(numSplats / 256);
        const chunks = new Float32Array(numChunks * 6);

        // derive min/max bounds per chunk from the centers
        for (let c = 0; c < numChunks; c++) {
            const start = c * 256;
            const end = Math.min(numSplats, start + 256);
            const mn = [Infinity, Infinity, Infinity];
            const mx = [-Infinity, -Infinity, -Infinity];
            for (let i = start; i < end; i++) {
                for (let a = 0; a < 3; a++) {
                    const v = centers[i * 3 + a];
                    if (v < mn[a]) mn[a] = v;
                    if (v > mx[a]) mx[a] = v;
                }
            }
            chunks.set(mn, c * 6);
            chunks.set(mx, c * 6 + 3);
        }

        const mapping = new Uint32Array(numSplats);
        for (let i = 0; i < numSplats; i++) mapping[i] = numSplats - 1 - i;
        const remapped = new Float32Array(numSplats * 3);
        for (let i = 0; i < numSplats; i++) {
            const src = mapping[i] * 3;
            remapped[i * 3] = centers[src];
            remapped[i * 3 + 1] = centers[src + 1];
            remapped[i * 3 + 2] = centers[src + 2];
        }

        scenarios.push({
            name: 'random-5000-with-chunks',
            messages: [
                {
                    order: new Uint32Array(numSplats).map((_, i) => i).buffer,
                    centers: centers.buffer,
                    chunks: chunks.buffer
                },
                { cameraPosition: { x: 10, y: 5, z: 20 }, cameraDirection: { x: 0.1, y: -0.2, z: -1 } },
                // move far enough to force another full sort
                { cameraPosition: { x: -15, y: -8, z: 25 }, cameraDirection: { x: -0.3, y: 0.4, z: -1 } },
                // sub-epsilon move: worker must skip, no new message
                { cameraPosition: { x: -15 + 0.0001, y: -8, z: 25 }, cameraDirection: { x: -0.3, y: 0.4, z: -1 } },
                // force a sort by replacing centers with a mapping applied
                { centers: remapped.buffer, mapping: mapping.buffer },
                { cameraPosition: { x: 0, y: 0, z: 30 }, cameraDirection: { x: 0, y: 0, z: -1 } },
                // mapping cleared
                { centers: centers.slice().buffer, mapping: null },
                // identical camera to the previous sort: the worker skips, no new message
                { cameraPosition: { x: 0, y: 0, z: 30 }, cameraDirection: { x: 0, y: 0, z: -1 } }
            ]
        });
    }

    // Scenario 2: no chunks provided - worker derives bounding spheres from centers itself.
    {
        const numSplats = 1300; // crosses a chunk boundary (256)
        const centers = makeCenters(numSplats, 10);
        scenarios.push({
            name: 'derived-chunks-1300',
            messages: [
                {
                    order: new Uint32Array(numSplats).map((_, i) => i).buffer,
                    centers: centers.buffer
                },
                { cameraPosition: { x: 3, y: -4, z: 7 }, cameraDirection: { x: 0, y: 0, z: -1 } },
                { cameraPosition: { x: 100, y: 100, z: 100 }, cameraDirection: { x: 1, y: 1, z: 1 } }
            ]
        });
    }

    // Scenario 3: tiny splat counts, including below the 10-bit bucket floor.
    for (const numSplats of [1, 2, 7, 64]) {
        const centers = makeCenters(numSplats, 5);
        scenarios.push({
            name: `small-${numSplats}`,
            messages: [
                {
                    order: new Uint32Array(numSplats).map((_, i) => i).buffer,
                    centers: centers.buffer
                },
                { cameraPosition: { x: 1, y: 2, z: 3 }, cameraDirection: { x: 0, y: 0, z: -1 } }
            ]
        });
    }

    // Scenario 4: coplanar / equal projected distances (range < 1e-6 branch).
    {
        const numSplats = 2000;
        const centers = new Float32Array(numSplats * 3);
        for (let i = 0; i < numSplats; i++) {
            centers[i * 3] = (rand() - 0.5) * 10;
            centers[i * 3 + 1] = (rand() - 0.5) * 10;
            centers[i * 3 + 2] = -5; // all at the same depth
        }
        scenarios.push({
            name: 'equal-depth-2000',
            messages: [
                {
                    order: new Uint32Array(numSplats).map((_, i) => i).buffer,
                    centers: centers.buffer
                },
                { cameraPosition: { x: 0, y: 0, z: 0 }, cameraDirection: { x: 0, y: 0, z: -1 } }
            ]
        });
    }

    // Scenario 5: non-finite center values exercising the sanitization in the derived-chunks path.
    {
        const numSplats = 600;
        const centers = makeCenters(numSplats, 8);
        for (let i = 0; i < 30; i++) {
            const idx = Math.floor(rand() * numSplats) * 3 + Math.floor(rand() * 3);
            centers[idx] = i % 2 === 0 ? NaN : Infinity;
        }
        scenarios.push({
            name: 'non-finite-centers-600',
            messages: [
                {
                    order: new Uint32Array(numSplats).map((_, i) => i).buffer,
                    centers: centers.buffer
                },
                { cameraPosition: { x: -2, y: 3, z: 9 }, cameraDirection: { x: 0.2, y: 0.1, z: -1 } }
            ]
        });
    }

    // Scenario 6: a large count giving more than the default compare-bit range headroom.
    {
        const numSplats = 60000;
        const centers = makeCenters(numSplats, 100);
        scenarios.push({
            name: 'large-60000',
            messages: [
                {
                    order: new Uint32Array(numSplats).map((_, i) => i).buffer,
                    centers: centers.buffer
                },
                { cameraPosition: { x: 20, y: -30, z: 80 }, cameraDirection: { x: 0, y: 0, z: -1 } }
            ]
        });
    }

    // Scenario 7: splats behind and in front of the camera to exercise the findZero count path.
    {
        const numSplats = 3000;
        const centers = new Float32Array(numSplats * 3);
        for (let i = 0; i < numSplats; i++) {
            centers[i * 3] = (rand() - 0.5) * 4;
            centers[i * 3 + 1] = (rand() - 0.5) * 4;
            centers[i * 3 + 2] = (rand() - 0.5) * 30; // spans the camera plane at z=0
        }
        scenarios.push({
            name: 'camera-inside-3000',
            messages: [
                {
                    order: new Uint32Array(numSplats).map((_, i) => i).buffer,
                    centers: centers.buffer
                },
                { cameraPosition: { x: 0, y: 0, z: 0 }, cameraDirection: { x: 0, y: 0, z: -1 } }
            ]
        });
    }

    return scenarios;
};

// ---------------------------------------------------------------------------
// UnifiedSortWorker (GSplatUnifiedSorter) scenarios
// ---------------------------------------------------------------------------

export const buildUnifiedScenarios = () => {
    const rand = mulberry32(0x554E4946); // 'UNIF'
    const scenarios = [];

    const makePositions = (count, spread) => {
        const positions = [];
        for (let i = 0; i < count; i++) {
            positions.push([
                (rand() - 0.5) * spread,
                (rand() - 0.5) * spread,
                (rand() - 0.5) * spread
            ]);
        }
        return positions;
    };

    const aabbOf = (positions) => {
        const aabbMin = [Infinity, Infinity, Infinity];
        const aabbMax = [-Infinity, -Infinity, -Infinity];
        for (const p of positions) {
            for (let a = 0; a < 3; a++) {
                if (p[a] < aabbMin[a]) aabbMin[a] = p[a];
                if (p[a] > aabbMax[a]) aabbMax[a] = p[a];
            }
        }
        return { aabbMin, aabbMax };
    };

    // One scenario per key mode, with several splat counts.
    const modes = [
        { radialSorting: false, logDepthKeys: false, nearClip: 0.01, tag: 'linear-bins' },
        { radialSorting: false, logDepthKeys: true, nearClip: 0.01, tag: 'log-depth' },
        { radialSorting: true, logDepthKeys: false, nearClip: 0.01, tag: 'radial' },
        { radialSorting: false, logDepthKeys: true, nearClip: 0, tag: 'log-depth-zero-near' }
    ];

    const counts = [4, 100, 5000, 30000];

    let idSeq = 0;
    for (const mode of modes) {
        for (const count of counts) {
            const positions = makePositions(count, 60);
            const centers = new Float32Array(positions.flat());
            const { aabbMin, aabbMax } = aabbOf(positions);
            const id = ++idSeq;

            // several sort jobs on the same centers: different camera directions/positions and
            // scales/offsets, including negative scale
            const jobs = [
                {
                    transformedPosition: { x: 0, y: 0, z: 0 },
                    transformedDirection: { x: 0, y: 0, z: 1 },
                    offset: 0,
                    scale: 1
                },
                {
                    transformedPosition: { x: 5, y: -7, z: 11 },
                    transformedDirection: { x: 0.3, y: -0.2, z: 1 },
                    offset: 2.5,
                    scale: 1.5
                },
                {
                    transformedPosition: { x: -10, y: 20, z: -30 },
                    transformedDirection: { x: -0.1, y: 0.4, z: 1 },
                    offset: -4,
                    scale: -1
                }
            ];

            const messages = [
                { command: 'addCenters', id, centers: centers.buffer },
                {
                    command: 'intervals',
                    ids: [id],
                    pixelOffsets: [[0]],
                    intervals: [[]],
                    totalActiveSplats: count,
                    version: 1
                }
            ];
            for (const job of jobs) {
                messages.push({
                    command: 'sort',
                    sortParams: [{ ...job, aabbMin, aabbMax }],
                    radialSorting: mode.radialSorting,
                    nearClip: mode.nearClip,
                    logDepthKeys: mode.logDepthKeys,
                    order: new Uint32Array(count).buffer
                });
            }

            scenarios.push({
                name: `${mode.tag}-${count}`,
                messages
            });
        }
    }

    // Multi-placement scenario: two center sets, explicit intervals and pixel offsets, exercising
    // the indexMap path and more than one sortParams entry per job.
    {
        const countA = 1500;
        const countB = 2500;
        const posA = makePositions(countA, 20);
        const posB = makePositions(countB, 40);
        const centersA = new Float32Array(posA.flat());
        const centersB = new Float32Array(posB.flat());
        const boxA = aabbOf(posA);
        const boxB = aabbOf(posB);
        const idA = ++idSeq;
        const idB = ++idSeq;

        // intervals: A uses a sub-range, B uses two ranges; pixel offsets leave a gap
        const intervalA = [100, 900];
        const intervalB = [0, 1000, 2000, countB];
        const activeA = intervalA[1] - intervalA[0];
        const activeB = (intervalB[1] - intervalB[0]) + (intervalB[3] - intervalB[2]);
        const total = activeA + activeB;

        const messages = [
            { command: 'addCenters', id: idA, centers: centersA.buffer },
            { command: 'addCenters', id: idB, centers: centersB.buffer },
            {
                command: 'intervals',
                ids: [idA, idB],
                pixelOffsets: [[10], [500, 2500]],
                intervals: [intervalA, intervalB],
                totalActiveSplats: total,
                version: 7
            }
        ];

        for (const mode of [
            { radialSorting: false, logDepthKeys: false },
            { radialSorting: false, logDepthKeys: true },
            { radialSorting: true, logDepthKeys: false }
        ]) {
            messages.push({
                command: 'sort',
                sortParams: [
                    {
                        transformedPosition: { x: 1, y: 2, z: 3 },
                        transformedDirection: { x: 0, y: 0, z: 1 },
                        offset: 0,
                        scale: 1,
                        aabbMin: boxA.aabbMin,
                        aabbMax: boxA.aabbMax
                    },
                    {
                        transformedPosition: { x: -4, y: 5, z: -6 },
                        transformedDirection: { x: 0.1, y: -0.1, z: 1 },
                        offset: 1,
                        scale: 2,
                        aabbMin: boxB.aabbMin,
                        aabbMax: boxB.aabbMax
                    }
                ],
                radialSorting: mode.radialSorting,
                nearClip: 0.05,
                logDepthKeys: mode.logDepthKeys,
                order: new Uint32Array(total).buffer
            });
        }

        // a new intervals version rebuilds the indexMap, then one more sort
        messages.push({
            command: 'intervals',
            ids: [idA, idB],
            pixelOffsets: [[0], [0]],
            intervals: [[], []],
            totalActiveSplats: countA + countB,
            version: 8
        });
        messages.push({
            command: 'sort',
            sortParams: [
                {
                    transformedPosition: { x: 0, y: 0, z: 0 },
                    transformedDirection: { x: 0, y: 0, z: 1 },
                    offset: 0,
                    scale: 1,
                    aabbMin: boxA.aabbMin,
                    aabbMax: boxA.aabbMax
                },
                {
                    transformedPosition: { x: 0, y: 0, z: 0 },
                    transformedDirection: { x: 0, y: 0, z: 1 },
                    offset: 0,
                    scale: 1,
                    aabbMin: boxB.aabbMin,
                    aabbMax: boxB.aabbMax
                }
            ],
            radialSorting: false,
            nearClip: 0.05,
            logDepthKeys: false,
            order: new Uint32Array(countA + countB).buffer
        });

        // remove one center source; later scenarios use fresh workers anyway
        messages.push({ command: 'removeCenters', id: idA });

        scenarios.push({ name: 'multi-placement-intervals', messages });
    }

    return scenarios;
};
