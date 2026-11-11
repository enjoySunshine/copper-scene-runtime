// Shared sorting core for the two Gaussian splatting workers:
//   - scene/gsplat/gsplat-sort-worker.js (GSplatSorter's legacy worker)
//   - scene/gsplat-unified/gsplat-unified-sort-worker.js (the unified renderer's worker)
//
// The two workers are not ordinary modules: on the main thread their entry functions are turned
// into source text with Function.prototype.toString, concatenated and run inside a Blob worker
// (browsers) or an eval worker (node worker_threads). The whole core is therefore wrapped in one
// self-contained factory function, createGSplatSortCore, which is stringified and injected into
// the worker scope (see gsplat-sorter.js and gsplat-unified-sorter.js). Keeping every primitive
// behind a single identifier matters for minified builds: esbuild renames a top-level function
// declaration consistently at both the injected declaration and the call site inside the
// stringified worker entry, whereas several separately named functions would be renamed in the
// core but keep their source names in the worker string. The same constraint governs the
// GSplatSortBinWeights class the unified sorter injects.
//
// The factory closes over nothing from this module and uses only language built-ins, so its
// toString output is valid standalone source. The arithmetic was factored out of the two worker
// implementations unchanged; the golden replay test (test/scene/gsplat-sort-golden.test.mjs) pins
// the workers' outputs to the pre-refactor results.

function createGSplatSortCore() {

    // Number of histogram bins both sorting paths divide the distance range into.
    const numBins = 32;

    // Binary search shared with the legacy worker's zero-distance lookup.
    const binarySearch = (m, n, compare_fn) => {
        while (m <= n) {
            const k = (n + m) >> 1;
            const cmp = compare_fn(k);
            if (cmp > 0) {
                m = k + 1;
            } else if (cmp < 0) {
                n = k - 1;
            } else {
                return k;
            }
        }
        return ~m;
    };

    // Number of key bits chosen from the splat count, clamped to a range that balances key
    // resolution against the counting-sort table size. Both workers use the identical formula.
    const compareBits = (numVertices) => {
        return Math.max(10, Math.min(20, Math.round(Math.log2(numVertices / 4))));
    };

    // (Re)allocates the per-splat key buffer and the counting-sort bucket table when their sizes
    // change, otherwise zeroes the bucket table for reuse.
    const allocSortBuffers = (distances, countBuffer, numVertices, bucketCount) => {
        if (!distances || distances.length !== numVertices) {
            distances = new Uint32Array(numVertices);
        }

        if (!countBuffer || countBuffer.length !== bucketCount) {
            countBuffer = new Uint32Array(bucketCount);
        } else {
            countBuffer.fill(0);
        }

        return { distances, countBuffer };
    };

    // Maps a normalized distance (in bins, 0..numBins) to a sort key through the per-bin
    // base/divider tables. `binBase` and `binDivider` may be plain arrays (legacy worker) or
    // Float32Arrays with a safety entry at index numBins (unified worker); the arithmetic is
    // identical either way.
    const linearSortKey = (d, binBase, binDivider) => {
        const bin = d >>> 0;
        return (binBase[bin] + binDivider[bin] * (d - bin)) >>> 0;
    };

    // Turns float distances, written over the key buffer, into log-spaced sort keys in place
    // (same as the projector shader). A positive float's bit pattern grows with its value almost
    // exactly as its log2, so keys taken from the bits give every doubling of distance the same
    // number of keys. The bits are measured from those of `nearValue`, and inverted so far splats
    // sort first.
    const logSortKeys = (nearValue, farValue, maxKey, distances, countBuffer, floatBits, floatBitsI32) => {
        floatBits[0] = nearValue;
        const minBits = floatBitsI32[0];
        floatBits[0] = farValue;
        const bitsRange = floatBitsI32[0] - minBits;
        const keyScale = bitsRange > 0 ? maxKey / bitsRange : 0;

        for (let i = 0; i < distances.length; i++) {
            const keyFromNear = (distances[i] - minBits) * keyScale;
            const sortKey = maxKey - (keyFromNear <= 0 ? 0 : (keyFromNear >= maxKey ? maxKey : keyFromNear >>> 0));
            distances[i] = sortKey;
            countBuffer[sortKey]++;
        }
    };

    // Same as logSortKeys for signed depths, whose negative values (the splats behind the camera)
    // take the last `behindKeys` keys, spaced the same way out to `behindValue`, so they sort after
    // the rest in the order of their signed depth. The sign picks each key's scale, clamp and
    // direction arithmetically: a branch on it runs at half the speed.
    const signedLogSortKeys = (nearValue, farValue, behindValue, behindKeys, maxKey, distances, countBuffer, floatBits, floatBitsI32) => {
        floatBits[0] = nearValue;
        const minBits = floatBitsI32[0];
        floatBits[0] = farValue;
        const bitsRange = floatBitsI32[0] - minBits;
        floatBits[0] = behindValue;
        const behindBitsRange = floatBitsI32[0] - minBits;

        // the nearest key in front of the camera, and the last behind it, counted from the first
        const nearestKey = maxKey - behindKeys;
        const lastBehindKey = behindKeys - 1;
        const keyScale = bitsRange > 0 ? nearestKey / bitsRange : 0;
        const behindKeyScale = behindBitsRange > 0 ? lastBehindKey / behindBitsRange : 0;
        const behindScaleStep = behindKeyScale - keyScale;
        const behindLimitStep = lastBehindKey - nearestKey;

        // read as int32, so the sign bit is never a float
        const keys = new Int32Array(distances.buffer, distances.byteOffset, distances.length);
        for (let i = 0; i < keys.length; i++) {
            const bits = keys[i];

            // 1 behind the camera, 0 in front; the bits of the magnitude are measured from the
            // near value's, and a key behind counts up from the one after the nearest, in front
            const behind = bits >>> 31;
            const keyFromNear = ((bits & 0x7fffffff) - minBits) * (keyScale + behind * behindScaleStep);
            const limit = nearestKey + behind * behindLimitStep;
            const key = keyFromNear <= 0 ? 0 : (keyFromNear >= limit ? limit : keyFromNear >>> 0);
            const sortKey = nearestKey + behind + (2 * behind - 1) * key;
            keys[i] = sortKey;
            countBuffer[sortKey]++;
        }
    };

    // Counting sort over the precomputed keys. With no indexMap the output indices are the
    // compact splat indices (legacy worker); with one they map through to work-buffer pixel
    // indices (unified worker). The overflow warning matches the unified worker's bounds
    // protection. `overflowState` is a mutable { value } flag so the once-only warning survives
    // across sorts without the core holding module state.
    const countingSort = (bucketCount, countBuffer, numVertices, distances, order, indexMap, overflowState) => {

        // accumulate counts
        for (let i = 1; i < bucketCount; i++) {
            countBuffer[i] += countBuffer[i - 1];
        }

        // fast check: after cumulative sum, last bucket = total valid splats
        // If less than numVertices, some sortKeys were out of bounds (AABB issue)
        const validCount = countBuffer[bucketCount - 1];
        if (indexMap && validCount !== numVertices && !overflowState.value) {
            overflowState.value = true;
            console.warn(`[SortWorker] ${numVertices - validCount} splats lost due to sortKey overflow. Check resource AABB bounds contain all the splats.`);
        }

        // build output array
        for (let i = 0; i < numVertices; i++) {
            const distance = distances[i];
            const destIndex = --countBuffer[distance];
            order[destIndex] = indexMap ? indexMap[i] : i;
        }
    };

    return {
        numBins,
        binarySearch,
        compareBits,
        allocSortBuffers,
        linearSortKey,
        logSortKeys,
        signedLogSortKeys,
        countingSort
    };
}

// Stringified source injected into the worker scope. The worker calls createGSplatSortCore() once
// and uses the returned object, so only this one identifier crosses the source-text boundary.
const sortCoreSource = createGSplatSortCore.toString();

export { createGSplatSortCore, sortCoreSource };
