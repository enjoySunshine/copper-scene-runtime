// `myself` is the worker's message port: `self` in a web worker, or the parentPort of a Node
// worker_threads worker. The creator passes it in as source text, so that bundlers never see the
// Node-only require.
//
// The shared sorting core is injected into the worker scope by gsplat-unified-sorter.js as source
// text and instantiated into the gsplatSortCore global; see gsplat-sort-core.js for the
// stringification constraints. GSplatSortBinWeights is injected the same way by the sorter.
/* global gsplatSortCore, GSplatSortBinWeights */
function UnifiedSortWorker(myself) {

    // cache of centers for each splat id
    const centersMap = new Map();
    let centersData;
    let distances;
    let countBuffer;
    let indexMap;

    // Sorting mode: false = forward vector (directional), true = radial distance (for cubemaps)
    let _radialSort = false;

    // The camera's near clip, which log spaced sort keys start inside of
    let _nearClip = 0;

    // Whether a directional sort's keys are log spaced: a perspective, non-fisheye camera
    let _logDepthKeys = false;

    // Flag to warn only once about sortKey overflow
    let _warnedSortKeyOverflow = false;

    // Camera-relative bin-based precision optimization.
    // Arrays are size 33 (numBins + 1) to include a safety entry at index 32.
    // This handles the edge case where floating point calculation (d >>> 0) produces
    // bin index 32 instead of 31. The GPU compute shader avoids this by clamping,
    // but the CPU path uses truncation which can overflow. The safety entry at [32]
    // ensures valid array access without bounds checking in the hot loop.
    const numBins = gsplatSortCore.numBins;
    const binBase = new Float32Array(numBins + 1);
    const binDivider = new Float32Array(numBins + 1);

    // Shared bin weights utility (class is injected via stringification from main thread)
    const binWeightsUtil = new GSplatSortBinWeights();

    /**
     * Unpacks interleaved bin weights received from main thread into separate arrays.
     * Called once per sort request (not per-splat).
     *
     * @param {Float32Array} binWeights - Interleaved [base0, div0, base1, div1, ...]
     */
    const unpackBinWeights = (binWeights) => {
        for (let i = 0; i < numBins; i++) {
            binBase[i] = binWeights[i * 2];
            binDivider[i] = binWeights[i * 2 + 1];
        }
        // Safety entry for edge case where bin >= numBins due to floating point
        binBase[numBins] = binBase[numBins - 1] + binDivider[numBins - 1];
        binDivider[numBins] = 0;
    };

    // Sort key evaluation iterates only active splats (padding is excluded).
    // The indexMap (built on intervals) provides the work-buffer pixel mapping.
    const evaluateSortKeysCommon = (sortParams, minDist, range, distances, countBuffer, centersData, processSplatFn) => {
        const { ids, intervals } = centersData;

        // pre-calculate inverse bin range
        const invBinRange = numBins / range;
        let compactIdx = 0;

        // loop over all the splat placements
        for (let paramIdx = 0; paramIdx < sortParams.length; paramIdx++) {
            const params = sortParams[paramIdx];

            // source centers
            const id = ids[paramIdx];
            const centers = centersMap.get(id);
            if (!centers) {
                console.error('UnifiedSortWorker: No centers found for id', id);
            }

            // Use provided intervals or process all centers
            const intervalsArray = intervals[paramIdx].length > 0 ? intervals[paramIdx] : [0, centers.length / 3];

            // loop over all intervals of centers
            for (let i = 0; i < intervalsArray.length; i += 2) {
                const intervalStart = intervalsArray[i] * 3;
                const intervalEnd = intervalsArray[i + 1] * 3;

                // Process each center in this interval using the provided function
                compactIdx = processSplatFn(centers, params, intervalStart, intervalEnd, compactIdx,
                    invBinRange, minDist, range, distances, countBuffer);
            }
        }
    };

    const evaluateSortKeysLinear = (sortParams, minDist, range, distances, countBuffer, centersData) => {
        evaluateSortKeysCommon(sortParams, minDist, range, distances, countBuffer, centersData,
            (centers, params, intervalStart, intervalEnd, compactIdx, invBinRange, minDist, range, distances, countBuffer) => {
                // camera related params
                const { transformedDirection, offset, scale } = params;
                const dx = transformedDirection.x;
                const dy = transformedDirection.y;
                const dz = transformedDirection.z;

                // pre-calculate camera related constants
                const sdx = dx * scale;
                const sdy = dy * scale;
                const sdz = dz * scale;
                const add = offset - minDist;

                // Process each center in this interval
                for (let srcIndex = intervalStart; srcIndex < intervalEnd; srcIndex += 3) {
                    const x = centers[srcIndex];
                    const y = centers[srcIndex + 1];
                    const z = centers[srcIndex + 2];

                    const dist = x * sdx + y * sdy + z * sdz + add;

                    // Bin-based mapping
                    const d = dist * invBinRange;
                    const sortKey = gsplatSortCore.linearSortKey(d, binBase, binDivider);

                    distances[compactIdx++] = sortKey;
                    countBuffer[sortKey]++;
                }

                return compactIdx;
            });
    };

    // scratch for reading a float's bit pattern, see logSortKeys
    const floatBits = new Float32Array(1);
    const floatBitsI32 = new Int32Array(floatBits.buffer);

    // Radial keys from the squared distance, whose bits follow log2 as well.
    const evaluateSortKeysRadial = (sortParams, nearDist, maxDist, maxKey, distances, countBuffer, centersData) => {
        const squaredDistances = new Float32Array(distances.buffer, 0, distances.length);

        evaluateSortKeysCommon(sortParams, 0, maxDist, distances, countBuffer, centersData,
            (centers, params, intervalStart, intervalEnd, compactIdx) => {
                // camera related params
                const { transformedPosition, scale } = params;
                const scaleSq = scale * scale;

                // camera position in local space
                const cx = transformedPosition.x;
                const cy = transformedPosition.y;
                const cz = transformedPosition.z;

                // Process each center in this interval
                for (let srcIndex = intervalStart; srcIndex < intervalEnd; srcIndex += 3) {
                    const dx = centers[srcIndex] - cx;
                    const dy = centers[srcIndex + 1] - cy;
                    const dz = centers[srcIndex + 2] - cz;

                    // World-space squared distance from camera
                    squaredDistances[compactIdx++] = (dx * dx + dy * dy + dz * dz) * scaleSq;
                }

                return compactIdx;
            });

        gsplatSortCore.logSortKeys(nearDist * nearDist, maxDist * maxDist, maxKey, distances, countBuffer, floatBits, floatBitsI32);
    };

    // Depth keys of a perspective camera, from the signed depth in front of it. The sort direction
    // is the camera's +Z, which points backwards, so the range runs from -minDist in front of the
    // camera to maxDist behind it. Splats closer than `nearDist` on either side share a key. A shader
    // can move splats from behind the camera into view (the sort reads the centers before any
    // modifier), so those keep their order too, in a sixteenth of the keys.
    const evaluateSortKeysDepth = (sortParams, nearDist, minDist, maxDist, maxKey, distances, countBuffer, centersData) => {
        const depths = new Float32Array(distances.buffer, 0, distances.length);

        evaluateSortKeysCommon(sortParams, minDist, -minDist, distances, countBuffer, centersData,
            (centers, params, intervalStart, intervalEnd, compactIdx) => {
                // camera related params
                const { transformedDirection, offset, scale } = params;
                const sdx = transformedDirection.x * scale;
                const sdy = transformedDirection.y * scale;
                const sdz = transformedDirection.z * scale;

                // Process each center in this interval
                for (let srcIndex = intervalStart; srcIndex < intervalEnd; srcIndex += 3) {
                    depths[compactIdx++] = -(centers[srcIndex] * sdx + centers[srcIndex + 1] * sdy + centers[srcIndex + 2] * sdz + offset);
                }

                return compactIdx;
            });

        gsplatSortCore.signedLogSortKeys(nearDist, -minDist, maxDist, (maxKey + 1) >>> 4, maxKey, distances, countBuffer, floatBits, floatBitsI32);
    };

    // compute min/max effective distance using 8-corner local AABB projection per splat
    const computeEffectiveDistanceRangeLinear = (sortParams) => {
        let minDist = Infinity;
        let maxDist = -Infinity;

        for (let paramIdx = 0; paramIdx < sortParams.length; paramIdx++) {
            const params = sortParams[paramIdx];
            const { transformedDirection, offset, scale, aabbMin, aabbMax } = params;
            const dx = transformedDirection.x;
            const dy = transformedDirection.y;
            const dz = transformedDirection.z;

            // For a direction d and AABB [min,max], the min/max of dot(d, p) over the box
            // is obtained by picking min/max per component based on the sign of d
            const pxMin = dx >= 0 ? aabbMin[0] : aabbMax[0];
            const pyMin = dy >= 0 ? aabbMin[1] : aabbMax[1];
            const pzMin = dz >= 0 ? aabbMin[2] : aabbMax[2];
            const pxMax = dx >= 0 ? aabbMax[0] : aabbMin[0];
            const pyMax = dy >= 0 ? aabbMax[1] : aabbMin[1];
            const pzMax = dz >= 0 ? aabbMax[2] : aabbMin[2];

            const dMin = pxMin * dx + pyMin * dy + pzMin * dz;
            const dMax = pxMax * dx + pyMax * dy + pzMax * dz;

            const eMin = dMin * scale + offset;
            const eMax = dMax * scale + offset;

            // handle negative scale by swapping
            const localMin = Math.min(eMin, eMax);
            const localMax = Math.max(eMin, eMax);

            if (localMin < minDist) minDist = localMin;
            if (localMax > maxDist) maxDist = localMax;
        }

        if (minDist === Infinity) {
            minDist = 0;
            maxDist = 0;
        }
        return { minDist, maxDist };
    };

    // compute min/max radial distance from camera to AABB corners (for radial sort)
    const computeEffectiveDistanceRangeRadial = (sortParams) => {
        let maxDist = -Infinity;

        for (let paramIdx = 0; paramIdx < sortParams.length; paramIdx++) {
            const params = sortParams[paramIdx];
            const { transformedPosition, scale, aabbMin, aabbMax } = params;
            const cx = transformedPosition.x;
            const cy = transformedPosition.y;
            const cz = transformedPosition.z;

            // Check all 8 corners of the AABB for max radial distance
            for (let i = 0; i < 8; i++) {
                const px = (i & 1) ? aabbMax[0] : aabbMin[0];
                const py = (i & 2) ? aabbMax[1] : aabbMin[1];
                const pz = (i & 4) ? aabbMax[2] : aabbMin[2];

                const dx = px - cx;
                const dy = py - cy;
                const dz = pz - cz;

                const distSq = dx * dx + dy * dy + dz * dz;
                const dist = Math.sqrt(distSq) * scale;

                if (dist > maxDist) maxDist = dist;
            }
        }

        // For radial sort, minDist is always 0 (camera is the origin of radial distances)
        const minDist = 0;
        if (maxDist < 0) {
            maxDist = 0;
        }
        return { minDist, maxDist };
    };

    const sort = (sortParams, order, centersData) => {
        const sortStartTime = performance.now();

        // distance bounds from AABB projections per splat
        const { minDist, maxDist } = _radialSort ?
            computeEffectiveDistanceRangeRadial(sortParams) :
            computeEffectiveDistanceRangeLinear(sortParams);

        const numVertices = centersData.totalActiveSplats;

        // calculate number of bits needed to store sorting result
        const compareBits = gsplatSortCore.compareBits(numVertices);

        // Radial keys, and the depth keys of a perspective camera, are log spaced (an orthographic
        // or fisheye camera can draw splats at or behind it, so its depth keys keep the bins). They
        // take 16 bits whatever the splat count: they are spread over every doubling of distance
        // from inside the near clip out to the scene, and 16 bits give each doubling a few thousand
        // keys, where fewer leave the splats of a small scene far from the camera sharing keys.
        // More would spread the counting sort over a count table too big for the cache
        const logKeys = _radialSort || _logDepthKeys;
        const keyBits = logKeys ? 16 : compareBits;

        const bucketCount = 2 ** keyBits + 1;

        // create distance buffer
        const buffers = gsplatSortCore.allocSortBuffers(distances, countBuffer, numVertices, bucketCount);
        distances = buffers.distances;
        countBuffer = buffers.countBuffer;

        // Log keys start ten doublings inside the near clip: the splats closer than it are drawn
        // too, with their depth clamped, so they need an order. An orthographic near clip can be
        // zero or negative, so a share of the range stands in for it then (radial only).
        const nearDist = (_nearClip > 0 ? _nearClip : maxDist * 1e-6) / 1024;

        if (_radialSort) {
            evaluateSortKeysRadial(sortParams, nearDist, maxDist, bucketCount - 1, distances, countBuffer, centersData);
        } else if (logKeys) {
            evaluateSortKeysDepth(sortParams, nearDist, minDist, maxDist, bucketCount - 1, distances, countBuffer, centersData);
        } else {
            const range = maxDist - minDist;

            // Set up camera-relative bin weighting for near-camera precision (using shared utility)
            const cameraBin = GSplatSortBinWeights.computeCameraBin(minDist, range);

            // Compute bin weights locally using shared utility
            const binWeights = binWeightsUtil.compute(cameraBin, bucketCount);
            unpackBinWeights(binWeights);

            evaluateSortKeysLinear(sortParams, minDist, range, distances, countBuffer, centersData);
        }

        // the warning flag is shared through a mutable holder so it stays inside the stringifiable
        // shared core; copy its (possibly updated) value back afterwards
        const overflowState = { value: _warnedSortKeyOverflow };
        gsplatSortCore.countingSort(bucketCount, countBuffer, numVertices, distances, order, indexMap, overflowState);
        _warnedSortKeyOverflow = overflowState.value;

        const count = numVertices;

        // send results
        const sortTime = performance.now() - sortStartTime;
        const transferList = [order.buffer];
        const response = {
            order: order.buffer,
            count,
            version: centersData.version,
            sortTime: sortTime
        };

        myself.postMessage(response, transferList);
    };

    /**
     * Builds the indexMap that maps compact splat index to work-buffer pixel index.
     * Called once when the intervals message arrives (world state change), not per-sort.
     *
     * @param {object} data - The intervals message data containing layout metadata.
     */
    const buildIndexMap = (data) => {
        const { ids, pixelOffsets, intervals, totalActiveSplats } = data;

        if (!indexMap || indexMap.length < totalActiveSplats) {
            indexMap = new Uint32Array(totalActiveSplats);
        }

        let compactIdx = 0;
        for (let paramIdx = 0; paramIdx < ids.length; paramIdx++) {
            const centers = centersMap.get(ids[paramIdx]);
            const offsets = pixelOffsets[paramIdx];
            const intervalsArray = intervals[paramIdx].length > 0 ? intervals[paramIdx] : [0, centers.length / 3];

            for (let i = 0; i < intervalsArray.length; i += 2) {
                let workBufferIndex = offsets[i / 2];
                const count = intervalsArray[i + 1] - intervalsArray[i];
                for (let j = 0; j < count; j++) {
                    indexMap[compactIdx++] = workBufferIndex++;
                }
            }
        }
    };

    myself.addEventListener('message', (message) => {
        const msgData = message.data ?? message;

        switch (msgData.command) {

            // add centers to map
            case 'addCenters': {
                centersMap.set(msgData.id, new Float32Array(msgData.centers));
                break;
            }

            // remove centers from map
            case 'removeCenters': {
                centersMap.delete(msgData.id);
                break;
            }

            // sort
            case 'sort': {
                _radialSort = msgData.radialSorting || false;
                _nearClip = msgData.nearClip;
                _logDepthKeys = msgData.logDepthKeys || false;
                const order = new Uint32Array(msgData.order);
                sort(msgData.sortParams, order, centersData);
                break;
            }

            // intervals
            case 'intervals': {
                centersData = msgData;
                buildIndexMap(centersData);
                break;
            }
        }
    });
}

export { UnifiedSortWorker };
