// Binary codec for the recorded sort-worker golden outputs.
//
// Only the fields the refactor promises to preserve are recorded: the order buffer, the count, and
// (unified worker only) the version. sortTime is deliberately excluded. Typed arrays are stored
// raw in little-endian byte order rather than as JSON number arrays to keep the fixtures small.
//
// File layout:
//   magic        4 bytes  'GOLD'
//   version      uint32   format version
//   numScenarios uint32
//   per scenario:
//     nameLength uint32, name bytes (utf8)
//     numRecords uint32
//     per record:
//       hasVersion uint8  (1: a uint32 version follows)
//       version?   uint32
//       count      uint32
//       orderLen   uint32  number of uint32 order entries
//       order      orderLen * uint32 little-endian

export const GOLDEN_MAGIC = 0x474F4C44; // 'GOLD'
export const GOLDEN_FORMAT_VERSION = 1;

// Accumulates the pieces of one golden file. Length prefixes that are only known once a section
// finishes are patched in place through the DataView kept for each.
export class GoldenWriter {
    constructor() {
        this._buffers = [];
        this._view = null;
        this._offset = 0;

        this._alloc(12);
        this._view.setUint32(0, GOLDEN_MAGIC, true);
        this._view.setUint32(4, GOLDEN_FORMAT_VERSION, true);
        this._scenarioCountOffset = 8;
        this._view.setUint32(this._scenarioCountOffset, 0, true);
        this._scenarioCount = 0;
    }

    _alloc(byteLength) {
        const buffer = new ArrayBuffer(byteLength);
        this._buffers.push(buffer);
        this._view = new DataView(buffer);
        this._viewOffset = this._offset;
        this._offset += byteLength;
    }

    _u8(value) {
        this._alloc(1);
        this._view.setUint8(0, value);
    }

    _u32(value) {
        this._alloc(4);
        this._view.setUint32(0, value >>> 0, true);
    }

    _string(value) {
        const bytes = new TextEncoder().encode(value);
        this._u32(bytes.length);
        this._buffers.push(bytes.buffer);
        this._offset += bytes.byteLength;
        this._view = null;
    }

    _rawArrayBuffer(arrayBuffer) {
        this._buffers.push(arrayBuffer);
        this._offset += arrayBuffer.byteLength;
        this._view = null;
    }

    beginScenario(name) {
        this._string(name);
        // record count prefix, patched in endScenario
        this._alloc(4);
        this._recordCountView = this._view;
        this._recordCount = 0;
        this._scenarioCount++;
    }

    writeRecord({ version, count, order }) {
        if (version === undefined) {
            this._u8(0);
        } else {
            this._u8(1);
            this._u32(version);
        }
        this._u32(count);
        this._u32(order.length);
        this._rawArrayBuffer(order.buffer.slice(order.byteOffset, order.byteOffset + order.byteLength));
        this._recordCount++;
    }

    endScenario() {
        this._recordCountView.setUint32(0, this._recordCount, true);
    }

    toBuffer() {
        // patch the scenario count in the header (the first buffer)
        new DataView(this._buffers[0]).setUint32(this._scenarioCountOffset, this._scenarioCount, true);
        return Buffer.concat(this._buffers.map(b => Buffer.from(b)));
    }
}

export const readGolden = (buffer) => {
    const bytes = buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer);
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    let offset = 0;

    const u8 = () => {
        const value = view.getUint8(offset);
        offset += 1;
        return value;
    };
    const u32 = () => {
        const value = view.getUint32(offset, true);
        offset += 4;
        return value;
    };
    const string = () => {
        const length = u32();
        const value = new TextDecoder().decode(new Uint8Array(view.buffer, view.byteOffset + offset, length));
        offset += length;
        return value;
    };

    if (u32() !== GOLDEN_MAGIC) {
        throw new Error('Invalid golden file: bad magic');
    }
    const formatVersion = u32();
    if (formatVersion !== GOLDEN_FORMAT_VERSION) {
        throw new Error(`Unsupported golden format version: ${formatVersion}`);
    }

    const numScenarios = u32();
    const scenarios = [];
    for (let s = 0; s < numScenarios; s++) {
        const name = string();
        const numRecords = u32();
        const records = [];
        for (let r = 0; r < numRecords; r++) {
            const hasVersion = u8() === 1;
            const version = hasVersion ? u32() : undefined;
            const count = u32();
            const orderLen = u32();
            const order = new Uint32Array(view.buffer.slice(view.byteOffset + offset, view.byteOffset + offset + orderLen * 4));
            offset += orderLen * 4;
            records.push({ version, count, order });
        }
        scenarios.push({ name, records });
    }

    return scenarios;
};
