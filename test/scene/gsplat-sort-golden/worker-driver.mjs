// Shared driver that runs either sort worker against a list of message scenarios using a fake
// message port, in plain Node. Used by the golden generator and the replay test so the two cannot
// drift apart.

// Fake message port: records the messages a worker posts and lets the driver deliver messages.
class FakePort {
    constructor() {
        this.listener = null;
        this.posted = [];
    }

    addEventListener(type, fn) {
        this.listener = fn;
    }

    postMessage(message) {
        this.posted.push(message);
    }

    send(message) {
        this.listener({ data: message });
    }
}

// GSplatSorter's main thread hands the previous order buffer back to the worker after each result
// (the worker sets its order to null after posting). The replay emulates that handoff.
const returnOrderBuffers = (port) => {
    for (const message of port.posted) {
        if (message.order instanceof ArrayBuffer) {
            port.send({ order: new ArrayBuffer(message.order.byteLength) });
        }
    }
};

// Deep-clones a scenario message so the worker can detach/consume its buffers without mutating the
// shared scenario definition.
const cloneMessage = (message) => {
    const payload = {};
    for (const key of Object.keys(message)) {
        const value = message[key];
        payload[key] = value instanceof ArrayBuffer ? value.slice(0) : structuredClone(value);
    }
    return payload;
};

// Runs one worker factory over one scenario and returns the worker's posted messages.
export const runScenario = (workerFactory, scenario, { returnOrders = false } = {}) => {
    const port = new FakePort();
    workerFactory(port);

    for (const message of scenario.messages) {
        port.send(cloneMessage(message));
        if (returnOrders) {
            returnOrderBuffers(port);
        }
    }

    return port.posted;
};

// Normalizes a posted worker message into the fields the golden files record.
export const normalizeRecord = (message) => {
    const record = {
        count: message.count,
        order: new Uint32Array(message.order)
    };
    if (message.version !== undefined) {
        record.version = message.version;
    }
    return record;
};
