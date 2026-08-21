// Optional GPU timing via 'timestamp-query'. Four timestamps per frame:
//   0 compute begin, 1 compute end, 2 render begin, 3 render end
// resolved into a small ring of MAP_READ buffers; frames are skipped whenever every
// ring slot is still mapped.

const QUERY_COUNT = 4;
const RING_SIZE = 3;
const BYTES = QUERY_COUNT * 8; // u64 per query

export interface GpuTimings {
  simMs: number | null;
  renderMs: number | null;
}

export class Timestamps {
  private readonly device: GPUDevice;
  private readonly querySet: GPUQuerySet;
  private readonly resolveBuffer: GPUBuffer;
  private readonly ring: GPUBuffer[];
  private readonly busy: boolean[];
  private simMs: number | null = null;
  private renderMs: number | null = null;

  constructor(device: GPUDevice) {
    this.device = device;
    this.querySet = device.createQuerySet({ label: 'gpuTimings', type: 'timestamp', count: QUERY_COUNT });
    this.resolveBuffer = device.createBuffer({
      label: 'timestampResolve',
      size: BYTES,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });
    this.ring = [];
    this.busy = [];
    for (let i = 0; i < RING_SIZE; i++) {
      this.ring.push(
        device.createBuffer({
          label: `timestampRead${i}`,
          size: BYTES,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        })
      );
      this.busy.push(false);
    }
  }

  get computePassWrites(): GPUComputePassTimestampWrites {
    return { querySet: this.querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 };
  }

  get renderPassWrites(): GPURenderPassTimestampWrites {
    return { querySet: this.querySet, beginningOfPassWriteIndex: 2, endOfPassWriteIndex: 3 };
  }

  get timings(): GpuTimings {
    return { simMs: this.simMs, renderMs: this.renderMs };
  }

  /** Call after the passes are encoded but before submit. Returns the slot to read. */
  resolve(encoder: GPUCommandEncoder): number {
    const slot = this.busy.indexOf(false);
    if (slot < 0) return -1; // every slot still mapped: skip this frame
    encoder.resolveQuerySet(this.querySet, 0, QUERY_COUNT, this.resolveBuffer, 0);
    encoder.copyBufferToBuffer(this.resolveBuffer, 0, this.ring[slot], 0, BYTES);
    this.busy[slot] = true;
    return slot;
  }

  /** Call after submit with the slot returned by resolve(). */
  read(slot: number): void {
    if (slot < 0) return;
    const buffer = this.ring[slot];
    void buffer
      .mapAsync(GPUMapMode.READ)
      .then(() => {
        const stamps = new BigUint64Array(buffer.getMappedRange().slice(0));
        buffer.unmap();
        this.busy[slot] = false;
        const sim = Number(stamps[1] - stamps[0]) / 1e6;
        const render = Number(stamps[3] - stamps[2]) / 1e6;
        if (Number.isFinite(sim) && sim >= 0) {
          this.simMs = this.simMs === null ? sim : this.simMs * 0.9 + sim * 0.1;
        }
        if (Number.isFinite(render) && render >= 0) {
          this.renderMs = this.renderMs === null ? render : this.renderMs * 0.9 + render * 0.1;
        }
      })
      .catch(() => {
        // device lost or buffer destroyed — stop using this slot
        this.busy[slot] = true;
      });
  }
}
