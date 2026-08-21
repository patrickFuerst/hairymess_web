# HairyMess Web

WebGPU port of [HairyMess](https://github.com/patrickFuerst/hairymess) — the 2015 OpenGL 4.4
compute-shader hair/fur simulation R&D project ([demo video](https://vimeo.com/131453436)).
The original ran only on Windows with NVIDIA GPUs; this port runs in the browser on any
WebGPU-capable device, with the entire simulation on the GPU.

## Run

```bash
npm install
npm run dev        # dev server
npm run build      # production build -> dist/ (static, deploy anywhere)
```

Requires a WebGPU browser: Chrome/Edge 113+, Safari 26+, Firefox 141+.

## Modes

| URL | Scene |
|-----|-------|
| `/` | **beast** — the walking beast, 17,365 strands × 8 particles (the checked-in 2015 configuration) |
| `/?model=sphere` | **FurryBall** — 80,601 strands × 16 particles = 1,289,616 particles, bouncing + rolling, root→tip color gradient (the configuration shown in the demo video) |
| `/?strands=N` | override sphere strand count (1,000–150,000) |
| `/?selftest` | headless correctness check: runs both solvers, reads particles back, asserts finiteness + segment lengths; result in title/console |

Both scenes: DFTL and PBD solvers (GUI-switchable), hair–hair friction/repulsion via a 64³ voxel
grid, ground-plane collision, stencil floor reflection, voxel-grid + bounding-box debug views, and
all the original's tuning parameters with its `settings.xml` defaults.

## Architecture (see PLAN.md / SPEC.md / IMPLEMENTATION.md)

- Raw WebGPU + WGSL, TypeScript, Vite. No 3D framework. lil-gui + wgpu-matrix only.
- **Everything stays on the GPU**: the particle buffer is `STORAGE | VERTEX` — the simulation
  writes it, the renderer draws line strips straight from it (native primitive-restart). Per-frame
  CPU→GPU traffic is a few hundred uniform bytes + one 4 KB joint palette.
- Compute passes per frame: skinning (baked joint palettes → roots), voxel fill (trilinear
  scatter via **fixed-point i32 atomics** — WGSL has no float atomics), voxel post
  (normalize + density gradient), optional 3× separable filter (ping-pong), then the strand solver
  (64-thread workgroups = whole strands in workgroup memory; barriers restructured into uniform
  control flow, which WGSL requires and which is stricter than the original's NVIDIA-tolerated
  races).
- Single render pass, 4× MSAA: background gradient → black skinned mesh → hair → floor stencil →
  mirrored fading reflection → debug overlays. Manual `pow(1/2.2)` gamma like the original.
- One command encoder / one submit per frame, all bind groups (including both ping-pong parities)
  prebuilt, `clearBuffer` for grid clears, GPU timestamp queries (when available) shown in the HUD.

## Model pipeline

`npm run bake` → `tools/bake_dae.py` (stdlib Python) parses the original Collada beast, bakes
final per-frame skinning palettes (world × inverseBind × bindShape, normalized over the union of
all 52 frames so the walk cycle rests exactly on the floor), and writes a compact 1.4 MB binary +
JSON manifest. Runtime model loading is ~50 lines; skinning runs in a compute shader.

## Measured (Apple Silicon, M-series)

| Scene | Particles | GPU sim | GPU render |
|-------|-----------|---------|------------|
| beast | 138,920 | ~0.7 ms | ~1.0 ms |
| FurryBall | 1,289,616 | ~6 ms | ~10 ms |
