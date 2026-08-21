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
| `/?model=sphere` | **FurryBall** — 79,801 strands × 16 particles = 1,276,816 particles, bouncing + rolling, root→tip color gradient (the demo video's tessellation — 80,601 strands — with each pole's 401 duplicate vertices welded to one, which fixes a fur crater at the poles; see IMPLEMENTATION.md) |
| `/?strands=N` | override sphere strand count (1,000–150,000) |
| `/?selftest` | headless correctness check: runs both solvers, reads particles back, asserts finiteness + segment lengths; result in title/console |

Both scenes: DFTL and PBD solvers (GUI-switchable), hair–hair friction/repulsion via a 64³ voxel
grid, ground-plane **and capsule body collision**, stencil floor reflection, voxel-grid,
bounding-box and collider debug views, and all the original's tuning parameters with its
`settings.xml` defaults.

## Controls

Drag to orbit, wheel to zoom, shift-drag (or right/middle drag) to pan, `f` for fullscreen.
**Hover the mouse over the model and the fur is brushed aside** and dragged along with the pointer;
holding a button orbits instead, and touch is orbit-only.

The panel adds, beyond the original's parameters:

| Control | Default | What it does |
|---|---|---|
| `render` | `ribbons` | `ribbons` draws each strand segment as a camera-facing quad of constant pixel width, tapering towards the tip. `lines` is the original 1px line-strip path — also the automatic fallback on devices that will not bind a storage buffer in the vertex stage. |
| `strand width` | 1.5 | Ribbon width in device pixels (0.5–4). |
| `shading` | on | Kajiya-Kay strand lighting plus volumetric self-shadowing from the voxel density grid, and form shading on the body. Off gives the original's flat colours and solid-black silhouette. |
| `draw colliders` | off | Wireframe of the body capsules (and the pointer brush) the simulation collides against. |

## Architecture (see PLAN.md / SPEC.md / IMPLEMENTATION.md)

- Raw WebGPU + WGSL, TypeScript, Vite. No 3D framework. lil-gui + wgpu-matrix only.
- **Everything stays on the GPU**: the particle buffer is `STORAGE | VERTEX` — the simulation
  writes it, the renderer draws line strips straight from it (native primitive-restart). Per-frame
  CPU→GPU traffic is a few hundred uniform bytes + one 4 KB joint palette.
- Compute passes per frame: skinning (baked joint palettes → roots + skinned normals), voxel fill
  (trilinear scatter via **fixed-point i32 atomics** — WGSL has no float atomics), voxel post
  (normalize + density gradient), optional 3× separable filter (ping-pong), then the strand solver
  (64-thread workgroups = whole strands in workgroup memory; barriers restructured into uniform
  control flow, which WGSL requires and which is stricter than the original's NVIDIA-tolerated
  races).
- **Fixed 1/120 s simulation step** on an accumulator (max 3 per frame). The voxel grid is rebuilt
  once per frame; only the solver repeats.
- The solver also writes each particle's **self-shadow transmittance** into the spare `prevPos.w`,
  three density taps towards the light — so the renderer gets deep-fur shading for free.
- Single render pass, 4× MSAA: background gradient → shaded near-black body → hair ribbons → floor
  stencil → mirrored fading reflection → debug overlays. Manual `pow(1/2.2)` gamma like the original.
- One command encoder / one submit per frame, all bind groups (including both ping-pong parities)
  prebuilt, `clearBuffer` for grid clears, GPU timestamp queries (when available) shown in the HUD.
- On the FurryBall the timestamps also drive an **adaptive downscale**: a frame over ~14 ms GPU
  rescales the strand count once and remembers it for the session.

## Model pipeline

`npm run bake` → `tools/bake_dae.py` (stdlib Python) parses the original Collada beast, bakes
final per-frame skinning palettes (world × inverseBind × bindShape, normalized over the union of
all 52 frames so the walk cycle rests exactly on the floor), and writes a compact 1.4 MB binary +
JSON manifest. Runtime model loading is ~50 lines; skinning runs in a compute shader.

## Measured (Apple Silicon, M-series, headless 1280×769)

| Scene | Particles | GPU sim | GPU render (ribbons) | GPU render (lines) |
|-------|-----------|---------|----------------------|--------------------|
| beast | 138,920 | ~0.6–0.9 ms | ~2–3 ms | ~1.2 ms |
| FurryBall | 1,289,616 | ~2 ms | ~6 ms | ~1.8 ms |

Sim time is per frame, i.e. one 1/120 s step at 120 fps and two at 60 fps.
