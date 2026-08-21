# Engine implementation spec (contract for implementation)

Reference GLSL sources: `/private/tmp/claude-502/-Users-patrickfurst-Projects-hairymess-web/0cd67925-87b1-4dbe-89d0-cb555c81f288/scratchpad/hairymess-src/bin/data/*.glsl`
Reference app logic: same dir `../src/ofApp.cpp`. Port behavior faithfully unless noted.

## Constants

```
NUM_HAIR_PARTICLES = 8        // particles per strand
WORK_GROUP_SIZE    = 64       // sim workgroup, 1D
STRANDS_PER_GROUP  = 8        // 64/8
VOXEL_GRID_SIZE    = 64       // 64^3 cells
VOXEL_LOCAL_SIZE   = 8        // 8x8x8 workgroups for grid passes
MIN_HAIR_LENGTH    = 1.0
MAX_HAIR_LENGTH    = 1.2
DENSITY_SCALE      = 4096.0   // fixed-point scale for density atomics
VELOCITY_SCALE     = 1024.0   // fixed-point scale for velocity atomics
MAX_DT             = 0.02     // clamp frame dt like original
```

Strand count = model vertex count (beast: 17365), **padded up** to a multiple of STRANDS_PER_GROUP
(17368) so every workgroup is full and barriers stay uniform (no early-outs in the sim kernel).
Padded strands: all 8 particles `fix=1`, parked at (−10000,−10000,−10000) — outside the voxel grid
(scatter skips them via bounds check), never indexed by the hair index buffer, inert.

## GPU buffers

`Particle` — storage, 48 B stride (drop the dead `vel` field from the original):
```wgsl
struct Particle {            // offset
  pos:     vec4f,            // 0   xyz position, w=1
  prevPos: vec4f,            // 16
  color:   vec4f,            // 32  rgb color, w = fix flag (0.0 / 1.0)
}
```
- `particles`: array<Particle> × paddedParticleCount. Usage STORAGE|VERTEX|COPY_DST|COPY_SRC.
  (COPY_SRC for the selftest readback.)
- `strandLengths`: array<f32> × paddedStrandCount. STORAGE|COPY_DST.
- `roots`: array<vec4f> × paddedStrandCount — skinning output, model-space (bind-shape applied),
  transformed by modelMatrix inside sim like the original. STORAGE|COPY_DST. Padded tail entries
  pre-written to parked position, skinning dispatch covers only real vertices.
- Skinning inputs: `bindPositions` array<vec4f> (model space bind pose), `skinJoints` array<vec4u>,
  `skinWeights` array<vec4f>, `palette` array<mat4x4f> × numJoints (UNIFORM or STORAGE, COPY_DST,
  rewritten per frame with the CPU-lerped palette).
- Skinned mesh output for rendering: `skinnedPositions` array<vec4f> × numVerts (STORAGE|VERTEX)
  written by the same skinning pass. Mesh index buffer: u32 triangles.
- Voxel atomics: `densityAtomic` array<atomic<i32>> × 64³ (4 B/cell); `velocityAtomic`
  array<atomic<i32>> × 64³×4 (x,y,z,unused — 16 B/cell keeps indexing simple). STORAGE|COPY_DST
  (clearBuffer needs COPY_DST? — no: clearBuffer requires no specific usage flag; keep COPY_DST off
  if unneeded).
- Voxel float ping-pong: `velocityGrid[2]` array<vec4f>×64³, `gradientGrid[2]` array<vec4f>×64³.
- Hair index buffer: u32, per real strand 8 indices + 0xFFFFFFFF restart → 17365×9 entries. INDEX.
- Uniform buffers below.

## Uniforms

One combined sim uniform struct (WGSL `uniform`), rewritten once per frame:
```wgsl
struct SimUniforms {
  modelMatrix:             mat4x4f,
  modelMatrixPrevInverted: mat4x4f,   // kept for parity; verletIntegration path unused
  modelTranslation:        vec4f,
  gravity:                 vec4f,
  minBB:                   vec4f,     // world, translated by modelTranslation on CPU? NO —
  maxBB:                   vec4f,     // static (−5,0,−5)..(5,10,5); grid space = pos − modelTranslation
  velocityDamping: f32,  numIterationsPBD: i32,  stiffness: f32,  friction: f32,
  repulsion: f32,        ftlDamping: f32,        deltaTime: f32,  gridSize: i32,
  numVerticesPerStrand: u32, numStrandsPerThreadGroup: u32, numStrands: u32, _pad: u32,
}
```
Filter pass axis: 3 tiny prebuilt uniform buffers (0/1/2) with their own bind groups — no per-frame
writes. Render uniforms per pass (viewProj, model, overrideColor, canvasHeight, mirror flag) —
small individual uniform buffers.

## Compute passes (order, one encoder)

1. `clearBuffer(densityAtomic)`, `clearBuffer(velocityAtomic)`.
2. **skinning.wgsl** — workgroup 64, dispatch ceil(numVerts/64), guard `if (i >= numVerts) return;`
   `p = Σ w_k * (palette[j_k] * bindPos)`; write `skinnedPositions[i]` and `roots[i] = p`.
3. **voxelFill.wgsl** — workgroup 64, dispatch ceil(paddedParticles/64).
   Per particle: gridPos = particle.pos − modelTranslation; velocity = (pos − prevPos)/dt.
   Trilinear scatter to 8 cells: `atomicAdd(density, i32(w * DENSITY_SCALE))`,
   `atomicAdd(velocity.xyz, i32(w * v * VELOCITY_SCALE))`.
   **Skip entirely if base cell index outside [0,gridSize)³** (handles parked padding; original
   only guarded upper corners — this is the corrected equivalent).
4. **voxelPost.wgsl** — workgroup 8×8×8, dispatch (8,8,8). Reads atomic buffers *bound as
   array<i32>* (separate pipeline → plain type view is legal). d = f32(di)/DENSITY_SCALE;
   v = f32(vi)/VELOCITY_SCALE; if d>0 v /= d; write `velocityGrid[W]`. Gradient of density via
   central differences (interior cells only, else 0), **not normalized**, write `gradientGrid[W]`.
   Then swap ping-pong index.
5. If `useFilter`: **voxelFilter.wgsl** ×3 (axis 0,1,2) — 3-tap box filter along axis, reads R
   writes W for both grids, swap after each. Out-of-range tap index falls back to center index
   (original behavior).
6. **simDFTL.wgsl** or **simPBD.wgsl** — workgroup 64, dispatch paddedParticles/64 exactly.

### Sim kernel structure (both variants)

Mirrors `hairSimulation.glsl` main + subroutine, with barriers legalized:

```
var<workgroup> sharedPos:   array<vec4f, 64>;
var<workgroup> sharedFix:   array<u32, 64>;
var<workgroup> sharedLen:   array<f32, 8>;    // per local strand

localVertexIndex   = local_invocation_id.x
localStrandIndex   = localVertexIndex / 8
globalStrandIndex  = workgroup_id.x * 8 + localStrandIndex
vertexIndexInStrand= localVertexIndex % 8

load: if vertexIndexInStrand==0 → pos = modelMatrix * roots[globalStrandIndex] (w=1), fix=1
      else → pos/prevPos/color/fix from particles[gid]
sharedLen[localStrandIndex] = strandLengths[globalStrandIndex]
workgroupBarrier();
```

**DFTL** (`DFTLApproach.glsl`):
- snapshot `posNextBefore = sharedPos[localVertexIndex+1]` (only if vertexIndexInStrand < 7) **before**
  any writes, then `workgroupBarrier()` — deterministic replacement for the original's racy read.
- if !fix: vel=(pos−prevPos)/dt; plane collision (pos,vel in/out, plane y=0 world);
  vel −= ftlDamping*(pos − posNextBefore)/dt (0 if last vertex);
  vel = friction/repulsion correction sampling grids at (pos − modelTranslation);
  sharedPos[i] = pos + vel*velocityDamping*dt + gravity*dt².
- `workgroupBarrier()`.
- if vertexIndexInStrand==0: serial loop j=0..6: sharedPos[base+j+1] = FTL length constraint
  toward sharedPos[base+j], rest = strandLen/8, stiffness factor (ignore fix like the original —
  its multiplier is computed but unused).
- `workgroupBarrier()`.
- write back: pos=sharedPos[i], prevPos=oldPos (position at kernel start, post-root-snap), color kept
  (roots keep color; the original overwrote root color to white — replicate: root writes color as
  loaded, which for roots is vec4(1) → visually white roots. Keep `fix` in color.w unchanged!
  Careful: color.w carries fix; "white" root color = rgb(1,1,1), w=1).

**PBD** (`PBDApproach.glsl`):
- if !fix: vel=(pos−prevPos)/dt; friction/repulsion at (pos − modelTranslation);
  sharedPos[i] = integrate as above.
- `workgroupBarrier()`.
- stiffness' = 1 − pow(1 − stiffness, 1/numIterations).
- loop numIterations (uniform value → uniform flow):
  - if localVertexIndex < 32 and (2i % 8) < 7: constrain pair (2i, 2i+1) both-ways with fix
    multipliers; `workgroupBarrier()`;
  - if localVertexIndex < 31 and ((2i+1) % 8) < 7: constrain pair (2i+1, 2i+2);
    `workgroupBarrier()`;
    (barriers must sit OUTSIDE the ifs — hoist: do the guarded work, then barrier, uniformly.)
- `workgroupBarrier()`; plane collision on sharedPos[i]; write back like DFTL.

Friction/repulsion (`hairSimulation.glsl` lines 94–107): trilinear-interpolate velocityGrid and
gradientGrid (guard cells outside [0,gridSize) → contribute 0);
`v = (1−friction)*v + friction*vGrid;  v += repulsion * gradient * dt`.

Plane collision (`computeHelper.glsl` calculatePlaneCollision): plane y=0 n=(0,1,0); if pos.y<0:
intersect segment prevPos→pos with plane, pos=hit point, v = w−u (reflect normal component:
u = dot(v,n)n; w = v−u; v = w−u).

## Render (single pass)

Targets: canvas `navigator.gpu.getPreferredCanvasFormat()`, MSAA 4 (resolve to canvas),
depth `depth24plus-stencil8` MSAA 4. Clear color = anything (background pass covers).
All fragment shaders apply gamma: `color.rgb = pow(color.rgb, vec3(1/2.2))` (match original).

Order within the pass:
1. **background**: fullscreen triangle, circular gradient — center ofLightGray(211/255) → edges
   white, depth write off, depth test off (`depthCompare:'always'`, depthWriteEnabled:false).
2. **mesh**: skinnedPositions as vertex buffer (vec4 stride 16) + triangle indices,
   `uniform modelMatrix, viewProj`, solid black, depth on.
3. **hair**: particles buffer as vertex buffer (stride 48: pos@0 float32x4, color@32 float32x4),
   topology `line-strip`, stripIndexFormat `uint32`, hair index buffer, drawIndexed. Alpha blend
   (src-alpha / one-minus-src-alpha), depth on. overrideColor=(1,1,1,1). Fragment discards nothing;
   color.a treated as 1 (color.w carries fix flag — force alpha 1.0 in shader).
4. **floor stencil**: 30×30 quad in XZ rotated 45° about Y, at y=0; colorWriteMask 0, depth write
   off, depth test off?? — original: depthMask false, no depth test change (test enabled, func LESS
   default → floor at y=0 may be occluded by mesh for stencil purposes; original had depth test on;
   replicate: depth test on, write off). Stencil: always pass, ref 1, replace on pass. Clear stencil
   at pass start (loadOp clear → already once per pass — fine, this is mid-pass; use stencilLoadOp
   'clear' at pass start; drawing order guarantees stencil is 0 until here).
5. **hair reflection**: same hair geometry, model pre-multiplied by scale(1,−1,1), stencil func
   equal ref 1 (keep ops), depth test on write on, floor fade fragment: WebGPU builtin position is
   top-left origin y-down → `y_gl = canvasHeight − pos.y`; alpha = clamp(0.01 + pow(y_gl/(canvasHeight/2), 4)
   … replicate original formula `delta = y_gl / (h − h/2)`, alpha = 0.01 + delta⁴, clamp 0..1),
   gray whiteSmoke tint, alpha blend.
6. **voxel debug** (toggle): point-list, 64³ vertices, vertex pulls velocityGrid[R]/gradientGrid[R]/
   density (bind atomic buffer as array<i32>) by vertex_index → cell center world pos + modelTranslation;
   color like original `voxelGrid_vs.glsl` (read it; approximately velocity magnitude / density
   coloring — faithful port of its color logic). Skip empty cells by emitting position w=0 →
   clip-space degenerate (or color a=0).
7. **bounding box debug** (toggle): 12 line-list edges of min/max BB (+modelTranslation), red.

MSAA: all pipelines multisample count 4. Canvas context alphaMode 'opaque'.

## Ping-pong bookkeeping

Physical buffers velocityGrid[0/1], gradientGrid[0/1]. Track `pingIndex` in TS. Post writes into
`W = 1 − ping`, then ping ^= 1. Each filter axis pass reads `ping` writes `1−ping`, then ping ^= 1.
Sim + debug read `ping`. Prebuild bind groups for both parities everywhere (post ×2, filter 3 axes ×2,
sim ×2 per algorithm, debug ×2). Assert no bind group creation inside the frame loop.

## Selftest (`?selftest`)

Run 120 frames with animation+filter on (DFTL 60, then switch PBD 60), then copy particle buffer to
a MAP_READ staging buffer and assert: (a) every component finite; (b) for ≥99% of real strands,
each segment length within 25% of rest length; (c) root positions of first 100 strands within 1e−3
of modelMatrix×roots readback... simplification: roots finite and inside |xyz| < 100. Log
`SELFTEST PASS`/`SELFTEST FAIL: reason` to console AND set `document.title`. Exposed as
`window.__selftestPromise` resolving to boolean.

## Timestamp queries

If `'timestamp-query'` in adapter.features: create querySet(count 16), timestampWrites on compute
passes (beginning/end of the whole compute stretch is enough: skinning-start … sim-end) and the
render pass; resolve into a small ring of MAP_READ buffers (3 in flight, skip when busy); expose
`stats.gpuSimMs`, `stats.gpuRenderMs` rolling averages for the HUD. Absent feature → stats stay null.

## Model manifest (`public/models/beast.json` + `beast.bin`)

```json
{ "version": 1, "vertexCount": 17365, "indexCount": N, "jointCount": 65,
  "frameCount": 52, "duration": 1.7, "modelMatrix": [16 floats, column-major],
  "buffers": { "positions": {"offset":0,"length":...}, "normals": {...}, "indices": {...},
               "joints": {...}, "weights": {...}, "palettes": {...} } }
```
`positions/normals` f32×3 tightly packed (model space, bind pose); `indices` u32; `joints` u16×4;
`weights` f32×4 normalized; `palettes` f32 column-major mat4 × jointCount × frameCount — FINAL
skinning matrices (world(t) × invBind × bindShape) in the DAE's own space; `modelMatrix` maps model
space → sim world (Z-up→Y-up, uniform scale, feet at y=0, centered, height ≈ 4.5). Loader lerps
adjacent frame palettes on CPU each frame (loop) and writes the palette buffer.
Animation clock scales with `playAnimation` toggle (pause holds phase).

Sphere fallback (`?model=sphere` or fetch failure): UV sphere radius 4 center (0,4,0), ~128×128
grid ≈ 16k verts, identity animation (single identity palette, jointCount 1, all weights joint 0),
same code path.

## Params defaults (from original settings.xml)

velocityDamping 0.9847, numIterations 30, stiffness 1.0, friction 0.0663, repulsion 117.35,
ftlDamping 1.0, gravity (0,−10,0), useFilter true, playAnimation true, drawFur true,
drawVoxelGrid false, drawBoundingBox false, algorithm 'DFTL'.
BB min (−5,0,−5) max (5,10,5). Camera: perspective 60°, near 0.1, far 10000, pos (10,15,10),
target (0,0,0).
