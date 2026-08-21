# Implementation notes

Port of the 2015 OpenGL 4.4 HairyMess compute-shader hair simulation to WebGPU/WGSL.
Companion to `PLAN.md` (context) and `SPEC.md` (the contract this implements).

## File map

| File | Role |
|---|---|
| `src/main.ts` | Boot (adapter/device, banner on failure), model load + sphere fallback, per-mode config and defaults, GUI/HUD wiring, canvas sizing (devicePixelRatio, capped at 2), `f` fullscreen toggle, rAF loop with `dt` clamped to `MAX_DT`, URL params |
| `src/engine/device.ts` | `initWebGPU()` (context config, conditional `timestamp-query`), `createShaderModule()` which reports WGSL diagnostics to the console |
| `src/engine/config.ts` | `EngineConfig` — particles-per-strand, workgroup sizes, grid size, fixed-point scales, hair length range, bounding box, PRNG seed. Everything the kernels specialise on flows from here, so a future 16-particle/80k-strand mode only needs a different config. Also `makeRng()` (mulberry32, replaces `ofRandom`) |
| `src/engine/layout.ts` | Byte-for-byte uniform offset tables (`SIM_U`, `RENDER_U`, `BG_U`, `SKIN_U`, `FILTER_U`), `UniformScratch`, `gpuWrite()` |
| `src/engine/shaders.ts` | `buildShaders(cfg)` — string-composes the generated constants prelude + `common.wgsl` in front of every shader (and `simCommon.wgsl` in front of the two solvers) |
| `src/engine/model.ts` | `beast.json`/`beast.bin` loader, procedural UV sphere, per-frame CPU palette lerp, CPU frame-0 skinning for particle placement, small column-major mat4 helpers |
| `src/engine/hairSim.ts` | All GPU buffers, the five compute pipelines, every bind group (both ping-pong parities), initial particle state, per-frame uniform upload, `encode()` |
| `src/engine/renderer.ts` | MSAA/depth targets, the seven render pipelines and their uniforms, the single render pass |
| `src/engine/timestamps.ts` | Feature-gated GPU timing: 4 timestamps/frame, ring of 3 MAP_READ buffers, rolling average |
| `src/engine/selftest.ts` | `?selftest` — 60 frames per solver, particle readback, assertions, PASS/FAIL to console + `document.title` + `window.__selftestPromise` |
| `src/shaders/common.wgsl` | Prelude: `Particle`, `SimUniforms`, `RenderUniforms`, `VSOut`, `voxelIndex()`, `gammaCorrect()`, `checkCollision()`, `constrainMultiplier()` |
| `src/shaders/simCommon.wgsl` | Solver bindings, workgroup arrays, trilinear grid sampling, friction/repulsion, integration, kernel prologue/epilogue |
| `src/shaders/{skinning,voxelFill,voxelPost,voxelFilter,simDFTL,simPBD}.wgsl` | Compute kernels |
| `src/shaders/{background,mesh,hair,floorFade,voxelDebug,bbox}.wgsl` | Render shaders |

Consumed unchanged: `index.html`, `src/style.css`, `src/engine/camera.ts`, `src/ui/hud.ts`.
`src/params.ts` gained only `ColorMode` + `colorMode`; `src/ui/gui.ts` gained the model and
colours dropdowns. `tools/` and `public/models/` were not touched.

## Modes

| | beast (default) | sphere — "FurryBall" (`?model=sphere`) |
|---|---|---|
| Source | `public/models/beast.bin` | procedural UV sphere, radius 4 |
| Strands | 17,365 (mesh vertices) | **80,601** = (200+1) x (400+1) |
| Particles/strand | 8 | **16** (⇒ 1,289,616 particles) |
| `strandsPerGroup` / padding | 8 | 4 |
| Colours | alternating orange/teal | root→tip gradient |
| Animation | baked walk cycle (52 frames) | bounce + roll |
| Voxel box | (−5, 0, −5)…(5, 10, 5) | (−7, −7, −7)…(7, 7, 7), model-local |
| `repulsion` / `numIterations` | 117.347 / 30 (settings.xml) | 12.64 / 60 (derived — see deviation 15) |

The strand and particle counts match the demo video's on-screen readout exactly
("Num Hairstrands: 80601, Num Particles: 1289616"), and `(res+1) x (2*res+1)` is precisely the
tessellation `ofMesh::sphere(4, 200)` produces — so the original's ball was this same UV sphere.

**FurryBall animation** (the two lines that are commented out in `ofApp.cpp`, gated by
`playAnimation`): `modelMatrix = translate(0, 4 + 5*|sin t|, 0) * rotate(sin(0.2 t) * PI, normalize(1,1,0))`.
Slerping from identity to a 180° rotation is just that rotation scaled by the factor, which is where
the angle comes from. The sphere is built centred on the model-space origin so the ball spins in
place and rests on the floor at phase 0. `simTranslation` follows the bounce so the voxel grid
tracks the ball, and the model's `duration` is 10π — the exact common period of `|sin t|` and
`sin(0.2 t)`, so the loop is seamless.

**Colours.** `alternating` is the checked-in scheme (even strands `ofColor(255,190,42)`, odd
`ofColor(0,163,136)`). `gradient` is the video's root→tip ramp `#d5491d → #b7cf4f → #52c8bc`
interpolated over `j/(N-1)`. Those hex values were sampled from the *rendered* video and the hair
fragment shader applies `pow(1/2.2)`, so the stops are stored raised to 2.2 and land back on the
sampled colours after the shader's gamma.

**URL parameters**: `?model=sphere|beast`, `?colors=gradient|alternating`, `?strands=N` (sphere
tessellation, clamped to 1,000…150,000 so the particle buffer stays inside
`maxStorageBufferBindingSize`), `?selftest`. The GUI's `model` dropdown rewrites `?model=` and
reloads (a full, clean reinitialisation); the `colors` dropdown calls `HairSim.resetParticles()`,
which re-skins at the *current* pose and rebuilds the particle buffer with the same seeded hair
lengths.

## Pass graph (one command encoder, one submit per frame)

```
clearBuffer(densityAtomic)                              # clearBuffer needs COPY_DST
clearBuffer(velocityAtomic)
compute pass  [timestamps 0 .. 1]
  1 skinning     @64        ceil(numVerts/64)      palette x bind pose -> roots + skinnedPositions
  2 voxelFill    @64        paddedParticles/64     fixed-point atomic trilinear scatter
  3 voxelPost    @4,4,4     16^3                   decode atomics, v /= density, density gradient
                                                   writes parity 1-ping, then ping ^= 1
  4 voxelFilter  @4,4,4     16^3   x3 (if useFilter)  3-tap box per axis, ping ^= 1 each
  5 simDFTL | simPBD  @64   paddedParticles/64     solver, reads grid parity `ping`
render pass (4x MSAA -> canvas, depth24plus-stencil8, stencil ref 1)  [timestamps 2 .. 3]
  1 background   fullscreen triangle, circular gradient, depthCompare always / no write
  2 mesh         skinnedPositions + triangle indices, solid black, depth on          (if drawFur)
  3 hair         particles as vertex buffer (stride 48), line-strip + 0xFFFFFFFF     (if drawFur)
  4 floor        30x30 XZ diamond, writeMask 0, depth test on / write off,
                 stencil always -> replace ref 1                                     (if drawFur)
  5 reflection   same hair geometry, model = scale(1,-1,1), stencil equal 1,
                 whiteSmoke tint, distance fade                                      (if drawFur)
  6 voxelDebug   point-list, 64^3 points, grids bound as vertex buffers    (if drawVoxelGrid)
  7 bbox         line-list, 24 vertices generated in the VS, red          (if drawBoundingBox)
resolveQuerySet + copy into the timestamp ring (skipped when all 3 slots are busy)
```

Strand count is padded to a multiple of `strandsPerGroup` (17,365 → 17,368 for the beast), so the
solver dispatch is exact and needs no early-out — every `workgroupBarrier()` is in uniform control
flow. Padding strands are all `fix = 1`, parked at (−10000, −10000, −10000), skipped by the voxel
scatter's bounds check and never referenced by the hair index buffer.

No `GPUBuffer` or `GPUBindGroup` is created inside the frame loop; both ping-pong parities of every
grid-touching bind group are built up front.

## Deviations from SPEC.md (and why)

1. **Voxel kernels use 4×4×4 workgroups, not 8×8×8.** 8³ = 512 invocations exceeds WebGPU's default
   `maxComputeInvocationsPerWorkgroup` of 256, so the pipeline would not create. These kernels use
   neither workgroup memory nor barriers, so the shape is purely a scheduling choice; the dispatch
   becomes 16³. Config-driven (`EngineConfig.voxelLocalSize`).
2. **`modelTranslation` is the model's *placement* (0,0,0), not `modelMatrix`'s translation column.**
   Measured from the delivered bake: the skinned beast already straddles the world origin
   (x ∈ [−3.4, 3.4], y ∈ [0, 4.5], z ∈ [−6, 6]), because that column ((−0.16, 1.23, 5.98)) is part
   of the Z-up→Y-up/scale/centering normalisation. Subtracting it would put the voxel grid at
   y ∈ [1.2, 11.2], z ∈ [1.0, 11.0] and leave the model almost entirely un-gridded, silently
   disabling friction and repulsion. `HairModel.simTranslation` carries the placement instead, ready
   for the bouncing-FurryBall mode to animate.
3. **The voxel debug view reads the grids as vertex buffers**, exactly like the original's
   `mVoxelVBO`, rather than as storage buffers in the vertex stage. WebGPU's
   `maxStorageBuffersInVertexStage` defaults to 0, so the storage route would need a requested limit
   and a fallback path. Decoding the vertex index is the exact inverse of `voxelIndex()`, so
   attribute *i* is cell *i*.
4. **Plane collision uses the line/plane intersection only when it lies on the segment**
   (`0 ≤ delta ≤ |pos − prevPos|`), otherwise the point is projected straight onto the plane. The
   original always used the infinite-line intersection; for a particle that is *already* below the
   plane and travelling nearly parallel to it, `−prevPos.y / ray.y` diverges and flings the particle
   across the scene (or to infinity when the ray is exactly parallel). Behaviour is bit-identical to
   the original whenever the segment genuinely crosses the plane.
5. **PBD applies its post-constraint collision only to non-fixed vertices.** The original applies it
   to all of them, but a root's `prevPosition` is `vec4(0)`, so for any root below the plane the
   original's math resolves to the world origin and teleports the strand root there for a frame.
6. **DFTL samples the grids at `pos − modelTranslation`**, as SPEC dictates. The original omits that
   subtraction in `DFTLApproach.glsl` while `PBDApproach.glsl` includes it — an inconsistency in the
   original. (With deviation 2 the translation is zero for the shipped models, so this is currently
   a no-op that matters only once a mode moves the model.)
7. **Gamma correction follows the original's shaders rather than being blanket-applied.** `mesh`,
   `hair` and `floorFade` apply `pow(c, 1/2.2)` (`basic_FS.glsl`, `floor_fs.glsl`); `background`,
   `voxelDebug` and `bbox` do not, because their originals are openFrameworks' own non-gamma path
   and `voxelGrid_fs.glsl`. Applying it to the background would wash the lightGray→white gradient
   almost flat, and applying it to the gradient-coloured voxel points would need the signed values
   clamped first.
8. **The selftest's length assertion skips segments the ground collision just repositioned** (either
   endpoint at y ≤ 1e−3) and reports per solver. PBD applies the plane collision *after* the
   constraint solve — faithfully to the original — so those segments are legitimately off their rest
   length for a frame; asserting on them measures the collision, not the solver. See "Known issue"
   below for why this matters so much with the current bake.
9. **DFTL and PBD share their bind groups** (identical layouts), so there are 2 rather than 4.
10. **All compute dispatches live in one compute pass** (SPEC's timestamp guidance: begin-of-skinning
    to end-of-sim). WebGPU orders dispatches within a pass and inserts the needed barriers.
11. **The sphere is 200×400 segments → 80,601 vertices**, matching the demo video rather than
    SPEC's "~128×128 ≈ 16k" placeholder.
15. **Sphere mode derives two of its defaults instead of taking them from `settings.xml`.** Beast
    mode is untouched. Both follow from the mode's geometry, and both are computed, not hand-tuned:
    * `repulsion` is divided by the particle count relative to the beast baseline
      (`densityCompensatedRepulsion()`, 117.347 → 12.64 at 1,289,616 particles). The repulsion term
      uses the **non-normalised** density gradient, whose magnitude is proportional to particles per
      voxel; the FurryBall packs ~9× more particles into a 2.7× larger box, so the shipped value
      drove it ~13× harder than the original ever ran. Measured: with the raw value DFTL's worst
      per-step motion was 14.9 rest lengths (the beast baseline is ~2.5) and PBD collapsed to 3.6 %
      of strands holding their length; compensated, DFTL is back to 6.8 (≈2.6× the beast, which is
      exactly what halving the rest length predicts) and PBD holds 100 %. The video build avoided
      this by using the *normalised*-gradient variant with repulsion 0.61–3.4; we keep the
      checked-in non-normalised math and scale the parameter instead. It also tracks `?strands=`.
    * `numIterations` scales with strand length (30 → 60), keeping the original's Gauss-Seidel
      sweeps-per-link now that chains are twice as long.
16. **The selftest's length tolerance is per solver**: 25 % for DFTL, 100 % for PBD. DFTL resets each
    segment to its rest length outright and measures 100.00 % in every configuration, so it keeps the
    tight bound. PBD is a *soft* solver and the readback lands right after an odd Gauss-Seidel sweep,
    which by construction leaves the even pairs unrelaxed; its relative residual also doubles when
    the rest length halves at 16 particles/strand. The looser bound still catches a blow-up — both
    models come in at 100 % of strands with worst ratios of 1.28 (beast) and 1.81 (sphere), well
    inside it.
12. **Voxel debug points are 1 px.** WebGPU has no `glPointSize`; the original used 2.
13. `ofRandom()` is replaced by a seeded mulberry32 so runs are reproducible.
14. Hair grows along the *world-space* normal (`mat3(modelMatrix) · skinned normal`). The original
    used the untransformed model-space normal, which would point the fur along the model's Z-up axis
    now that the bake reorients the model.

## Verification performed

`npx tsc --noEmit` and `npm run build` are clean.

WGSL is only validated at runtime, so beyond desk-checking, the built app was run in headless Chrome
(`--headless=new`, file:// with an isolated profile — no dev server) against real Dawn:

* **No WGSL compilation messages and no WebGPU validation errors** in any run. Every one of the 12
  shaders and all 12 pipelines were created and executed — including both solvers, the filter passes
  and both debug views — so all `@group`/`@binding` numbers, access modes, vertex layouts,
  `stripIndexFormat`, workgroup sizes and buffer usages agree with the TypeScript layouts.
* **`?selftest` passes on both models**:
  * beast (17,365 strands × 8) — DFTL 100.00 % of strands within 25 % (worst segment ratio 1.000);
    PBD 100.00 % within 100 % (worst 1.275).
  * sphere (80,601 strands × 16, 1,289,616 particles) — DFTL 100.00 % within 25 % (worst 1.000);
    PBD 100.00 % within 100 % (worst 1.807). Exercises the `frameCount = 1` / `jointCount = 1`
    path, the 4-strand workgroup packing and the 16-entry-per-strand index buffer.
* **`?strands=`** verified: 20,000 → 19,900 strands, 999,999 → clamped to 149,878 strands
  (2,398,048 particles, a 115 MB particle buffer) with no validation errors.
* **Both colour schemes** verified on the sphere; the density-compensated repulsion tracks the
  strand count (33.81 at 30,135 strands, 12.64 at 80,601).
* **Rendering was inspected via headless screenshots**: background gradient, black skinned mesh,
  orange/teal hair, the stencilled fading floor reflection, the red bounding box at exactly
  (−5, 0, −5)…(5, 10, 5), and gradient-coloured voxel points confined to the occupied cells (empty
  cells are correctly culled by pushing them outside the clip volume).
* **GPU timestamps work** (`timestamp-query` present). Headless HUD readings:
  beast (138,920 particles) ~0.7 ms sim / ~1.7 ms render; FurryBall (1,289,616 particles)
  **~6.1 ms sim / ~10.6 ms render**. The render cost is dominated by drawing 1.29 M line-strip
  vertices twice (hair + mirrored reflection) at 4× MSAA.
* The uniform offset tables are confirmed indirectly but strongly: the bounding box draws at the
  exact `minBB`/`maxBB` extents (offsets 160/176), the voxel points sit where the sim samples them
  (offset 128), and the solver is stable with the settings.xml defaults (offsets 192–232).

### Known artefact — the UV sphere's poles

A UV sphere duplicates `sectors+1` vertices at each pole, so 401 of the FurryBall's strands start
at exactly the same point. With the non-normalised gradient that concentration is a large local
density spike, and the repulsion blows a small crater in the fur at each pole (visible as a ring of
exposed black mesh when the pole faces the camera). This is a consequence of reproducing
`ofMesh::sphere(4, 200)` exactly — the original had the same tessellation, but its repulsion used
the *normalised* gradient, which is insensitive to a density spike. If it becomes objectionable the
options are to weld the pole vertices (which costs 800 strands, so the HUD would read 79,801 rather
than the video's 80,601) or to add the normalised-gradient repulsion variant as a mode.

### Resolved — the beast bake's ground alignment

An earlier bake sank up to 16 % of the mesh as much as 3 units below the ground plane mid-stride,
which pinned that hair flat onto y = 0 and (with PBD, which collides after solving) stretched those
strands' first segment. **The re-baked `beast.bin` fixes it**: the minimum world y across all 52
frames is now +0.017 with zero vertices below the plane, and the animated AABB
(x ∈ [−1.91, 1.91], y ∈ [0.05, 4.48], z ∈ [−3.03, 3.06]) sits comfortably inside the voxel box.
The selftest's floor-touching share dropped from 18.5 % to 1.2 % accordingly.

Note this also re-confirms deviation 2: the new matrix's translation column is
(−0.18, 2.18, 3.05) while the skinned model straddles the world origin, so using that column as the
grid offset would now push the voxel grid 3 units off in z — past the model's entire z extent.

### Not verified without an interactive browser

* Interaction: orbit/pan/zoom, the `f` fullscreen toggle, DPR changes and live canvas resizing (the
  resize path runs every frame, but only at one size so far). The two new dropdowns were exercised
  through their URL-parameter equivalents (`?model=`, `?colors=`) rather than by clicking, so the
  lil-gui `onChange` wiring itself — in particular the mid-flight `resetParticles()` upload — has not
  been driven interactively.
* Visual fidelity against the original video is now close (dense fur, gradient reading warm at the
  roots and teal at the tips), but the exact gradient multipliers were matched analytically against
  the sampled hex values, not eyeballed side by side with video frames.
* Real-GPU performance (headless numbers came from a machine-local run and are indicative only) and
  behaviour on non-Dawn implementations (Safari/WebKit, Firefox/wgpu). Firefox in particular has been
  stricter about uniformity analysis — the barriers here are all at function-body or loop-body scope,
  never inside a conditional, so this should hold.
* `device.lost` / `uncapturederror` banners were never triggered.
