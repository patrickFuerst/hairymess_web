# Implementation notes

Port of the 2015 OpenGL 4.4 HairyMess compute-shader hair simulation to WebGPU/WGSL.
Companion to `PLAN.md` (context) and `SPEC.md` (the contract this implements).

## File map

| File | Role |
|---|---|
| `src/main.ts` | Boot (adapter/device, banner on failure), model load + sphere fallback, per-mode config and defaults, GUI/HUD wiring, canvas sizing (devicePixelRatio, capped at 2), `f` fullscreen toggle, rAF loop with `dt` clamped to `MAX_DT`, URL params, adaptive quality, device-lost recovery |
| `src/engine/device.ts` | `initWebGPU()` (context config, conditional `timestamp-query`, requested `maxStorageBuffersInVertexStage`), `createShaderModule()` which reports WGSL diagnostics to the console |
| `src/engine/config.ts` | `EngineConfig` — particles-per-strand, workgroup sizes, grid size, fixed-point scales, hair length range, bounding box, PRNG seed. Everything the kernels specialise on flows from here, so a future 16-particle/80k-strand mode only needs a different config. Also `makeRng()` (mulberry32, replaces `ofRandom`) |
| `src/engine/layout.ts` | Byte-for-byte uniform offset tables (`SIM_U`, `RENDER_U`, `BG_U`, `SKIN_U`, `FILTER_U`), `UniformScratch`, `gpuWrite()` |
| `src/engine/shaders.ts` | `buildShaders(cfg)` — string-composes the generated constants prelude + `common.wgsl` in front of every shader (and `simCommon.wgsl` in front of the two solvers) |
| `src/engine/model.ts` | `beast.json`/`beast.bin` loader (including the optional SPEC v2 `colliders` section), procedural UV sphere, per-frame CPU palette lerp, CPU frame-0 skinning for particle placement, `skinColliders()`, small column-major mat4 helpers |
| `src/engine/hairSim.ts` | All GPU buffers, the five compute pipelines, every bind group (both ping-pong parities), initial particle state, the fixed-step accumulator, per-frame uniform + collider upload, `encode()` |
| `src/engine/renderer.ts` | MSAA/depth targets, the ten render pipelines and their uniforms, the single render pass |
| `src/engine/interaction.ts` | `PointerBrush` — hover ray vs. per-model proxy capsule, brush collider placement and velocity |
| `src/engine/timestamps.ts` | Feature-gated GPU timing: 4 timestamps/frame, ring of 3 MAP_READ buffers, rolling average |
| `src/engine/selftest.ts` | `?selftest` — 60 frames per solver, particle readback, assertions, PASS/FAIL to console + `document.title` + `window.__selftestPromise` |
| `src/shaders/common.wgsl` | Prelude: `Particle`, `SimUniforms`, `RenderUniforms`, `VSOut`, `LIGHT_DIR`, `voxelIndex()`, `gammaCorrect()`, `reflectionFade()`, `kajiyaKay()`, `checkCollision()`, `resolveCapsule()`, `constrainMultiplier()` |
| `src/shaders/simCommon.wgsl` | Solver bindings, workgroup arrays, trilinear grid sampling, friction/repulsion, integration, `resolveColliders()`, `selfShadow()`, kernel prologue/epilogue |
| `src/shaders/{skinning,voxelFill,voxelPost,voxelFilter,simDFTL,simPBD}.wgsl` | Compute kernels |
| `src/shaders/{background,mesh,hair,ribbon,voxelDebug,bbox,colliders}.wgsl` | Render shaders |

Consumed unchanged: `index.html`, `src/style.css`, `src/engine/camera.ts`, `src/ui/hud.ts`.
`tools/` and `public/models/` were not touched. `floorFade.wgsl` is gone: the mirrored reflection is
the same pipeline as the upright hair with `RenderUniforms.fade = 1`, so both hair paths serve both
passes from one shader each.

## Modes

| | beast (default) | sphere — "FurryBall" (`?model=sphere`) |
|---|---|---|
| Source | `public/models/beast.bin` | procedural UV sphere, radius 4 |
| Strands | 17,365 (mesh vertices) | **79,801** = (200+1) x (400+1) − 800 (poles welded) |
| Particles/strand | 8 | **16** (⇒ 1,276,816 particles) |
| `strandsPerGroup` / padding | 8 | 4 |
| Colours | alternating orange/teal | root→tip gradient |
| Animation | baked walk cycle (52 frames) | bounce + roll |
| Voxel box | (−5, 0, −5)…(5, 10, 5) | (−7, −7, −7)…(7, 7, 7), model-local |
| `repulsion` / `numIterations` | 117.347 / 30 (settings.xml) | 12.64 / 60 (derived — see deviation 15) |

`(res+1) x (2*res+1)` is precisely the tessellation `ofMesh::sphere(4, 200)` produces, and matches
the demo video's on-screen readout exactly ("Num Hairstrands: 80601, Num Particles: 1289616") — so
the original's ball was this same UV sphere. This build welds each pole's duplicated vertices down
to one (see "Resolved — the UV sphere's poles" below), which is why the actual strand/particle
counts are 79,801 / 1,276,816 rather than the video's numbers.

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
`maxStorageBufferBindingSize`; also switches off adaptive quality), `?selftest`. The GUI's `model`
dropdown rewrites `?model=` and reloads (a full, clean reinitialisation); the `colors` dropdown calls
`HairSim.resetParticles()`, which re-skins at the *current* pose and rebuilds the particle buffer
with the same seeded hair lengths.

## Pass graph (one command encoder, one submit per frame)

```
clearBuffer(densityAtomic)                              # clearBuffer needs COPY_DST
clearBuffer(velocityAtomic)
compute pass  [timestamps 0 .. 1]
  1 skinning     @64        ceil(numVerts/64)      palette x bind pose -> roots.xyz (w kept)
                                                   + skinnedVerts (position + normal)
  2 voxelFill    @64        paddedParticles/64     fixed-point atomic trilinear scatter
  3 voxelPost    @4,4,4     16^3                   decode atomics, v /= density, density gradient
                                                   writes parity 1-ping, then ping ^= 1
  4 voxelFilter  @4,4,4     16^3   x3 (if useFilter)  3-tap box per axis, ping ^= 1 each
  5 simDFTL | simPBD  @64   paddedParticles/64     solver, reads grid parity `ping`
                                                   dispatched ONCE PER SUBSTEP (0..3)
render pass (4x MSAA -> canvas, depth24plus-stencil8, stencil ref 1)  [timestamps 2 .. 3]
  1 background   fullscreen triangle, circular gradient, depthCompare always / no write
  2 mesh         skinnedVerts (stride 32) + triangle indices, hemispheric + N.L + rim (if drawFur)
  3 hair         ribbons: 6 verts/segment, particles pulled from a vertex-stage storage
                 buffer, screen-space quad expansion + Kajiya-Kay                    (if drawFur)
                 lines:   particles as vertex buffer (stride 48), line-strip + 0xFFFFFFFF
  4 floor        60x60 XZ diamond, writeMask 0, depth test on / write off,
                 stencil always -> replace ref 1                                     (if drawFur)
  5 reflection   same hair pipeline with fade = 1, model = scale(1,-1,1), stencil equal 1,
                 whiteSmoke tint, distance fade                                      (if drawFur)
  6 voxelDebug   point-list, 64^3 points, grids bound as vertex buffers    (if drawVoxelGrid)
  7 bbox         line-list, 24 vertices generated in the VS, red          (if drawBoundingBox)
  8 colliders    line-list, 152 verts per capsule generated in the VS, depthCompare always
                                                                              (if drawColliders)
resolveQuerySet + copy into the timestamp ring (skipped when all 3 slots are busy)
```

## Bindings

Compute — every group is `@group(0)` unless noted:

| Pipeline | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| `skinning` | u `SkinUniforms` | r `bindPositions` | r `skinJoints` | r `skinWeights` | r `palette` | rw `roots` | rw `skinnedVerts` | r `bindNormals` |
| `voxelFill` | u `SimUniforms` | r `particles` | rw `densityAtomic` | rw `velocityAtomic` | | | | |
| `voxelPost` | u `SimUniforms` | r `density` (i32) | r `velocity` (i32) | rw `velocityGrid[W]` | rw `gradientGrid[W]` | | | |
| `voxelFilter` | r `velocityGrid[R]` | r `gradientGrid[R]` | rw `velocityGrid[W]` | rw `gradientGrid[W]` | | | | |
| `simDFTL` / `simPBD` | u `SimUniforms` | rw `particles` | r `roots` (w = rest length) | r `velocityGrid[R]` | r `gradientGrid[R]` | r `density` (i32) | | |

`voxelFilter` additionally takes `@group(1) @binding(0)` = the axis uniform (three static buffers).
Five storage bindings in the solver, seven in skinning — both inside the default 8 per stage.

Render — three layouts:

| Layout | 0 | 1 | Used by |
|---|---|---|---|
| uniform only | u `RenderUniforms` (VS+FS) | | background (`BgUniforms`), mesh, floor, hair lines, reflection lines |
| + sim | u `RenderUniforms` | u `SimUniforms` (VS) | voxelDebug, bbox, colliders |
| + particles | u `RenderUniforms` | r `particles` (VS storage) | hair ribbons, reflection ribbons |

Vertex buffers: mesh/floor stride 32 (`position` float32x4 @0, `normal` float32x4 @16); hair lines
stride 48 (`pos` @0, `color` @32, `shade` = `prevPos.w` float32 @28); ribbons use none.

`SIM_U` grew from 240 to 1392 bytes: the old `pad0` at 236 is now `colliderCount`, followed by three
`array<vec4f, 24>` collider arrays at 240 / 624 / 1008. `RENDER_U` grew from 160 to 192 bytes:
`cameraPos` @144, `canvasWidth` @160, `canvasHeight` @164, `strandWidth` @168, `shading` @172,
`fade` @176.

Strand count is padded to a multiple of `strandsPerGroup` (17,365 → 17,368 for the beast), so the
solver dispatch is exact and needs no early-out — every `workgroupBarrier()` is in uniform control
flow. Padding strands are all `fix = 1`, parked at (−10000, −10000, −10000), skipped by the voxel
scatter's bounds check and never referenced by the hair index buffer.

No `GPUBuffer` or `GPUBindGroup` is created inside the frame loop; both ping-pong parities of every
grid-touching bind group are built up front.

## Hair rendering

**Ribbons (default).** One non-indexed `draw(6 * (N-1) * numStrands)`. `vertex_index` decodes to
`segment = vi / 6`, `corner = vi % 6`, `strand = segment / (N-1)`, `k = segment % (N-1)`, particles
`strand*N + k` and `+1`; the draw count only covers real strands, so the parked padding strands
cannot produce geometry. Both endpoints are projected, the segment direction is taken in *screen*
space and the quad is offset along its perpendicular by `strandWidth` device pixels (converted back
through `clip.w`), so a strand keeps its width at any distance. Width tapers linearly to 0.3 at the
tip. A zero-length segment falls back to a fixed direction and collapses to a dot rather than
producing NaNs.

The particle buffer is bound as a **storage buffer in the vertex stage**, which WebGPU defaults to
zero. `initWebGPU()` therefore asks for `maxStorageBuffersInVertexStage: 1` — but only when the
adapter reports the limit at all, because an unknown `requiredLimits` key is a hard error on older
implementations (which allowed vertex-stage storage under `maxStorageBuffersPerShaderStage` anyway).
If the limit is refused, `params.renderMode` is forced to `lines`, the GUI dropdown offers only that
value and a console warning explains why. Verified by temporarily hard-coding the capability to
`false`: the app runs, logs the warning, offers only `lines` and passes `?selftest`.

**Kajiya-Kay.** Tangent = `normalize(p1 - p0)` in world space, `L = normalize(0.4, 1.0, 0.25)`,
diffuse `sqrt(1 - dot(T,L)^2)` over an ambient floor of 0.45, specular
`pow(sqrt(1 - dot(T,H)^2), 32) * 0.25` with `H = normalize(L + V)` and `V` from the new `cameraPos`
uniform. The `pow(1/2.2)` gamma still runs last. The line path has no tangent (getting one would
mean pulling the neighbouring particle, which is exactly what that path exists to avoid), so
`shading` there means the self-shadow term only.

**Fur self-shadowing** is computed in the *solver*, not the renderer: `selfShadow()` takes three
density taps 1.5 cells apart marching towards `L`, and writes
`clamp(exp(-0.35 * sum / max(localDensity, 1)), 0.35, 1)` into the particle's otherwise unused
`prevPos.w`. Measuring occlusion *relative to the density at the particle* makes the term
self-calibrating: the beast packs ~140k particles into a 10³ box and the FurryBall ~1.3M into a 14³
box, so their absolute densities differ by an order of magnitude while the ratio deep inside the
coat does not. Turning `shading` off stops the renderer applying it; the solver still computes it
(one extra grid read per particle per step, and it keeps the buffer meaningful for a later toggle).

## Body collision

`beast.json`'s SPEC v2 `colliders` array is parsed at load, capped at `MAX_COLLIDERS - 1 = 23` (one
slot is reserved for the pointer brush) and validated entry by entry — a malformed capsule is
dropped with a warning, and a manifest with no section at all simply means ground-plane-only
collision. Every frame `skinColliders()` produces `modelMatrix x (palette[joint] x head)` per
endpoint and the result goes into `SimUniforms` (no extra binding: the arrays live in the uniform
buffer the solver, the bbox pass and the collider overlay already share). Both solvers run
`resolveColliders()` in exactly the place `checkCollision()` runs today — DFTL right after the
velocity estimate, PBD after the constraint solve — projecting the particle onto the capsule surface
and reflecting the normal velocity component with the plane response's `v = w - u`. Fixed particles
are skipped, as they are for the plane.

Sphere mode synthesises one degenerate capsule (a == b at the model origin, radius 4) and lets the
generic path carry it through the bounce — the original's commented-out sphere collision, now on.

## Fixed-timestep substepping

`SUBSTEP_DT = 1/120`, at most `MAX_SUBSTEPS = 3` steps per frame with the excess dropped outright.
The voxel grid is rebuilt **once** per frame (fill/post/filter, from the frame's particle state);
only the solver dispatch repeats, and because `deltaTime` is now the constant `SUBSTEP_DT` there is
nothing to re-upload between steps — the loop is `n` `dispatchWorkgroups` calls on one already-bound
pipeline. `voxelFill`'s `(pos - prevPos) / dt` uses the same `SUBSTEP_DT`, which is exactly the
interval `prevPos` now spans. Animation, skinning and collider positions are **held for the whole
frame**: at 40 fps that is at most 25 ms of collider lag, far cheaper than re-skinning and
re-uploading per step.

## Pointer interaction

A hovering mouse (`pointerType === 'mouse'`, `buttons === 0`) casts a ray through a per-model proxy —
the sphere at `simTranslation` with radius 4.5, or a vertical capsule from y 0.5 to 4.0 at radius 2.2
for the beast. Within `radius + 1.5` the closest approach is stepped back to the entry point and
snapped onto the proxy surface; that point becomes a degenerate capsule of radius 0.6 with
`colliderB.w = 1` marking it as carrying velocity `(hit - prevHit) / frameDt`, clamped to 40 units/s.
On contact the kernels add `v = mix(v, vCollider, 0.7)` to the usual pushout. The brush switches off
when the pointer leaves the canvas, a button goes down (so dragging still orbits) or 0.5 s pass
without movement; touch and pen are untouched.

Because PBD collides *after* integrating, a velocity change there would be discarded — so
`CollisionResult.dragged` is set only by a moving collider, and PBD rewrites `prevPos = pos - v*dt`
for exactly those particles. Static geometry (the plane, the body capsules) keeps the original's
`prevPos = oldPosition`.

## Robustness

* **Rest lengths live in `roots.w`.** The separate `strandLengths` storage buffer is gone; the
  skinning kernel writes `vec4f(p.xyz, roots[i].w)` so the packed value survives, and the solver
  reads `sharedLen[...] = roots[globalStrandIndex].w`.
* **Adaptive quality** (sphere only, and only without an explicit `?strands=`): after 120 frames the
  averaged GPU sim + render time is compared against a 14 ms budget; over it, the strand count is
  rescaled by `12 / measured` (floor 10,000), stored in `sessionStorage` and the page reloaded — at
  most twice per session. Skipped entirely without `timestamp-query`.
* **Device loss** logs a warning and reloads once, guarded by a `sessionStorage` flag that is cleared
  after five successful frames, so a crash loop ends at the banner instead of reloading forever.

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
7. **Gamma correction follows the original's shaders rather than being blanket-applied.** `hair` and
   `ribbon` apply `pow(c, 1/2.2)` in both their upright and their mirrored role (`basic_FS.glsl`,
   `floor_fs.glsl`), and `mesh` does too in its flat branch; `background`, `voxelDebug`, `bbox` and
   `colliders` do not, because the first three's originals are openFrameworks' own non-gamma path
   and `voxelGrid_fs.glsl`, and the last is a new debug overlay in the same spirit. Applying it to
   the background would wash the lightGray→white gradient almost flat, and applying it to the
   gradient-coloured voxel points would need the signed values clamped first. The body's *shaded*
   branch is the exception — see deviation 20.
8. **The selftest's length assertion skips segments the ground collision just repositioned** (either
   endpoint at y ≤ 1e−3) and reports per solver. PBD applies the plane collision *after* the
   constraint solve — faithfully to the original — so those segments are legitimately off their rest
   length for a frame; asserting on them measures the collision, not the solver. See "Known issue"
   below for why this matters so much with the current bake.
9. **DFTL and PBD share their bind groups** (identical layouts), so there are 2 rather than 4.
10. **All compute dispatches live in one compute pass** (SPEC's timestamp guidance: begin-of-skinning
    to end-of-sim). WebGPU orders dispatches within a pass and inserts the needed barriers.
11. **The sphere is 200×400 segments → 80,601 vertices before welding its poles (79,801 after)**,
    matching the demo video's tessellation rather than SPEC's "~128×128 ≈ 16k" placeholder.
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
17. **Collider radii are taken as sim-world lengths, not model-space ones.** SPEC v2 calls the
    section "bind/model space", which is true of the endpoints: the shipped bake's heads are raw DAE
    coordinates (y up to 182) that the palette's 0.01 and the model matrix's 2.62 bring down to
    world. Its radii, though, are 0.101 … 0.436 against a 4.5-tall beast — already fitted in world
    units. Scaling them by the same 0.0262 produced centimetre-thin capsules that nothing could
    touch; the `draw colliders` overlay showed the whole rig collapse to bare lines. `loadModel()`
    logs the world radius range so a future bake that changes convention is visible immediately.
18. **The self-shadow term is relative, not absolute.** SPEC's sketch is `exp(-k * sum)` with a tuned
    `k`; a single `k` cannot serve both modes, whose per-cell densities differ by ~10x (see "Hair
    rendering"). Dividing by the local density costs one extra tap and needs no per-scene constant.
19. **`shading` also drives the body.** SPEC only asks for a hair toggle, but tying the mesh to the
    same checkbox makes "off" mean the original look everywhere: `mesh.wgsl` then returns
    `gammaCorrect(overrideColor)` with `overrideColor = (0,0,0)`, i.e. the original's solid black,
    bit for bit. The body's own base colour lives in the shader so the flat path cannot pick it up.
20. **The mesh works in display space.** Its constants are final pixel values rather than linear ones
    put through `pow(1/2.2)` — at these levels the gamma curve is so steep that a linear 0.04 base
    lands at 0.24, well out of "near-black". Measured output on the sphere: 0.08 on the unlit side to
    0.21 lit, against 0.00 flat with shading off.
21. **The selftest excludes capsule contact from its length assertion**, exactly as it already
    excluded ground contact and for the same reason (deviation 8): PBD collides after the constraint
    solve, so a particle resting on a capsule is legitimately off its rest length for a frame. Only
    *movable* particles count as in contact — every FurryBall root sits exactly on its own collider
    by construction and would otherwise disqualify every first segment. Without this the sphere's PBD
    run measured 97.90 % (below the 99 % bar); with it, 100 % of 74,042 checked strands.

## Verification performed

`npx tsc --noEmit` and `npm run build` are clean.

WGSL is only validated at runtime, so beyond desk-checking, the built app was run in headless Chrome
against real Dawn, driven over the DevTools protocol so the **actual lil-gui widgets** could be
clicked and synthetic pointer events dispatched (which also closes most of the old "not verified
without an interactive browser" list):

* **No WGSL compilation messages and no WebGPU validation errors** in any run. All 13 shaders and 14
  pipelines were created and executed. A sweep of 28 GUI states per model — ribbons × lines ×
  DFTL × PBD × shading on/off × {no overlay, colliders, voxel grid + bounding box}, plus three
  mid-flight colour switches, the strand-width slider and `draw fur` off/on — produced no console
  error and no banner.
* **`?selftest` passes on both models**, with substepping and colliders active:
  * beast (17,365 strands × 8, 22 body capsules) — DFTL 100.00 % of 17,365 checked strands within
    25 % (worst ratio 1.000, max step 1.26 rest lengths); PBD 100.00 % of 17,364 within 100 %
    (worst 1.236). 120 fixed solver steps per solver, 0.6 % / 1.4 % of strands on the floor.
  * sphere (80,601 strands × 16, 1,289,616 particles, 1 capsule) — DFTL 100.00 % of 69,079 within
    25 % (worst 1.000, max step 1.74); PBD 100.00 % of 74,042 within 100 % (worst 1.427, max step
    1.89). 31.8 % / 10.4 % of strands rest on the ball collider.
  * Substepping visibly *improved* stability: the sphere's worst-case per-step motion fell from 6.83
    to 1.74 rest lengths (DFTL) and 3.12 to 1.89 (PBD) at the same simulated duration.
* **The vertex-storage fallback** was exercised by hard-coding the capability to `false` for one
  build: warning logged, `render` dropdown reduced to `["lines"]`, `?selftest` still PASS.
* **Adaptive quality** was driven over budget by seeding `sessionStorage` with 150,000 strands at
  2× device scale: 149,878 strands measured 18.0 ms/frame → rescaled to 99,723 → reloaded at 100,128
  strands → 12.2 ms, within budget, and the count persisted. At the default 80,601 it correctly
  measures 8.8 ms and leaves the scene alone.
* **Colliders** were inspected with `draw colliders` on and `draw fur` off: 22 capsules trace a
  recognisable rig (torso, limbs, head) inside the beast, and the sphere's degenerate capsule draws
  as three orthogonal great circles. This is how deviation 17 (radius units) was caught.
* **Pointer brushing** was driven with synthetic hovering `pointermove` events: the brush capsule
  appears in the overlay on the proxy surface under the cursor, and an A/B against an unbrushed run
  shows the fur combed along the sweep. A left-button drag orbits and does *not* brush.
* **The body shading** was measured on a sparse sphere (`?strands=1200`, where the mesh is visible
  between strands): 0.078 → 0.208 top to bottom with shading on, exactly 0.000 with it off.
* **GPU timestamps** (headless, 1280×769, Apple Silicon — indicative only):

  | Scene | before (lines, 1 step/frame) | ribbons + shading | lines |
  |---|---|---|---|
  | beast, 138,920 particles | 0.95 sim / 1.13 render | 0.5–0.9 / **2.0–3.1** | 0.9 / 1.2 |
  | FurryBall, 1,289,616 particles | 2.57 / 2.05 | 2.0 / **5.9–6.2** | 2.0 / 1.8 |

  Ribbons cost ~3× the line path's render time (5.6× the vertices, plus real fill), drawn twice for
  the reflection. Sim time is *flat or lower* despite substepping, colliders and the self-shadow taps
  because the headless loop runs at ~120 fps, where the accumulator issues one step per frame; at
  60 fps it issues two and the sim cost roughly doubles.
* The uniform offset tables are confirmed indirectly but strongly: the bounding box draws at the
  exact `minBB`/`maxBB` extents (offsets 160/176), the voxel points sit where the sim samples them
  (offset 128), the solver is stable with the settings.xml defaults (offsets 192–232), and the
  collider arrays at 240/624/1008 draw the rig in the right place.

### Resolved — the UV sphere's poles

A UV sphere duplicated `sectors+1` vertices at each pole, so 401 of the FurryBall's strands used to
start at exactly the same point; with the non-normalised gradient that concentration was a large
local density spike, and the repulsion blew a small crater in the fur at each pole (a ring of exposed
black mesh when the pole faced the camera). **`createSphereModel` now welds each pole to a single
vertex** — the triangle mesh fans to it instead of the old duplicate-vertex quads (which were
degenerate anyway) — dropping the strand count from the video's 80,601 to 79,801, `2 * sectors` fewer
roots with the rest of the tessellation unchanged. Verified in headless Chrome: `?model=sphere&selftest`
still passes (DFTL 100.00% of 69,270 and PBD 100.00% of 73,686 checked strands within tolerance) and
the fur is visibly continuous at both poles with no exposed-mesh ring, animation paused or running.

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

### Still not verified

* **`device.lost` recovery.** The handler, its `sessionStorage` guard and the clear-after-five-frames
  step are desk-checked only; a genuine device loss could not be provoked (`device.destroy()` reports
  `reason: 'destroyed'`, which the handler deliberately ignores). The guard key *is* confirmed absent
  after a healthy run, so the clearing half works.
* **The `f` fullscreen toggle, DPR changes and live canvas resizing.** The resize path runs every
  frame but has only been driven at two sizes (1280×769 and 3800×2400 via `--force-device-scale-factor`).
* **Model switching through the dropdown** still goes via a full reload, so only its URL-parameter
  equivalent is covered.
* **Touch input** (which must stay pure orbit) — no synthetic touch events were dispatched; the guard
  is a single `pointerType !== 'mouse'` early-out.
* **Non-Dawn implementations** (Safari/WebKit, Firefox/wgpu). Firefox is stricter about uniformity
  analysis — every barrier here is at function-body or loop-body scope, never inside a conditional —
  and neither has been tried with a vertex-stage storage buffer, which is the one new capability the
  ribbon path needs. The line fallback exists precisely for that.
* **Visual fidelity against the original video**: the gradient multipliers were matched analytically
  against sampled hex values, not eyeballed side by side with video frames, and the video build had
  no strand shading at all to compare against.
