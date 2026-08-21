# HairyMess Web — Port Plan

Port of [patrickFuerst/hairymess](https://github.com/patrickFuerst/hairymess) (2015, OpenGL 4.4 compute
shaders + openFrameworks, Windows/NVIDIA only) to the browser.

## Technology choice

**WebGPU + WGSL, raw (no 3D framework), TypeScript + Vite.**

WebGPU is the only web graphics API with compute shaders, storage buffers and workgroup shared
memory — a 1:1 conceptual match for the original's OpenGL 4.4 compute pipeline. It ships today in
Chrome/Edge (since 113), Safari 26+ and Firefox 141+. WebGL2 has no compute path, so it is not a
viable target for this simulation.

| Original (OpenGL 4.4)                         | Web port (WebGPU)                                     |
|-----------------------------------------------|-------------------------------------------------------|
| GLSL compute shaders                          | WGSL compute shaders                                  |
| SSBOs (`std140`/`std430`)                     | storage buffers (explicit WGSL layout)                |
| `shared` arrays + `barrier()`                 | `var<workgroup>` + `workgroupBarrier()` (restructured into uniform control flow) |
| `GL_NV_shader_atomic_float` voxel scatter     | fixed-point `atomic<i32>` scatter (portable)          |
| GLSL subroutines (DFTL / PBD)                 | two compute pipelines                                 |
| `glClearBufferData`                           | `GPUCommandEncoder.clearBuffer()`                     |
| CPU skinning via Assimp (ofxAssimpModelLoader)| GPU skinning in a compute pass (joint palettes baked offline) |
| Primitive-restart line strips                 | `line-strip` topology + 0xFFFFFFFF restart (native)   |
| 8× MSAA, stencil floor reflection             | 4× MSAA, `depth24plus-stencil8` stencil reflection    |
| ofxGui panel                                  | lil-gui panel + HTML HUD                              |

## Efficiency principles (the point of the exercise)

- **Zero per-frame geometry traffic.** Particles live in one GPU buffer with usage
  `STORAGE | VERTEX`; the simulation writes it, the renderer draws lines straight from it. No copies,
  no readbacks.
- **GPU skinning.** The walk-cycle joint matrices are baked offline into per-frame palettes
  (65 joints × 52 frames ≈ 216 KB). Per frame the CPU lerps one palette (4 KB upload); a compute pass
  skins the 17k roots on GPU. The same skinned buffer feeds both the hair roots and the mesh render.
- **One command encoder, one submit per frame**; prebuilt bind groups (ping-pong via two prebuilt
  groups, no per-frame allocation); voxel buffers cleared with `clearBuffer` (GPU-side zero fill).
- **Fixed-point atomics** for the trilinear density/velocity scatter (WGSL has no float atomics):
  density scale 4096, velocity scale 1024 — precision ~2.4e-4 with huge overflow headroom.
- **Workgroup memory strand solver** exactly like the original: 64 threads/group = 8 strands × 8
  particles, positions solved in `var<workgroup>` arrays. The original relied on NVIDIA-specific
  barrier laxness; the port hoists barriers into uniform control flow (required by WGSL, and more
  correct).
- Optional **GPU timestamp queries** (feature-gated) surface per-pass times in the HUD.

## Frame graph

1. `clearBuffer` voxel atomic buffers (density i32, velocity i32×4)
2. **Skinning** compute: joint palette × bind positions → world-space root positions (17k threads)
3. **Voxel fill** compute: trilinear fixed-point atomic scatter of density + velocity (139k threads)
4. **Voxel post** compute: velocity /= density; density gradient via central differences (64³)
5. **Voxel filter** compute ×3 (optional): separable 3-tap box blur, ping-pong (64³ each)
6. **Simulation** compute (DFTL or PBD pipeline): integrate, friction/repulsion from grid,
   plane collision, length constraints in workgroup memory (139k threads)
7. **Render** (single pass, 4× MSAA + depth/stencil): background gradient → black skinned mesh →
   hair line strips → floor stencil mask → mirrored hair reflection with fade → optional voxel debug
   points

## Model pipeline

`tools/bake_dae.py` parses the original `beast_walking_inplace_17k.dae` (Collada, baked per-joint
matrix animation, 65 joints, 52 keyframes) with Python stdlib XML, and writes a compact binary
(`public/models/beast.bin` + `beast.json` manifest): positions, normals, triangle indices, 4×
joints/weights per vertex, and **final skinning palettes per frame** (world × inverseBind ×
bindShape, Z-up→Y-up and scale normalization applied). Runtime model loading is ~50 lines — no
Assimp, no glTF parser, no scene graph.

Fallback: `?model=sphere` (or a failed model fetch) grows the fur on a procedural UV sphere — the
original "Furry Ball".

## Faithfulness

Defaults match the original `bin/data/settings.xml` (velocityDamping 0.985, 30 PBD iterations,
stiffness 1.0, friction 0.066, repulsion 117, ftlDamping 1.0, gravity −10y, filter on). Both solver
algorithms, the voxel-grid debug view, bounding-box display, the alternating orange/teal strand
colors, gamma-corrected output, and the floor-reflection trick are all ported.

## Verification

- `?selftest` mode: runs N frames, reads the particle buffer back, asserts all positions finite,
  segment lengths ≈ rest length, roots tracking the skinned mesh; reports PASS/FAIL in title+console.
- Manual: dev server + browser screenshot review; FPS/GPU timings in HUD.
