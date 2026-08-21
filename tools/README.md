# tools/

Offline model-baking pipeline: converts the Blender/Collada export of the Beast
walk-cycle into the compact binary format the WebGPU engine loads at runtime.

## Usage

```
npm run bake
```

(equivalently: `python3 tools/bake_dae.py`)

This reads `assets_src/beast_walking_inplace_17k.dae` and writes:

- `public/models/beast.bin` — positions, normals, triangle indices, per-vertex
  joint indices/weights, and per-frame skinning palettes, packed little-endian.
- `public/models/beast.json` — manifest describing counts, `modelMatrix`, and the
  byte offset/length of each section in `beast.bin`.

The exact output schema is defined in `SPEC.md` ("Model manifest") — that's the
contract the engine (`src/`) codes against; treat it as the source of truth over
this README.

After writing the files, `bake_dae.py` automatically runs `validate_bake.py` in a
fresh subprocess (so validation re-reads the artifacts from disk rather than
trusting in-memory state) and exits non-zero if anything looks wrong.

## Requirements

Python 3.12, standard library only — `xml.etree.ElementTree`, `struct`, `array`,
`json`, `math`. No numpy, no pip install.

## Validating an existing bake

```
python3 tools/validate_bake.py
```

Runs standalone against whatever is currently in `public/models/`. Checks: all
floats finite, indices within range, joint indices within range, weights sum to
~1 per vertex, CPU-skins **every** frame with the stored palettes, prints a
per-frame world-space minY/maxY table, and asserts: every frame's minY >= -0.01
(no frame sinks below the floor), the union of Y bounds across all frames is
minY in [-0.01,0.01] / maxY in [4.49,4.51] (the intended height-4.5,
feet-at-floor normalization), the union X/Z bounds are within a sane range,
the animation actually moves vertices between frames, and palettes differ
across frames.

## Files

- `bake_dae.py` — the baker. Parses geometry, skin (bind shape matrix, joint
  names/order, inverse bind matrices, per-vertex influences), baked per-joint
  animation tracks, and the visual-scene node hierarchy; computes one skinning
  palette per (frame, joint); CPU-skins **every** frame to derive `modelMatrix`
  from the union of bounds across the whole cycle (see note below); writes the
  binary + JSON.
- `validate_bake.py` — independent reload-and-check script, described above.

## Notes / assumptions worth knowing about

- The DAE's `<polylist>` is generically parsed (VERTEX/NORMAL/TEXCOORD input
  offsets are read from the file, not assumed) and fan-triangulated per polygon,
  even though every polygon in this particular file already has `vcount=3`.
- Per-vertex normals are the normalized sum of every per-corner normal indexed to
  that position (the DAE's normal stream has a different count than the position
  stream — normals are split/indexed independently, as Blender exports them).
- A vertex can have more than 4 skin influences in this file (up to 6 observed);
  the top 4 by weight are kept and renormalized to sum to 1, per the schema.
- The scene root ("Armature") node is not the identity — it carries a real
  `<scale>0.01 0.01 0.01</scale>` that converts the joint hierarchy's raw
  centimeter-scale numbers into meters, consistent with the mesh's own
  `bind_shape_matrix` (also a 0.01 scale). This was verified empirically:
  `worldTransform(joint, bind pose) x inverseBind(joint)` is only the identity
  matrix when Armature's own static transform is included in the product — so
  "world transform from the scene root" is taken literally, starting at
  Armature, not at the first joint.
- `modelMatrix` is derived by CPU-skinning every frame (not just frame 0) in
  Python, with the same top-4/renormalized weights that ship in `beast.bin` (not
  the full unfiltered influence list), so it matches what the runtime will
  actually render. The scale/translate are fit to the **union** of axis-fixed
  bounds across all 52 frames, not to frame 0 alone: an earlier version
  normalized against frame 0 only, and since frame 0 happens to be a relatively
  tall/neutral pose in this walk cycle (not representative of the full cycle's Y
  extent), mid-stride frames sank up to ~3 world units below the floor under
  that normalization. Using the union guarantees every frame's lowest point is
  at/above y=0 and the cycle's single tallest point is exactly y=4.5; individual
  frames legitimately fall short of touching both bounds (e.g. a crouch phase
  stays below the max height, which is correct — it shouldn't be stretched to
  fill it). `bake_dae.py` prints a per-frame minY/maxY table under both the old
  and new normalization so a regression here is easy to spot again.
- The axis fix applied before that scale/translate (`AXIS_FIX` in `bake_dae.py`)
  is the **identity** — no rotation — even though the DAE's `<up_axis>` tag reads
  `Z_UP`. That tag turned out not to reflect the actual baked data: an earlier
  version rotated -90° about X (a standard Z-up→Y-up fix), which put the
  skeleton's largest extent (1.72 units) into the horizontal plane and rendered
  the character tipped onto its side. The raw per-frame skinned bounds are
  already Y-up (model Y is both the largest extent, consistent with height
  dominating for this upright bipedal rig, and its minimum sits almost exactly
  at 0, the signature of a "feet on the ground" axis) — Blender's exporter must
  have baked the orientation into the joint/animation data directly. See the
  `AXIS_FIX` comment block in `bake_dae.py` for the full numeric evidence.
