#!/usr/bin/env python3
"""
validate_bake.py -- Sanity-checks public/models/beast.bin + beast.json against the
schema defined in SPEC.md ("Model manifest").

Always reloads both files fresh from disk (does not import bake_dae.py or reuse any
in-memory state), so it also catches serialization bugs, not just baking-logic bugs.

Usage:
    python3 tools/validate_bake.py

Exits non-zero (and prints "VALIDATION FAILED") on any check failure.
Called automatically by bake_dae.py after a successful bake; also runnable standalone.
"""

from __future__ import annotations

import json
import math
import struct
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
JSON_PATH = PROJECT_ROOT / "public" / "models" / "beast.json"
BIN_PATH = PROJECT_ROOT / "public" / "models" / "beast.bin"


def slice_f32(raw: bytes, offset: int, length: int) -> list[float]:
    n = length // 4
    return list(struct.unpack_from(f"<{n}f", raw, offset))


def slice_u32(raw: bytes, offset: int, length: int) -> list[int]:
    n = length // 4
    return list(struct.unpack_from(f"<{n}I", raw, offset))


def slice_u16(raw: bytes, offset: int, length: int) -> list[int]:
    n = length // 2
    return list(struct.unpack_from(f"<{n}H", raw, offset))


def apply_model_matrix(m_col_major: list[float], p: tuple[float, float, float]
                        ) -> tuple[float, float, float]:
    """m is a flat column-major 4x4 (m[col*4+row]); apply to point p (implicit w=1)."""
    x, y, z = p
    rx = m_col_major[0] * x + m_col_major[4] * y + m_col_major[8] * z + m_col_major[12]
    ry = m_col_major[1] * x + m_col_major[5] * y + m_col_major[9] * z + m_col_major[13]
    rz = m_col_major[2] * x + m_col_major[6] * y + m_col_major[10] * z + m_col_major[14]
    return (rx, ry, rz)


# Same math as apply_model_matrix, under a name that also reads correctly when the matrix is
# a per-joint skinning palette rather than modelMatrix (colliders section below: SPEC has the
# runtime apply `palette[joint] x head` with weight 1, then modelMatrix -- both steps are this
# same column-major-4x4-times-point operation).
apply_mat4_point = apply_model_matrix


def matrix_uniform_scale(m_col_major: list[float]) -> float:
    """Length of the first basis column. bake_dae.py's modelMatrix is always
    Translate(tx,ty,tz) @ Scale(s,s,s) @ AxisFix(identity) -- a uniform scale -- so any basis
    column's length equals that scale factor s. Used only to print an eyeball-friendly
    world-space radius alongside the collider placement table below; SPEC does not say the
    runtime itself must scale the stored radius by this (unlike headA/headB, whose transform
    is fully specified), so this is a diagnostic multiply, not a schema requirement."""
    return math.sqrt(m_col_major[0] ** 2 + m_col_major[1] ** 2 + m_col_major[2] ** 2)


def fail(msg: str) -> "NoReturn":
    print(f"VALIDATION FAILED: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    if not JSON_PATH.exists() or not BIN_PATH.exists():
        fail(f"missing output files -- expected {JSON_PATH} and {BIN_PATH}")

    meta = json.loads(JSON_PATH.read_text())
    raw = BIN_PATH.read_bytes()

    for key in ("version", "vertexCount", "indexCount", "jointCount", "frameCount",
                "duration", "modelMatrix", "buffers"):
        if key not in meta:
            fail(f"beast.json missing top-level key {key!r}")

    V = meta["vertexCount"]
    I = meta["indexCount"]
    J = meta["jointCount"]
    F = meta["frameCount"]
    buffers = meta["buffers"]
    for name in ("positions", "normals", "indices", "joints", "weights", "palettes"):
        if name not in buffers:
            fail(f"beast.json buffers missing {name!r}")

    print(f"Manifest: version={meta['version']} vertexCount={V} indexCount={I} "
          f"jointCount={J} frameCount={F} duration={meta['duration']}")

    positions = slice_f32(raw, buffers["positions"]["offset"], buffers["positions"]["length"])
    normals = slice_f32(raw, buffers["normals"]["offset"], buffers["normals"]["length"])
    indices = slice_u32(raw, buffers["indices"]["offset"], buffers["indices"]["length"])
    joints = slice_u16(raw, buffers["joints"]["offset"], buffers["joints"]["length"])
    weights = slice_f32(raw, buffers["weights"]["offset"], buffers["weights"]["length"])
    palettes = slice_f32(raw, buffers["palettes"]["offset"], buffers["palettes"]["length"])

    if len(positions) != V * 3:
        fail(f"positions length {len(positions)} != vertexCount*3 ({V * 3})")
    if len(normals) != V * 3:
        fail(f"normals length {len(normals)} != vertexCount*3 ({V * 3})")
    if len(indices) != I:
        fail(f"indices length {len(indices)} != indexCount ({I})")
    if len(joints) != V * 4:
        fail(f"joints length {len(joints)} != vertexCount*4 ({V * 4})")
    if len(weights) != V * 4:
        fail(f"weights length {len(weights)} != vertexCount*4 ({V * 4})")
    if len(palettes) != 16 * J * F:
        fail(f"palettes length {len(palettes)} != 16*jointCount*frameCount ({16 * J * F})")
    if len(meta["modelMatrix"]) != 16:
        fail(f"modelMatrix has {len(meta['modelMatrix'])} floats, expected 16")

    print("Buffer lengths match manifest header. Checking finiteness...")

    def assert_all_finite(name: str, values) -> None:
        for idx, v in enumerate(values):
            if not math.isfinite(v):
                fail(f"{name}[{idx}] is not finite: {v}")

    assert_all_finite("positions", positions)
    assert_all_finite("normals", normals)
    assert_all_finite("weights", weights)
    assert_all_finite("palettes", palettes)
    assert_all_finite("modelMatrix", meta["modelMatrix"])
    print("  all finite: OK")

    max_index = max(indices) if indices else -1
    if max_index >= V:
        fail(f"max index {max_index} >= vertexCount {V}")
    print(f"  indices < vertexCount: OK (max index {max_index})")

    max_joint = max(joints) if joints else -1
    if max_joint >= J:
        fail(f"max joint index {max_joint} >= jointCount {J}")
    print(f"  joints < jointCount: OK (max joint index {max_joint})")

    bad_sum_count = 0
    worst = None
    for i in range(V):
        s = weights[i * 4] + weights[i * 4 + 1] + weights[i * 4 + 2] + weights[i * 4 + 3]
        if not (0.999 <= s <= 1.001):
            bad_sum_count += 1
            if worst is None or abs(s - 1.0) > abs(worst[1] - 1.0):
                worst = (i, s)
    if bad_sum_count:
        fail(f"{bad_sum_count} vertices have weight sum outside [0.999, 1.001] "
             f"(worst: vertex {worst[0]} sum={worst[1]})")
    print(f"  weights sum to 1.0 (+/- 0.001) for all {V} vertices: OK")

    # ---- CPU-skin every frame with the stored palettes; apply modelMatrix ----
    # (modelMatrix is normalized against the UNION of bounds across all frames -- see
    # bake_dae.py -- so no single frame is expected to touch both y=0 and y=4.5; what must
    # hold is that EVERY frame's lowest point stays at/above the floor, and the union across
    # the whole cycle hits the [0, 4.5] target exactly.)
    model_matrix = meta["modelMatrix"]

    def palette_for(frame: int, joint: int) -> list[float]:
        base = (frame * J + joint) * 16
        return palettes[base:base + 16]

    def skin_and_transform(frame: int) -> list[tuple[float, float, float]]:
        pal_cache = [palette_for(frame, j) for j in range(J)]
        out = [(0.0, 0.0, 0.0)] * V
        for i in range(V):
            px, py, pz = positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]
            base = i * 4
            sx = sy = sz = 0.0
            for k in range(4):
                w = weights[base + k]
                if w == 0.0:
                    continue
                j = joints[base + k]
                m = pal_cache[j]
                rx = m[0] * px + m[4] * py + m[8] * pz + m[12]
                ry = m[1] * px + m[5] * py + m[9] * pz + m[13]
                rz = m[2] * px + m[6] * py + m[10] * pz + m[14]
                sx += w * rx
                sy += w * ry
                sz += w * rz
            world = apply_model_matrix(model_matrix, (sx, sy, sz))
            if not (math.isfinite(world[0]) and math.isfinite(world[1]) and math.isfinite(world[2])):
                fail(f"non-finite skinned world position at frame {frame} vertex {i}: {world}")
            out[i] = world
        return out

    # Frames 0 and 26 are needed in full (not just their bounds) for the displacement
    # check below; every other frame only needs its bounds, so we don't retain 52 full
    # V-length position lists at once.
    KEEP_FULL_FRAMES = {0, 26}
    frame_worlds: dict[int, list[tuple[float, float, float]]] = {}
    frame_bounds: list[tuple[float, float, float, float, float, float]] = []

    print(f"CPU-skinning all {F} frames and applying modelMatrix...")
    for f in range(F):
        world = skin_and_transform(f)
        xs = [p[0] for p in world]
        ys = [p[1] for p in world]
        zs = [p[2] for p in world]
        frame_bounds.append((min(xs), max(xs), min(ys), max(ys), min(zs), max(zs)))
        if f in KEEP_FULL_FRAMES:
            frame_worlds[f] = world
    if max(KEEP_FULL_FRAMES) >= F:
        fail(f"expected frame {max(KEEP_FULL_FRAMES)} to exist but frameCount is {F}")

    print(f"\n  per-frame world Y bounds (all {F} frames):")
    print(f"  {'frame':>5} {'minY':>9} {'maxY':>9}")
    below_floor = []
    for f, (min_x, max_x, min_y, max_y, min_z, max_z) in enumerate(frame_bounds):
        flag = "  <-- SINKS BELOW FLOOR" if min_y < -0.01 else ""
        print(f"  {f:5d} {min_y:9.4f} {max_y:9.4f}{flag}")
        if min_y < -0.01:
            below_floor.append((f, min_y))

    if below_floor:
        fail(f"{len(below_floor)} frame(s) sink below the floor (minY < -0.01): "
             f"{below_floor[:5]}{' ...' if len(below_floor) > 5 else ''}")
    print(f"  all {F} frames stay at/above the floor (minY >= -0.01): OK")

    union_min_x = min(b[0] for b in frame_bounds)
    union_max_x = max(b[1] for b in frame_bounds)
    union_min_y = min(b[2] for b in frame_bounds)
    union_max_y = max(b[3] for b in frame_bounds)
    union_min_z = min(b[4] for b in frame_bounds)
    union_max_z = max(b[5] for b in frame_bounds)
    print(f"\n  union world bounds: X[{union_min_x:.3f},{union_max_x:.3f}] "
          f"Y[{union_min_y:.3f},{union_max_y:.3f}] Z[{union_min_z:.3f},{union_max_z:.3f}]")

    if not (-0.01 <= union_min_y <= 0.01):
        fail(f"union minY {union_min_y} not in [-0.01, 0.01]")
    if not (4.49 <= union_max_y <= 4.51):
        fail(f"union maxY {union_max_y} not in [4.49, 4.51]")
    print("  union minY in [-0.01,0.01] and maxY in [4.49,4.51]: OK")

    # "Sane" X/Z bound: generous enough not to fight a long-bodied quadruped mid-stride,
    # tight enough to catch a genuinely broken (e.g. exploded/mis-normalized) bake.
    XZ_SANE_LIMIT = 8.0
    if not (abs(union_min_x) < XZ_SANE_LIMIT and abs(union_max_x) < XZ_SANE_LIMIT):
        fail(f"union X bounds [{union_min_x},{union_max_x}] exceed sane limit +/-{XZ_SANE_LIMIT}")
    if not (abs(union_min_z) < XZ_SANE_LIMIT and abs(union_max_z) < XZ_SANE_LIMIT):
        fail(f"union Z bounds [{union_min_z},{union_max_z}] exceed sane limit +/-{XZ_SANE_LIMIT}")
    print(f"  union X/Z bounds within +/-{XZ_SANE_LIMIT}: OK")

    # ---- animation actually moves ----
    w0, w26 = frame_worlds[0], frame_worlds[26]
    max_disp = 0.0
    for i in range(V):
        ax, ay, az = w0[i]
        bx, by, bz = w26[i]
        d = math.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2)
        if d > max_disp:
            max_disp = d
    print(f"  max vertex displacement frame 0 -> frame 26: {max_disp:.4f} world units")
    if not (max_disp > 0.05):
        fail(f"max displacement {max_disp} <= 0.05 -- animation does not appear to move")

    # ---- a mid joint's palette actually changes across frames ----
    mid_joint = J // 2
    p0 = palette_for(0, mid_joint)
    p26 = palette_for(26, mid_joint)
    max_diff = max(abs(a - b) for a, b in zip(p0, p26))
    print(f"  joint {mid_joint} palette max abs diff frame 0 vs frame 26: {max_diff:.5f}")
    if not (max_diff > 1e-6):
        fail(f"mid joint {mid_joint} palette does not differ between frame 0 and frame 26")

    # ---- colliders extension (v2, optional) ----
    colliders = meta.get("colliders")
    if colliders is None:
        print("\nNo 'colliders' section in the manifest -- OK, it's optional per SPEC "
              "(engines fall back to floor-only collision).")
    else:
        print(f"\nValidating 'colliders' section ({len(colliders)} capsule(s))...")
        if not (6 <= len(colliders) <= 24):
            fail(f"collider count {len(colliders)} not in [6,24]")
        for idx, cap in enumerate(colliders):
            for key in ("jointA", "headA", "jointB", "headB", "radius"):
                if key not in cap:
                    fail(f"colliders[{idx}] missing key {key!r}")
            if not (0 <= cap["jointA"] < J):
                fail(f"colliders[{idx}].jointA {cap['jointA']} out of range [0,{J})")
            if not (0 <= cap["jointB"] < J):
                fail(f"colliders[{idx}].jointB {cap['jointB']} out of range [0,{J})")
            if len(cap["headA"]) != 3:
                fail(f"colliders[{idx}].headA has {len(cap['headA'])} components, expected 3")
            if len(cap["headB"]) != 3:
                fail(f"colliders[{idx}].headB has {len(cap['headB'])} components, expected 3")
            for field in ("headA", "headB"):
                if not all(math.isfinite(v) for v in cap[field]):
                    fail(f"colliders[{idx}].{field} has a non-finite component: {cap[field]}")
            if not math.isfinite(cap["radius"]):
                fail(f"colliders[{idx}].radius is not finite: {cap['radius']}")
            if not (cap["radius"] > 0.0):
                fail(f"colliders[{idx}].radius must be positive, got {cap['radius']}")
            # radius is specified as a plain scalar in the final "world-space collider
            # array" (SPEC: Capsule.a.w), with no palette/modelMatrix multiply defined for
            # it anywhere (unlike headA/headB) -- so unlike the endpoints, it must ALREADY
            # be a sane world-space magnitude in the JSON itself. Bound it well above any
            # plausible real radius (union Y extent is exactly 4.5 by construction) but well
            # below "clearly the wrong units", to catch a raw/pre-bind_shape_matrix-space
            # radius slipping through uncaught (that bug produced radii of ~6-44 here).
            if not (cap["radius"] < 0.5 * (union_max_y - union_min_y)):
                fail(f"colliders[{idx}].radius {cap['radius']:.4f} is implausibly large next "
                     f"to the model's world height {union_max_y - union_min_y:.4f} -- looks "
                     f"like it's still in raw/bind-shape space rather than world space")
        print(f"  schema OK: count in [6,24], joint indices in range, headA/headB finite, "
              f"radius finite/positive/plausible")

        # CPU-skin each capsule endpoint with weight 1 to its own joint (SPEC: "Runtime skins
        # each endpoint with weight 1 to its joint"), at frames 0 and 26 -- the same two
        # frames already CPU-skinned above for the displacement check -- then apply
        # modelMatrix, exactly like the runtime is specified to. Print a human-eyeballable
        # table and assert every capsule's midpoint (frames 0 and 26) lands inside the union
        # world bounds already computed above. Unlike headA/headB, "radius" carries no
        # palette/modelMatrix transform of its own (see bake_dae.py's colliders_json comment)
        # -- it's written to the manifest already in world units, so it's printed verbatim.
        print(f"\n  modelMatrix uniform scale (informational only; NOT applied to radius, "
              f"which is already world-space -- see bake_dae.py): {matrix_uniform_scale(model_matrix):.6f}")
        print(f"  Per-capsule world-space placement:")
        print(f"  {'idx':>3} {'jointA':>6} {'jointB':>6} {'frame':>5}  "
              f"{'world A':^24}  {'world B':^24}  {'radius':>9}")
        for idx, cap in enumerate(colliders):
            jA, jB = cap["jointA"], cap["jointB"]
            head_a = tuple(cap["headA"])
            head_b = tuple(cap["headB"])
            for f in (0, 26):
                local_a = apply_mat4_point(palette_for(f, jA), head_a)
                local_b = apply_mat4_point(palette_for(f, jB), head_b)
                world_a = apply_model_matrix(model_matrix, local_a)
                world_b = apply_model_matrix(model_matrix, local_b)
                if not all(math.isfinite(v) for v in (*world_a, *world_b)):
                    fail(f"colliders[{idx}] frame {f}: non-finite world endpoint "
                         f"(A={world_a}, B={world_b})")
                mid = tuple((world_a[k] + world_b[k]) / 2.0 for k in range(3))
                eps = 1e-4
                if not (union_min_x - eps <= mid[0] <= union_max_x + eps and
                        union_min_y - eps <= mid[1] <= union_max_y + eps and
                        union_min_z - eps <= mid[2] <= union_max_z + eps):
                    fail(f"colliders[{idx}] frame {f}: midpoint {mid} outside union world "
                         f"bounds X[{union_min_x:.3f},{union_max_x:.3f}] "
                         f"Y[{union_min_y:.3f},{union_max_y:.3f}] "
                         f"Z[{union_min_z:.3f},{union_max_z:.3f}]")
                print(f"  {idx:3d} {jA:6d} {jB:6d} {f:5d}  "
                      f"({world_a[0]:6.3f},{world_a[1]:6.3f},{world_a[2]:6.3f})  "
                      f"({world_b[0]:6.3f},{world_b[1]:6.3f},{world_b[2]:6.3f})  "
                      f"{cap['radius']:9.4f}")
        print(f"\n  all {len(colliders)} capsules' midpoints (frames 0, 26) lie within the "
              f"union world bounds: OK")

    # ---- file sizes ----
    bin_size = BIN_PATH.stat().st_size
    json_size = JSON_PATH.stat().st_size
    total = bin_size + json_size
    print(f"\nbeast.bin:  {bin_size:,} bytes ({bin_size / 1e6:.3f} MB)")
    print(f"beast.json: {json_size:,} bytes ({json_size / 1e3:.1f} KB)")
    print(f"total:      {total:,} bytes ({total / 1e6:.3f} MB)  (expected roughly 1.5-2.5 MB)")

    print("\nVALIDATION PASSED")


if __name__ == "__main__":
    main()
