#!/usr/bin/env python3
"""
bake_dae.py -- Offline model-baking pipeline for the HairyMess WebGPU port.

Parses the Collada (.dae) export of the "Beast" walk-cycle mesh and writes a compact
binary + JSON manifest pair consumed directly by the runtime loader:

    public/models/beast.bin
    public/models/beast.json

Python 3.12, standard library only (xml.etree.ElementTree, struct, array, json, math).
No numpy, no third-party packages.

Usage:
    python3 tools/bake_dae.py
    (equivalently: npm run bake)

The exact output schema is defined by SPEC.md, section "Model manifest" -- read that
file before changing anything here. After baking, this script shells out to
validate_bake.py (a fresh process, so it re-reads the files from disk rather than
reusing in-memory state) and fails loudly if validation does not pass.

---------------------------------------------------------------------------------------
Notes on the Collada math that are easy to get wrong (learned by inspecting this file):

* All matrices in the DAE (bind_shape_matrix, joint <matrix> elements, baked animation
  output matrices, inverse bind matrices) are stored as 16 floats in ROW-MAJOR order in
  document order. This script keeps everything in that same row-major convention
  internally (a flat 16-list, m[row*4+col]) and only transposes to column-major at the
  very end, when writing modelMatrix / palettes to match SPEC.md.

* The scene root "Armature" node is NOT the identity transform: it carries a real
  <scale>0.01 0.01 0.01</scale> (its <translate>/<rotate> siblings are all zero/angle-0
  in this file). This 0.01 converts the raw joint-hierarchy numbers -- which are
  expressed in centimeters, e.g. translations around ~100 -- into meters, consistently
  with the mesh's own bind_shape_matrix (also a 0.01 scale, converting the raw mesh
  positions, also centimeters, e.g. ~160, into meters).
  This was verified empirically: worldTransform(Hips, bind pose) x inverseBind(Hips)
  is the identity matrix ONLY when the Armature node's own static transform is included
  in the product. Omitting it (treating Armature as identity) leaves a stray 100x scale
  in the result. So "worldTransform = product down the node hierarchy from scene root"
  really does mean starting at Armature and composing its real local matrix, not
  skipping it.
"""

from __future__ import annotations

import array
import json
import math
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from collections import namedtuple
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DAE_PATH = PROJECT_ROOT / "assets_src" / "beast_walking_inplace_17k.dae"
OUT_DIR = PROJECT_ROOT / "public" / "models"
OUT_BIN = OUT_DIR / "beast.bin"
OUT_JSON = OUT_DIR / "beast.json"
VALIDATE_SCRIPT = SCRIPT_DIR / "validate_bake.py"

# Facts about this specific DAE, used only as sanity assertions (the parser itself is
# generic and does not hardcode these -- if the source model ever changes and these
# no longer hold, we want a loud, specific failure rather than silently wrong output).
EXPECTED_VERTEX_COUNT = 17365
EXPECTED_JOINT_COUNT = 65
EXPECTED_FRAME_COUNT = 52


# =========================================================================================
# Small flat 4x4 matrix helpers (row-major storage: m[row*4 + col]; standard semantics,
# i.e. matrix-vector product treats the vector as a column: v' = M @ v).
# =========================================================================================

def mat4_identity() -> list[float]:
    return [1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0]


def mat4_mul(a: list[float], b: list[float]) -> list[float]:
    """Standard 4x4 matrix product a @ b (both row-major flat lists)."""
    r = [0.0] * 16
    for i in range(4):
        ai0, ai1, ai2, ai3 = a[i * 4], a[i * 4 + 1], a[i * 4 + 2], a[i * 4 + 3]
        r[i * 4 + 0] = ai0 * b[0] + ai1 * b[4] + ai2 * b[8] + ai3 * b[12]
        r[i * 4 + 1] = ai0 * b[1] + ai1 * b[5] + ai2 * b[9] + ai3 * b[13]
        r[i * 4 + 2] = ai0 * b[2] + ai1 * b[6] + ai2 * b[10] + ai3 * b[14]
        r[i * 4 + 3] = ai0 * b[3] + ai1 * b[7] + ai2 * b[11] + ai3 * b[15]
    return r


def mat4_translate(x: float, y: float, z: float) -> list[float]:
    return [1.0, 0.0, 0.0, x,
            0.0, 1.0, 0.0, y,
            0.0, 0.0, 1.0, z,
            0.0, 0.0, 0.0, 1.0]


def mat4_scale(sx: float, sy: float, sz: float) -> list[float]:
    return [sx, 0.0, 0.0, 0.0,
            0.0, sy, 0.0, 0.0,
            0.0, 0.0, sz, 0.0,
            0.0, 0.0, 0.0, 1.0]


def mat4_rotate_axis_angle_deg(x: float, y: float, z: float, deg: float) -> list[float]:
    """Collada <rotate> semantics: axis (x,y,z) + angle in degrees, Rodrigues formula."""
    n = math.sqrt(x * x + y * y + z * z)
    if n < 1e-12 or deg == 0.0:
        return mat4_identity()
    x, y, z = x / n, y / n, z / n
    a = math.radians(deg)
    c = math.cos(a)
    s = math.sin(a)
    C = 1.0 - c
    return [
        x * x * C + c, x * y * C - z * s, x * z * C + y * s, 0.0,
        y * x * C + z * s, y * y * C + c, y * z * C - x * s, 0.0,
        z * x * C - y * s, z * y * C + x * s, z * z * C + c, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]


def mat4_apply_point(m: list[float], p: tuple[float, float, float]) -> tuple[float, float, float]:
    """Apply m to point p=(x,y,z) (implicit w=1), assuming the last row of m is 0 0 0 1."""
    x, y, z = p
    return (
        m[0] * x + m[1] * y + m[2] * z + m[3],
        m[4] * x + m[5] * y + m[6] * z + m[7],
        m[8] * x + m[9] * y + m[10] * z + m[11],
    )


def mat4_to_col_major(m: list[float]) -> list[float]:
    """Transpose a row-major flat 4x4 into a column-major flat 4x4."""
    return [m[c + r * 4] for c in range(4) for r in range(4)]


# =========================================================================================
# XML plumbing
# =========================================================================================

def strip_namespaces(root: ET.Element) -> None:
    """Collada declares a default xmlns; strip it from every tag so plain tag names
    ('mesh', 'polylist', ...) work with find()/findall() without namespace ceremony."""
    for el in root.iter():
        if "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]


def index_by_id(root: ET.Element) -> dict[str, ET.Element]:
    ids: dict[str, ET.Element] = {}
    for el in root.iter():
        el_id = el.attrib.get("id")
        if el_id is not None:
            ids[el_id] = el
    return ids


def parse_floats(text: str | None) -> list[float]:
    if not text:
        return []
    return [float(tok) for tok in text.split()]


def parse_ints(text: str | None) -> list[int]:
    if not text:
        return []
    return [int(tok) for tok in text.split()]


def resolve_source(ids: dict[str, ET.Element], ref: str) -> ET.Element:
    """Resolve a Collada '#id' URI reference against the id registry."""
    key = ref.lstrip("#")
    if key not in ids:
        raise KeyError(f"Unresolved Collada source reference: {ref!r}")
    return ids[key]


def float_array_of(ids: dict[str, ET.Element], source_id: str) -> list[float]:
    src = resolve_source(ids, source_id)
    arr = src.find("float_array")
    if arr is None:
        raise ValueError(f"<source id={source_id!r}> has no <float_array>")
    return parse_floats(arr.text)


# =========================================================================================
# Geometry: positions, per-vertex averaged normals, triangle indices
# =========================================================================================

class GeometryData:
    __slots__ = ("positions", "normals", "indices", "vertex_count")

    def __init__(self, positions, normals, indices, vertex_count):
        self.positions = positions        # flat list len V*3
        self.normals = normals            # flat list len V*3 (averaged, normalized)
        self.indices = indices            # flat list of ints, len I (triangle list)
        self.vertex_count = vertex_count  # V


def parse_geometry(root: ET.Element, ids: dict[str, ET.Element]) -> GeometryData:
    geometries = root.find("library_geometries")
    geometry = geometries.find("geometry")
    mesh = geometry.find("mesh")

    polylist = mesh.find("polylist")
    if polylist is None:
        raise ValueError("Expected a <polylist> in the mesh (none found)")

    # --- Resolve the VERTEX input indirection: polylist's VERTEX input points at the
    # <vertices> element, which in turn points at the actual POSITION source. ---
    inputs = polylist.findall("input")
    offsets: dict[str, int] = {}
    source_ids: dict[str, str] = {}
    for inp in inputs:
        semantic = inp.attrib["semantic"]
        offsets[semantic] = int(inp.attrib["offset"])
        source_ids[semantic] = inp.attrib["source"]

    if "VERTEX" not in offsets or "NORMAL" not in offsets:
        raise ValueError(f"polylist missing VERTEX/NORMAL input, got semantics={list(offsets)}")

    vertices_elem = resolve_source(ids, source_ids["VERTEX"])
    position_input = vertices_elem.find("input")
    if position_input is None or position_input.attrib.get("semantic") != "POSITION":
        raise ValueError("<vertices> element does not have a POSITION input as expected")
    positions = float_array_of(ids, position_input.attrib["source"])

    normals_pool = float_array_of(ids, source_ids["NORMAL"])

    voff = offsets["VERTEX"]
    noff = offsets["NORMAL"]
    stride = max(offsets.values()) + 1

    vertex_count = len(positions) // 3
    if len(positions) % 3 != 0:
        raise ValueError(f"positions float_array length {len(positions)} not divisible by 3")

    vcount = parse_ints(polylist.find("vcount").text)
    p = parse_ints(polylist.find("p").text)
    poly_count = int(polylist.attrib["count"])
    if len(vcount) != poly_count:
        raise ValueError(f"vcount length {len(vcount)} != polylist count {poly_count}")

    expected_p_len = sum(vcount) * stride
    if len(p) != expected_p_len:
        raise ValueError(f"<p> length {len(p)} != expected {expected_p_len} (sum(vcount)*stride)")

    normal_accum = [[0.0, 0.0, 0.0] for _ in range(vertex_count)]
    normal_hits = [0] * vertex_count
    indices: list[int] = []

    ptr = 0
    degenerate_polys = 0
    for vc in vcount:
        if vc < 3:
            # Degenerate polygon (point/line) -- skip, cannot triangulate.
            ptr += vc * stride
            degenerate_polys += 1
            continue

        corner_v = [0] * vc
        corner_n = [0] * vc
        for c in range(vc):
            base = ptr + c * stride
            vi = p[base + voff]
            ni = p[base + noff]
            corner_v[c] = vi
            corner_n[c] = ni
            nx, ny, nz = (normals_pool[ni * 3], normals_pool[ni * 3 + 1], normals_pool[ni * 3 + 2])
            acc = normal_accum[vi]
            acc[0] += nx
            acc[1] += ny
            acc[2] += nz
            normal_hits[vi] += 1
        ptr += vc * stride

        # Fan-triangulate (handles the general n-gon case; this file's polygons are all
        # already triangles, so each polygon here contributes exactly one triangle).
        for i in range(1, vc - 1):
            indices.append(corner_v[0])
            indices.append(corner_v[i])
            indices.append(corner_v[i + 1])

    if degenerate_polys:
        print(f"  WARNING: skipped {degenerate_polys} degenerate (<3-vertex) polygons", file=sys.stderr)

    normals_out = [0.0] * (vertex_count * 3)
    zero_normal_count = 0
    for vi in range(vertex_count):
        nx, ny, nz = normal_accum[vi]
        length = math.sqrt(nx * nx + ny * ny + nz * nz)
        if length < 1e-12:
            zero_normal_count += 1
            nx, ny, nz = 0.0, 0.0, 1.0
        else:
            nx, ny, nz = nx / length, ny / length, nz / length
        normals_out[vi * 3] = nx
        normals_out[vi * 3 + 1] = ny
        normals_out[vi * 3 + 2] = nz

    if zero_normal_count:
        print(f"  WARNING: {zero_normal_count} vertices had no incident normal "
              f"(never referenced by a polygon corner); defaulted to (0,0,1)", file=sys.stderr)

    return GeometryData(positions=positions, normals=normals_out, indices=indices,
                         vertex_count=vertex_count)


# =========================================================================================
# Skin controller: bind shape matrix, joint names/order, inverse bind matrices, per-vertex
# influence lists (joint index, weight).
# =========================================================================================

class SkinData:
    __slots__ = ("bind_shape_matrix", "joint_names", "inverse_bind", "vertex_influences")

    def __init__(self, bind_shape_matrix, joint_names, inverse_bind, vertex_influences):
        self.bind_shape_matrix = bind_shape_matrix          # 16 floats, row-major
        self.joint_names = joint_names                      # list[str] len J -- palette order
        self.inverse_bind = inverse_bind                    # list[list[float]] len J, each 16 row-major
        self.vertex_influences = vertex_influences           # list[list[(joint_idx, weight)]] len V


def parse_skin(root: ET.Element, ids: dict[str, ET.Element]) -> SkinData:
    controllers = root.find("library_controllers")
    controller = controllers.find("controller")
    skin = controller.find("skin")

    bsm_elem = skin.find("bind_shape_matrix")
    bind_shape_matrix = parse_floats(bsm_elem.text) if bsm_elem is not None else mat4_identity()
    if len(bind_shape_matrix) != 16:
        raise ValueError(f"bind_shape_matrix has {len(bind_shape_matrix)} floats, expected 16")

    joints_elem = skin.find("joints")
    joint_source_id = None
    inv_bind_source_id = None
    for inp in joints_elem.findall("input"):
        if inp.attrib["semantic"] == "JOINT":
            joint_source_id = inp.attrib["source"]
        elif inp.attrib["semantic"] == "INV_BIND_MATRIX":
            inv_bind_source_id = inp.attrib["source"]
    if joint_source_id is None or inv_bind_source_id is None:
        raise ValueError("<joints> missing JOINT or INV_BIND_MATRIX input")

    joint_source = resolve_source(ids, joint_source_id)
    name_array = joint_source.find("Name_array")
    if name_array is not None and name_array.text:
        joint_names = name_array.text.split()
    else:
        id_array = joint_source.find("IDREF_array")
        joint_names = id_array.text.split()
    joint_count = len(joint_names)

    inv_bind_flat = float_array_of(ids, inv_bind_source_id)
    if len(inv_bind_flat) != joint_count * 16:
        raise ValueError(f"inverse bind array has {len(inv_bind_flat)} floats, "
                          f"expected {joint_count * 16} ({joint_count} joints x 16)")
    inverse_bind = [inv_bind_flat[j * 16:(j + 1) * 16] for j in range(joint_count)]

    vw = skin.find("vertex_weights")
    vw_count = int(vw.attrib["count"])
    joff = woff = None
    weight_source_id = None
    for inp in vw.findall("input"):
        if inp.attrib["semantic"] == "JOINT":
            joff = int(inp.attrib["offset"])
        elif inp.attrib["semantic"] == "WEIGHT":
            woff = int(inp.attrib["offset"])
            weight_source_id = inp.attrib["source"]
    if joff is None or woff is None:
        raise ValueError("<vertex_weights> missing JOINT or WEIGHT input")
    vw_stride = max(joff, woff) + 1

    weight_pool = float_array_of(ids, weight_source_id)

    vcount = parse_ints(vw.find("vcount").text)
    v = parse_ints(vw.find("v").text)
    if len(vcount) != vw_count:
        raise ValueError(f"vertex_weights vcount length {len(vcount)} != count {vw_count}")
    expected_v_len = sum(vcount) * vw_stride
    if len(v) != expected_v_len:
        raise ValueError(f"vertex_weights <v> length {len(v)} != expected {expected_v_len}")

    vertex_influences: list[list[tuple[int, float]]] = []
    ptr = 0
    for vc in vcount:
        infl = []
        for _ in range(vc):
            base = ptr
            j_idx = v[base + joff]
            w_idx = v[base + woff]
            infl.append((j_idx, weight_pool[w_idx]))
            ptr += vw_stride
        vertex_influences.append(infl)

    return SkinData(bind_shape_matrix=bind_shape_matrix, joint_names=joint_names,
                     inverse_bind=inverse_bind, vertex_influences=vertex_influences)


# =========================================================================================
# Animations: one baked <animation> per joint, sampled at 52 shared time keys.
# =========================================================================================

class AnimData:
    __slots__ = ("times", "matrices_by_key")

    def __init__(self, times, matrices_by_key):
        self.times = times                    # list[float] len F (seconds)
        self.matrices_by_key = matrices_by_key  # dict[str -> list[list[float]]] (F matrices, row-major)


def parse_animations(root: ET.Element, ids: dict[str, ET.Element]) -> AnimData:
    lib = root.find("library_animations")
    matrices_by_key: dict[str, list[list[float]]] = {}
    shared_times: list[float] | None = None

    for anim in lib.findall("animation"):
        channel = anim.find("channel")
        sampler_id = channel.attrib["source"].lstrip("#")
        target = channel.attrib["target"]
        # Target syntax is "<sid-path>/<member>"; we only baked whole-matrix channels
        # (target "<JointSid>/transform"), so the key is everything before the first '/'.
        key = target.split("/", 1)[0]

        sampler = ids[sampler_id]
        input_source_id = output_source_id = None
        for inp in sampler.findall("input"):
            if inp.attrib["semantic"] == "INPUT":
                input_source_id = inp.attrib["source"]
            elif inp.attrib["semantic"] == "OUTPUT":
                output_source_id = inp.attrib["source"]
        if input_source_id is None or output_source_id is None:
            raise ValueError(f"animation {anim.attrib.get('id')} sampler missing INPUT/OUTPUT")

        times = float_array_of(ids, input_source_id)
        outputs = float_array_of(ids, output_source_id)
        if len(outputs) != len(times) * 16:
            raise ValueError(f"animation {anim.attrib.get('id')}: output length {len(outputs)} "
                              f"!= {len(times)} keys x 16")

        if shared_times is None:
            shared_times = times
        elif times != shared_times:
            # Not fatal (we key everything by explicit frame index, not by re-sampling
            # time), but this would indicate the "52 shared keys" assumption is wrong.
            raise ValueError(f"animation {anim.attrib.get('id')} has different time keys "
                              f"than other joints -- per-joint time resampling is not implemented")

        matrices = [outputs[f * 16:(f + 1) * 16] for f in range(len(times))]
        matrices_by_key[key] = matrices

    if shared_times is None:
        raise ValueError("No <animation> elements found in library_animations")

    return AnimData(times=shared_times, matrices_by_key=matrices_by_key)


# =========================================================================================
# Visual scene hierarchy + per-frame world transforms
# =========================================================================================

def node_key(node: ET.Element) -> str | None:
    """Preferred lookup key for a node: sid, falling back to id, falling back to name --
    matches how the skin's Name_array and the animation channel targets address nodes."""
    return node.attrib.get("sid") or node.attrib.get("id") or node.attrib.get("name")


def static_local_matrix(node: ET.Element) -> list[float]:
    """Compose a node's local matrix from its transform child elements, in document
    order (Collada semantics: earlier elements are applied closer to the node's own
    content, i.e. m = m @ transform for each child in order)."""
    m = mat4_identity()
    for child in node:
        tag = child.tag
        if tag == "matrix":
            vals = parse_floats(child.text)
            if len(vals) != 16:
                raise ValueError(f"<matrix> on node {node.attrib.get('id')} has {len(vals)} floats")
            m = mat4_mul(m, vals)
        elif tag == "translate":
            x, y, z = parse_floats(child.text)
            m = mat4_mul(m, mat4_translate(x, y, z))
        elif tag == "rotate":
            x, y, z, deg = parse_floats(child.text)
            m = mat4_mul(m, mat4_rotate_axis_angle_deg(x, y, z, deg))
        elif tag == "scale":
            x, y, z = parse_floats(child.text)
            m = mat4_mul(m, mat4_scale(x, y, z))
        # Other children (instance_geometry, instance_controller, extra, nested <node>, ...)
        # are not transform contributions; ignore them here.
    return m


def find_scene_root_node(root: ET.Element, node_id: str) -> ET.Element:
    vis = root.find("library_visual_scenes").find("visual_scene")
    for top in vis.findall("node"):
        if top.attrib.get("id") == node_id:
            return top
    raise ValueError(f"No top-level node with id={node_id!r} found in the visual scene")


def compute_world_transforms_for_frame(
    armature_node: ET.Element,
    frame_idx: int,
    anim: AnimData,
) -> dict[str, list[float]]:
    """Walk the hierarchy from the scene root (Armature) down, returning a dict keyed by
    every node's sid/id/name aliases -> that node's WORLD matrix at this frame."""
    out: dict[str, list[float]] = {}

    def recurse(node: ET.Element, parent_world: list[float]) -> None:
        key = node_key(node)
        frames = anim.matrices_by_key.get(key) if key else None
        local = frames[frame_idx] if frames is not None else static_local_matrix(node)
        world = mat4_mul(parent_world, local)
        for alias in (node.attrib.get("sid"), node.attrib.get("id"), node.attrib.get("name")):
            if alias:
                out.setdefault(alias, world)
        for child in node.findall("node"):
            recurse(child, world)

    recurse(armature_node, mat4_identity())
    return out


# =========================================================================================
# Top-4 joint/weight selection (Collada allows arbitrarily many influences per vertex;
# the GPU skinning path is fixed at 4).
# =========================================================================================

def select_top4(vertex_influences: list[list[tuple[int, float]]], joint_count: int
                 ) -> tuple[array.array, array.array]:
    v = len(vertex_influences)
    joints_flat = array.array("H", [0]) * (v * 4)
    weights_flat = array.array("f", [0.0]) * (v * 4)

    max_influences_seen = 0
    fallback_count = 0
    for i, infl in enumerate(vertex_influences):
        max_influences_seen = max(max_influences_seen, len(infl))
        top = sorted(infl, key=lambda jw: jw[1], reverse=True)[:4]
        total = sum(w for _, w in top)
        if total > 1e-12:
            top = [(j, w / total) for j, w in top]
        elif top:
            # All-zero weights (shouldn't happen with real skin data) -- rigidly bind to
            # the first listed joint rather than producing NaNs.
            fallback_count += 1
            top = [(top[0][0], 1.0)] + top[1:]
        else:
            # No influences at all for this vertex -- bind to joint 0 with full weight.
            fallback_count += 1
            top = [(0, 1.0)]

        for k in range(4):
            if k < len(top):
                j, w = top[k]
                if not (0 <= j < joint_count):
                    raise ValueError(f"vertex {i} references joint index {j} >= joint_count {joint_count}")
                joints_flat[i * 4 + k] = j
                weights_flat[i * 4 + k] = w
            else:
                joints_flat[i * 4 + k] = 0
                weights_flat[i * 4 + k] = 0.0

    if fallback_count:
        print(f"  WARNING: {fallback_count} vertices had degenerate (zero/empty) skin "
              f"weights; rigidly bound to a fallback joint", file=sys.stderr)
    print(f"  max influences on any single vertex: {max_influences_seen} (truncated to top 4)")

    return joints_flat, weights_flat


# =========================================================================================
# Palettes: for frame f, joint j -> worldTransform(j,f) x inverseBind(j) x bindShapeMatrix
# =========================================================================================

def compute_palettes(
    armature_node: ET.Element,
    anim: AnimData,
    joint_names: list[str],
    inverse_bind: list[list[float]],
    bind_shape_matrix: list[float],
    frame_count: int,
) -> list[list[list[float]]]:
    """Returns palettes[frame][joint] = 16-float row-major mat4."""
    palettes: list[list[list[float]]] = []
    for f in range(frame_count):
        world_by_key = compute_world_transforms_for_frame(armature_node, f, anim)
        frame_palettes = []
        for j, name in enumerate(joint_names):
            if name not in world_by_key:
                raise ValueError(f"Joint {name!r} (Name_array index {j}) not found in the "
                                  f"visual scene hierarchy under Armature")
            world = world_by_key[name]
            palette = mat4_mul(mat4_mul(world, inverse_bind[j]), bind_shape_matrix)
            frame_palettes.append(palette)
        palettes.append(frame_palettes)
    return palettes


# =========================================================================================
# CPU skinning (used to compute per-frame axis-fixed bounds for modelMatrix normalization)
# =========================================================================================

def cpu_skin_frame(
    positions: list[float],
    joints_flat: array.array,
    weights_flat: array.array,
    palette_frame: list[list[float]],
) -> list[tuple[float, float, float]]:
    vertex_count = len(positions) // 3
    out = [(0.0, 0.0, 0.0)] * vertex_count
    for i in range(vertex_count):
        px, py, pz = positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]
        sx = sy = sz = 0.0
        base = i * 4
        for k in range(4):
            w = weights_flat[base + k]
            if w == 0.0:
                continue
            j = joints_flat[base + k]
            m = palette_frame[j]
            rx = m[0] * px + m[1] * py + m[2] * pz + m[3]
            ry = m[4] * px + m[5] * py + m[6] * pz + m[7]
            rz = m[8] * px + m[9] * py + m[10] * pz + m[11]
            sx += w * rx
            sy += w * ry
            sz += w * rz
        out[i] = (sx, sy, sz)
    return out


# =========================================================================================
# modelMatrix: axis fix, then uniform scale + translate.
#
# AXIS FIX -- despite the DAE's <up_axis> tag reading Z_UP, this content's baked joint/mesh
# data is empirically already Y-up, and the axis fix must be IDENTITY (no rotation).
#
# How we know: with the previous Rx(-90) "Z-up -> Y-up" fix in place, the union-of-all-frames
# axis-fixed bounds came out X[-0.4854,0.5866] Y[-0.6105,0.6524] Z[-1.7150,0.0014] -- i.e. the
# *world Z* axis (horizontal under a Y-up convention) carried the largest extent (1.72), which
# visually reads as the character lying on its side. Undoing that rotation to recover the raw
# per-frame skinned (model-space) bounds gives:
#     model X in [-0.4854, 0.5866]  extent 1.072   (roughly centered on 0 -> left/right)
#     model Y in [-0.0014, 1.7150]  extent 1.716   (starts almost exactly at 0 -> up/down)
#     model Z in [-0.6105, 0.6524]  extent 1.263   (roughly centered on 0 -> front/back)
# Model Y is both the largest extent (consistent with height dominating for an upright
# bipedal walker -- this rig has full per-finger hands, not a quadruped) *and* its minimum
# sits almost exactly at 0, the classic signature of a "feet on the ground" vertical axis in
# rest/bind-ish data. Both signals independently point the same way: the raw DAE data is
# already Y-up, and Blender's Collada exporter must have baked that orientation into the
# joint hierarchy and animation curves directly without actually needing (or the <up_axis>
# tag not accurately reflecting) a separate Z-up-to-Y-up conversion. So: model axes already
# equal world axes here -- no rotation, and definitely no reason to introduce a mirror.
# =========================================================================================

AXIS_FIX = mat4_identity()


def _mat3_determinant(m: list[float]) -> float:
    a, b, c = m[0], m[1], m[2]
    d, e, f = m[4], m[5], m[6]
    g, h, i = m[8], m[9], m[10]
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


# Handedness sanity check: AXIS_FIX must be a proper (non-mirroring) transform. For the
# current identity this is trivially true, but this guards against a future edit here
# accidentally introducing a reflection (determinant -1) instead of a rotation (+1).
_axis_fix_det = _mat3_determinant(AXIS_FIX)
assert abs(_axis_fix_det - 1.0) < 1e-9, (
    f"AXIS_FIX has determinant {_axis_fix_det:.6f}, expected +1 (a mirror would flip winding "
    f"and turn the mesh inside out from the renderer's point of view)")


# IMPORTANT (unrelated to the axis fix above): the scale + translate normalization is
# derived from the UNION of bounds across ALL frames of the animation, not from frame 0
# alone. An earlier version of this script normalized against frame 0 only; the engine team
# found that mid-stride frames then sink up to ~3 world units below the floor (frame 0
# happens to be a relatively "tall, neutral" pose in this walk cycle, not representative of
# the full cycle's Y extent). Using the union means: every frame's lowest point is
# guaranteed >= y=0, the tallest point across the whole cycle is exactly y=4.5, and only the
# frame(s) that touch those extremes will exactly reach 0 / 4.5 -- other frames legitimately
# float above the floor and top out below 4.5 (e.g. a crouch phase shouldn't be stretched to
# full height).

# Axis-fixed (but not yet scaled/translated) bounding box of one frame's skinned mesh.
Bounds = namedtuple("Bounds", "min_x max_x min_y max_y min_z max_z")


def frame_bounds_axis_fixed(
    positions: list[float],
    joints_flat: array.array,
    weights_flat: array.array,
    palette_frame: list[list[float]],
    frame_label: int | str = "",
) -> Bounds:
    """CPU-skin one frame, apply the axis fix, and return its AABB -- without retaining the
    full per-vertex array, so this is cheap to call once per animation frame."""
    skinned = cpu_skin_frame(positions, joints_flat, weights_flat, palette_frame)
    fixed = [mat4_apply_point(AXIS_FIX, p) for p in skinned]
    xs = [p[0] for p in fixed]
    ys = [p[1] for p in fixed]
    zs = [p[2] for p in fixed]
    if not (all(math.isfinite(v) for v in xs)
            and all(math.isfinite(v) for v in ys)
            and all(math.isfinite(v) for v in zs)):
        raise ValueError(f"Non-finite skinned coordinate in frame {frame_label!r}")
    return Bounds(min(xs), max(xs), min(ys), max(ys), min(zs), max(zs))


def compute_scale_translate(bounds: Bounds) -> tuple[float, float, float, float]:
    """Given an axis-fixed AABB, return (scale, tx, ty, tz) such that applying scale then
    this translation maps min_y -> 0, height -> 4.5, and the x/z center -> the origin."""
    height = bounds.max_y - bounds.min_y
    if height <= 1e-9:
        raise ValueError(f"Degenerate Y bounds: extent = {height}")
    scale = 4.5 / height
    tx = -((bounds.min_x + bounds.max_x) / 2.0) * scale
    ty = -bounds.min_y * scale
    tz = -((bounds.min_z + bounds.max_z) / 2.0) * scale
    return scale, tx, ty, tz


def build_model_matrix(scale: float, tx: float, ty: float, tz: float) -> list[float]:
    """Row-major modelMatrix = Translate(tx,ty,tz) @ Scale(scale) @ AxisFix."""
    return mat4_mul(mat4_mul(mat4_translate(tx, ty, tz), mat4_scale(scale, scale, scale)), AXIS_FIX)


def world_y_bounds(bounds: Bounds, scale: float, ty: float) -> tuple[float, float]:
    """Axis-fixed bounds scale/translate independently per axis (axisFix is a pure
    permutation+sign-flip, scale/translate are diagonal), so world Y bounds can be derived
    directly from the axis-fixed Y bounds without re-skinning or re-transforming vertices."""
    return bounds.min_y * scale + ty, bounds.max_y * scale + ty


def world_bounds_full(bounds: Bounds, scale: float, tx: float, ty: float, tz: float) -> Bounds:
    """Same idea as world_y_bounds() but all three axes, for reporting proportions."""
    return Bounds(
        min_x=bounds.min_x * scale + tx, max_x=bounds.max_x * scale + tx,
        min_y=bounds.min_y * scale + ty, max_y=bounds.max_y * scale + ty,
        min_z=bounds.min_z * scale + tz, max_z=bounds.max_z * scale + tz,
    )


# =========================================================================================
# Binary packing
# =========================================================================================

assert array.array("f").itemsize == 4, "platform float itemsize is not 4 bytes"
assert array.array("I").itemsize == 4, "platform unsigned-int itemsize is not 4 bytes"
assert array.array("H").itemsize == 2, "platform unsigned-short itemsize is not 2 bytes"


def _to_little_endian_bytes(arr: array.array) -> bytes:
    if sys.byteorder != "little":
        arr = array.array(arr.typecode, arr)
        arr.byteswap()
    return arr.tobytes()


def pack_f32(values) -> bytes:
    return _to_little_endian_bytes(array.array("f", values))


def pack_u32(values) -> bytes:
    return _to_little_endian_bytes(array.array("I", values))


def pack_u16(values) -> bytes:
    return _to_little_endian_bytes(array.array("H", values))


def write_sections(path: Path, sections: list[tuple[str, bytes]]) -> dict[str, dict[str, int]]:
    """Writes sections back-to-back with 4-byte-aligned start offsets; returns the
    {name: {offset, length}} layout dict for the JSON manifest."""
    offset = 0
    layout: dict[str, dict[str, int]] = {}
    chunks: list[bytes] = []
    for name, data in sections:
        pad = (-offset) % 4
        if pad:
            chunks.append(b"\x00" * pad)
            offset += pad
        layout[name] = {"offset": offset, "length": len(data)}
        chunks.append(data)
        offset += len(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        for c in chunks:
            f.write(c)
    return layout


# =========================================================================================
# Main
# =========================================================================================

def main() -> None:
    t_start = time.time()

    if not DAE_PATH.exists():
        print(f"ERROR: input DAE not found: {DAE_PATH}", file=sys.stderr)
        sys.exit(1)

    print(f"Parsing {DAE_PATH} ({DAE_PATH.stat().st_size / 1e6:.2f} MB)...")
    tree = ET.parse(DAE_PATH)
    root = tree.getroot()
    strip_namespaces(root)
    ids = index_by_id(root)

    asset = root.find("asset")
    up_axis = asset.findtext("up_axis", default="Y_UP")
    print(f"  up_axis={up_axis} (informational only -- NOT used to choose the axis fix; "
          f"see the AXIS_FIX comment above main(): this content is empirically Y-up in its "
          f"baked joint/mesh data regardless of what this tag says, so AXIS_FIX is identity)")

    print("Parsing geometry (positions/normals/indices)...")
    geo = parse_geometry(root, ids)
    print(f"  vertexCount={geo.vertex_count}  indexCount={len(geo.indices)}")

    print("Parsing skin controller (bind shape, joints, inverse binds, vertex weights)...")
    skin = parse_skin(root, ids)
    print(f"  jointCount={len(skin.joint_names)}  bind_shape_matrix={skin.bind_shape_matrix}")

    print("Parsing library_animations (baked per-joint matrix tracks)...")
    anim = parse_animations(root, ids)
    frame_count = len(anim.times)
    duration = anim.times[-1] - anim.times[0]
    print(f"  frameCount={frame_count}  duration={duration}")

    print("Parsing visual scene hierarchy and locating the Armature root...")
    armature_node = find_scene_root_node(root, "Armature")

    # --- Sanity checks against the facts this script was written against. A mismatch
    # here doesn't necessarily mean the parser is wrong, but it's worth a loud heads-up
    # since several validation thresholds below assume these specific values. ---
    if geo.vertex_count != EXPECTED_VERTEX_COUNT:
        print(f"  NOTE: vertexCount {geo.vertex_count} != expected {EXPECTED_VERTEX_COUNT}", file=sys.stderr)
    if len(skin.joint_names) != EXPECTED_JOINT_COUNT:
        print(f"  NOTE: jointCount {len(skin.joint_names)} != expected {EXPECTED_JOINT_COUNT}", file=sys.stderr)
    if frame_count != EXPECTED_FRAME_COUNT:
        print(f"  NOTE: frameCount {frame_count} != expected {EXPECTED_FRAME_COUNT}", file=sys.stderr)
    if len(skin.vertex_influences) != geo.vertex_count:
        raise ValueError(f"vertex_weights count {len(skin.vertex_influences)} != "
                          f"positions vertex count {geo.vertex_count}")

    print("Selecting top-4 joints/weights per vertex...")
    joints_flat, weights_flat = select_top4(skin.vertex_influences, len(skin.joint_names))

    print(f"Computing palettes for {frame_count} frames x {len(skin.joint_names)} joints...")
    palettes = compute_palettes(armature_node, anim, skin.joint_names, skin.inverse_bind,
                                 skin.bind_shape_matrix, frame_count)

    print(f"CPU-skinning all {frame_count} frames to compute axis-fixed bounds "
          f"(needed for modelMatrix normalization across the whole cycle, not just frame 0)...")
    t_skin = time.time()
    per_frame_bounds: list[Bounds] = [
        frame_bounds_axis_fixed(geo.positions, joints_flat, weights_flat, palettes[f], frame_label=f)
        for f in range(frame_count)
    ]
    print(f"  done in {time.time() - t_skin:.2f}s")

    # --- Diagnostic: what the OLD frame-0-only normalization would have produced, per
    # frame. This is what the engine team observed sinking below the floor. ---
    old_scale, old_tx, old_ty, old_tz = compute_scale_translate(per_frame_bounds[0])
    print(f"\n  [diagnostic] OLD (frame-0-only) normalization: scale={old_scale:.6f} "
          f"translate=({old_tx:.4f},{old_ty:.4f},{old_tz:.4f})")
    print("  [diagnostic] per-frame world Y bounds under the OLD frame-0-only modelMatrix:")
    print(f"  {'frame':>5} {'minY':>9} {'maxY':>9}")
    worst_frame, worst_min_y = 0, math.inf
    sinking_frames = 0
    for f, b in enumerate(per_frame_bounds):
        wy0, wy1 = world_y_bounds(b, old_scale, old_ty)
        if wy0 < worst_min_y:
            worst_frame, worst_min_y = f, wy0
        if wy0 < -0.01:
            sinking_frames += 1
        flag = "  <-- SINKS BELOW FLOOR" if wy0 < -0.01 else ""
        print(f"  {f:5d} {wy0:9.4f} {wy1:9.4f}{flag}")
    print(f"  [diagnostic] {sinking_frames}/{frame_count} frames sink below the floor under "
          f"the OLD normalization; worst is frame {worst_frame} at minY={worst_min_y:.4f}")

    # --- Fix: normalize against the UNION of axis-fixed bounds across all frames, so no
    # frame's lowest point can ever end up below y=0 and the cycle's tallest point is
    # exactly y=4.5 (rather than forcing every individual frame to span the full height). ---
    union_bounds = Bounds(
        min_x=min(b.min_x for b in per_frame_bounds),
        max_x=max(b.max_x for b in per_frame_bounds),
        min_y=min(b.min_y for b in per_frame_bounds),
        max_y=max(b.max_y for b in per_frame_bounds),
        min_z=min(b.min_z for b in per_frame_bounds),
        max_z=max(b.max_z for b in per_frame_bounds),
    )
    new_scale, new_tx, new_ty, new_tz = compute_scale_translate(union_bounds)
    ux_extent = union_bounds.max_x - union_bounds.min_x
    uy_extent = union_bounds.max_y - union_bounds.min_y
    uz_extent = union_bounds.max_z - union_bounds.min_z
    print(f"\n  union ({frame_count}-frame) axis-fixed bounds (pre-scale), per axis:")
    print(f"    X[{union_bounds.min_x:8.4f},{union_bounds.max_x:8.4f}]  extent={ux_extent:.4f}")
    print(f"    Y[{union_bounds.min_y:8.4f},{union_bounds.max_y:8.4f}]  extent={uy_extent:.4f}  (up axis, drives scale)")
    print(f"    Z[{union_bounds.min_z:8.4f},{union_bounds.max_z:8.4f}]  extent={uz_extent:.4f}")
    print(f"  NEW modelMatrix (union-normalized): scale={new_scale:.6f} "
          f"translate=({new_tx:.4f},{new_ty:.4f},{new_tz:.4f})")

    frame0_world = world_bounds_full(per_frame_bounds[0], new_scale, new_tx, new_ty, new_tz)
    print(f"\n  frame-0 WORLD bounds (after full modelMatrix) -- expect height~4.5, "
          f"width/depth smaller:")
    print(f"    X[{frame0_world.min_x:8.4f},{frame0_world.max_x:8.4f}]  "
          f"width  ={frame0_world.max_x - frame0_world.min_x:.4f}")
    print(f"    Y[{frame0_world.min_y:8.4f},{frame0_world.max_y:8.4f}]  "
          f"height ={frame0_world.max_y - frame0_world.min_y:.4f}")
    print(f"    Z[{frame0_world.min_z:8.4f},{frame0_world.max_z:8.4f}]  "
          f"depth  ={frame0_world.max_z - frame0_world.min_z:.4f}")

    # --- Facing direction: report only, do not act on it (no rotation is applied here).
    # Front-back is whichever horizontal axis has the ~1.26-unit extent -- with AXIS_FIX now
    # identity, that's model/world Z. Two independent signals, both weak/near-symmetric (as
    # expected for an in-place walk cycle, where limb swing is roughly symmetric fore/aft): ---
    z_center = (union_bounds.min_z + union_bounds.max_z) / 2.0
    print(f"\n  [facing] front-back axis = world Z (union extent {uz_extent:.4f} matches the "
          f"'~1.26' axis). Union Z center offset from 0: {z_center:+.4f} "
          f"({abs(z_center) / uz_extent * 100:.1f}% of the extent -- ", end="")
    print("too small to call a direction from this alone)" if abs(z_center) < 0.05 * uz_extent
          else f"leans toward {'+Z' if z_center > 0 else '-Z'})")

    world0 = compute_world_transforms_for_frame(armature_node, 0, anim)
    hips_key = next((n for n in ("BeastBaseMesh_Hips",) if n in world0), None)
    head_key = next((n for n in ("BeastBaseMesh_Head", "BeastBaseMesh_HeadTop_End") if n in world0), None)
    if hips_key and head_key:
        hips_z = world0[hips_key][11]  # row-major translation.z
        head_z = world0[head_key][11]
        dz = head_z - hips_z
        print(f"  [facing] secondary signal, frame 0: {head_key}.z - {hips_key}.z = {dz:+.4f} "
              f"({'head sits toward +Z of hips' if dz > 0.001 else 'head sits toward -Z of hips' if dz < -0.001 else 'head directly above hips in Z'})")
    else:
        print("  [facing] secondary signal skipped: Hips/Head joint not found by expected name")

    print("\n  per-frame world Y bounds under the NEW union-normalized modelMatrix:")
    print(f"  {'frame':>5} {'minY':>9} {'maxY':>9}")
    for f, b in enumerate(per_frame_bounds):
        wy0, wy1 = world_y_bounds(b, new_scale, new_ty)
        print(f"  {f:5d} {wy0:9.4f} {wy1:9.4f}")

    model_matrix_row_major = build_model_matrix(new_scale, new_tx, new_ty, new_tz)
    model_matrix_col_major = mat4_to_col_major(model_matrix_row_major)

    # --- Pack binary sections ---
    print("Packing binary buffers...")
    positions_bytes = pack_f32(geo.positions)
    normals_bytes = pack_f32(geo.normals)
    indices_bytes = pack_u32(geo.indices)
    joints_bytes = pack_u16(joints_flat)
    weights_bytes = pack_f32(weights_flat)

    palettes_flat: list[float] = []
    for f in range(frame_count):
        for j in range(len(skin.joint_names)):
            palettes_flat.extend(mat4_to_col_major(palettes[f][j]))
    palettes_bytes = pack_f32(palettes_flat)

    layout = write_sections(OUT_BIN, [
        ("positions", positions_bytes),
        ("normals", normals_bytes),
        ("indices", indices_bytes),
        ("joints", joints_bytes),
        ("weights", weights_bytes),
        ("palettes", palettes_bytes),
    ])

    manifest = {
        "version": 1,
        "vertexCount": geo.vertex_count,
        "indexCount": len(geo.indices),
        "jointCount": len(skin.joint_names),
        "frameCount": frame_count,
        "duration": duration,
        "modelMatrix": model_matrix_col_major,
        "buffers": layout,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(manifest, f, indent=2)

    bin_size = OUT_BIN.stat().st_size
    json_size = OUT_JSON.stat().st_size
    print(f"Wrote {OUT_BIN} ({bin_size:,} bytes = {bin_size / 1e6:.3f} MB)")
    print(f"Wrote {OUT_JSON} ({json_size:,} bytes)")
    print(f"Total: {(bin_size + json_size) / 1e6:.3f} MB")
    print(f"Baking done in {time.time() - t_start:.2f}s")

    # --- Run validation as a fresh subprocess so it re-reads the artifacts from disk.
    # Flush first: stdout is block-buffered when piped, and the child inherits the same
    # fd and writes to it directly, so without this its output can appear to precede
    # ours even though it necessarily runs after. ---
    print("\nRunning validation (tools/validate_bake.py)...")
    sys.stdout.flush()
    result = subprocess.run([sys.executable, str(VALIDATE_SCRIPT)], cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print("\nVALIDATION FAILED", file=sys.stderr)
        sys.exit(result.returncode)
    print("\nbake_dae.py: BAKE + VALIDATION PASSED")


if __name__ == "__main__":
    main()
