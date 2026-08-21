// Mouse-hover fur brushing.
//
// A hovering mouse (no buttons down — dragging still orbits, and touch is untouched)
// casts a ray through a cheap per-model proxy volume. Where it lands, the simulation
// gets one extra degenerate capsule collider that carries a velocity, so the fur is
// both pushed aside and dragged along with the pointer.

import { mat4, type Mat4 } from 'wgpu-matrix';
import type { OrbitCamera } from './camera';
import type { BrushCollider } from './hairSim';
import type { HairModel } from './model';

/** radius of the brush capsule itself */
const BRUSH_RADIUS = 0.6;
/** how far outside the proxy surface a near miss still counts as a brush */
const BRUSH_REACH = 1.5;
/** the pointer can only drag the fur this fast, however wildly it is waved */
const MAX_BRUSH_SPEED = 40;
/** a pointer that has not moved for this long stops brushing */
const IDLE_TIMEOUT_MS = 500;

type Vec3 = [number, number, number];

/** A capsule (a segment plus a radius) standing in for the model's silhouette. */
interface Proxy {
  a: Vec3;
  b: Vec3;
  radius: number;
}

function sub(a: Vec3, b: Vec3): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}
function dot(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
function length(a: Vec3): number {
  return Math.hypot(a[0], a[1], a[2]);
}

/**
 * The proxy the pointer ray is tested against. Deliberately crude — it only has to be
 * roughly the size of the furry silhouette, not the actual body.
 */
function proxyFor(model: HairModel): Proxy {
  if (model.name === 'sphere') {
    const t = model.simTranslation;
    return { a: [t[0], t[1], t[2]], b: [t[0], t[1], t[2]], radius: 4.5 };
  }
  return { a: [0, 0.5, 0], b: [0, 4.0, 0], radius: 2.2 };
}

/** Closest point to `p` on the segment a..b — the CPU twin of the WGSL helper. */
function closestPointOnSegment(a: Vec3, b: Vec3, p: Vec3): Vec3 {
  const ab = sub(b, a);
  const denominator = dot(ab, ab);
  if (denominator < 1e-12) return [a[0], a[1], a[2]];
  const s = Math.min(1, Math.max(0, dot(sub(p, a), ab) / denominator));
  return [a[0] + ab[0] * s, a[1] + ab[1] * s, a[2] + ab[2] * s];
}

/**
 * Closest approach between the ray `o + t*d` (t >= 0, |d| = 1) and the segment a..b.
 * Minimising |o + t d - a - s ab|^2 gives s = (E - D B) / (C - B^2), t = s B - D with
 * B = d.ab, C = ab.ab, D = d.(o-a), E = ab.(o-a); s is clamped to the segment and t
 * re-derived from it, then clamped in front of the eye.
 */
function closestRayToSegment(
  o: Vec3,
  d: Vec3,
  a: Vec3,
  b: Vec3
): { t: number; onAxis: Vec3; distance: number } {
  const ab = sub(b, a);
  const ao = sub(o, a);
  const abab = dot(ab, ab);
  const abd = dot(ab, d);
  const abao = dot(ab, ao);
  const dao = dot(d, ao);
  const denominator = abab - abd * abd;

  let s = 0;
  if (abab >= 1e-9 && Math.abs(denominator) >= 1e-9) {
    s = Math.min(1, Math.max(0, (abao - dao * abd) / denominator));
  }
  const t = Math.max(0, s * abd - dao);
  const onAxis: Vec3 = [a[0] + ab[0] * s, a[1] + ab[1] * s, a[2] + ab[2] * s];
  const point: Vec3 = [o[0] + d[0] * t, o[1] + d[1] * t, o[2] + d[2] * t];
  return { t, onAxis, distance: length(sub(point, onAxis)) };
}

export class PointerBrush {
  private readonly canvas: HTMLCanvasElement;
  private clientX = 0;
  private clientY = 0;
  private active = false;
  private lastMoveMs = 0;
  private previous: Vec3 | null = null;
  private current: BrushCollider | null = null;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    canvas.addEventListener('pointermove', this.onPointerMove);
    canvas.addEventListener('pointerdown', this.stop);
    canvas.addEventListener('pointerleave', this.stop);
    canvas.addEventListener('pointercancel', this.stop);
  }

  private onPointerMove = (event: PointerEvent): void => {
    // touch and pen keep their orbit behaviour; a held button is a camera drag
    if (event.pointerType !== 'mouse' || event.buttons !== 0) {
      this.stop();
      return;
    }
    this.clientX = event.clientX;
    this.clientY = event.clientY;
    this.lastMoveMs = performance.now();
    this.active = true;
  };

  private stop = (): void => {
    this.active = false;
    this.previous = null;
    this.current = null;
  };

  /** Call once per frame, before HairSim.update(). Returns the frame's brush, if any. */
  update(camera: OrbitCamera, model: HairModel, aspect: number, frameDt: number): BrushCollider | null {
    if (!this.active || performance.now() - this.lastMoveMs > IDLE_TIMEOUT_MS) {
      this.stop();
      return null;
    }

    const rect = this.canvas.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return null;
    const ndcX = ((this.clientX - rect.left) / rect.width) * 2 - 1;
    const ndcY = 1 - ((this.clientY - rect.top) / rect.height) * 2;

    const inverse = mat4.inverse(camera.viewProjectionMatrix(aspect) as Mat4) as Float32Array;
    const far = unproject(inverse, ndcX, ndcY, 1);
    if (!far) return null;
    const eye = camera.position;
    const origin: Vec3 = [eye[0], eye[1], eye[2]];
    const direction = sub(far, origin);
    const dirLength = length(direction);
    if (dirLength < 1e-6) return null;
    const d: Vec3 = [direction[0] / dirLength, direction[1] / dirLength, direction[2] / dirLength];

    const proxy = proxyFor(model);
    const near = closestRayToSegment(origin, d, proxy.a, proxy.b);
    if (near.distance > proxy.radius + BRUSH_REACH) {
      this.previous = null;
      this.current = null;
      return null;
    }

    // Step back to where the ray would enter the proxy (exact for a sphere, close
    // enough for a capsule), then snap that point onto the surface so the brush always
    // sits in the fur layer rather than inside the body.
    const inset = Math.sqrt(Math.max(proxy.radius * proxy.radius - near.distance * near.distance, 0));
    const t = Math.max(0, near.t - inset);
    const entry: Vec3 = [origin[0] + d[0] * t, origin[1] + d[1] * t, origin[2] + d[2] * t];
    const surface = closestPointOnSegment(proxy.a, proxy.b, entry);
    let out = sub(entry, surface);
    const outLength = length(out);
    out = outLength > 1e-6
      ? [out[0] / outLength, out[1] / outLength, out[2] / outLength]
      : [0, 1, 0];
    const hit: Vec3 = [
      surface[0] + out[0] * proxy.radius,
      surface[1] + out[1] * proxy.radius,
      surface[2] + out[2] * proxy.radius,
    ];

    let velocity: Vec3 = [0, 0, 0];
    if (this.previous && frameDt > 1e-5) {
      velocity = [
        (hit[0] - this.previous[0]) / frameDt,
        (hit[1] - this.previous[1]) / frameDt,
        (hit[2] - this.previous[2]) / frameDt,
      ];
      const speed = length(velocity);
      if (speed > MAX_BRUSH_SPEED) {
        const k = MAX_BRUSH_SPEED / speed;
        velocity = [velocity[0] * k, velocity[1] * k, velocity[2] * k];
      }
    }
    this.previous = hit;

    this.current = {
      x: hit[0],
      y: hit[1],
      z: hit[2],
      radius: BRUSH_RADIUS,
      vx: velocity[0],
      vy: velocity[1],
      vz: velocity[2],
    };
    return this.current;
  }
}

/** column-major inverse-viewProj applied to an NDC point, with the perspective divide */
function unproject(inverse: Float32Array, x: number, y: number, z: number): Vec3 | null {
  const w = inverse[3] * x + inverse[7] * y + inverse[11] * z + inverse[15];
  if (Math.abs(w) < 1e-9) return null;
  return [
    (inverse[0] * x + inverse[4] * y + inverse[8] * z + inverse[12]) / w,
    (inverse[1] * x + inverse[5] * y + inverse[9] * z + inverse[13]) / w,
    (inverse[2] * x + inverse[6] * y + inverse[10] * z + inverse[14]) / w,
  ];
}
