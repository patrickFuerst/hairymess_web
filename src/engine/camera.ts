import { mat4, vec3, type Mat4 } from 'wgpu-matrix';

// Orbit camera in the spirit of ofEasyCam: drag = orbit, wheel = dolly,
// shift/right/middle drag = pan. Touch: 1 finger orbit, 2 finger pinch/pan.
export class OrbitCamera {
  target = vec3.fromValues(0, 0, 0);
  radius: number;
  theta: number; // azimuth, radians
  phi: number; // polar angle from +Y, radians

  fovYDeg = 60;
  near = 0.1;
  far = 10000;

  private canvas: HTMLCanvasElement;
  private pointers = new Map<number, { x: number; y: number }>();
  private lastPinchDist = 0;

  constructor(canvas: HTMLCanvasElement, position: [number, number, number] = [10, 15, 10]) {
    this.canvas = canvas;
    const p = vec3.fromValues(...position);
    const d = vec3.sub(p, this.target);
    this.radius = vec3.len(d);
    this.theta = Math.atan2(d[0], d[2]);
    this.phi = Math.acos(Math.min(1, Math.max(-1, d[1] / this.radius)));

    canvas.addEventListener('pointerdown', this.onPointerDown);
    canvas.addEventListener('pointermove', this.onPointerMove);
    canvas.addEventListener('pointerup', this.onPointerUp);
    canvas.addEventListener('pointercancel', this.onPointerUp);
    canvas.addEventListener('wheel', this.onWheel, { passive: false });
    canvas.addEventListener('contextmenu', (e) => e.preventDefault());
  }

  get position(): Float32Array {
    const sinPhi = Math.sin(this.phi);
    return new Float32Array([
      this.target[0] + this.radius * sinPhi * Math.sin(this.theta),
      this.target[1] + this.radius * Math.cos(this.phi),
      this.target[2] + this.radius * sinPhi * Math.cos(this.theta),
    ]);
  }

  viewMatrix(): Mat4 {
    return mat4.lookAt(this.position, this.target, vec3.fromValues(0, 1, 0));
  }

  projectionMatrix(aspect: number): Mat4 {
    return mat4.perspective((this.fovYDeg * Math.PI) / 180, aspect, this.near, this.far);
  }

  viewProjectionMatrix(aspect: number): Mat4 {
    return mat4.multiply(this.projectionMatrix(aspect), this.viewMatrix());
  }

  private onPointerDown = (e: PointerEvent) => {
    this.canvas.setPointerCapture(e.pointerId);
    this.pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (this.pointers.size === 2) {
      const [a, b] = [...this.pointers.values()];
      this.lastPinchDist = Math.hypot(a.x - b.x, a.y - b.y);
    }
  };

  private onPointerMove = (e: PointerEvent) => {
    const prev = this.pointers.get(e.pointerId);
    if (!prev) return;
    const dx = e.clientX - prev.x;
    const dy = e.clientY - prev.y;
    this.pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });

    if (this.pointers.size === 2) {
      // pinch zoom + two-finger pan
      const [a, b] = [...this.pointers.values()];
      const dist = Math.hypot(a.x - b.x, a.y - b.y);
      if (this.lastPinchDist > 0) {
        this.radius *= this.lastPinchDist / dist;
        this.radius = Math.min(500, Math.max(0.5, this.radius));
      }
      this.lastPinchDist = dist;
      this.pan(dx * 0.5, dy * 0.5);
      return;
    }

    const panning = e.shiftKey || e.buttons === 2 || e.buttons === 4;
    if (panning) {
      this.pan(dx, dy);
    } else {
      this.theta -= dx * 0.008;
      this.phi -= dy * 0.008;
      this.phi = Math.min(Math.PI - 0.01, Math.max(0.01, this.phi));
    }
  };

  private onPointerUp = (e: PointerEvent) => {
    this.pointers.delete(e.pointerId);
    this.lastPinchDist = 0;
  };

  private onWheel = (e: WheelEvent) => {
    e.preventDefault();
    this.radius *= Math.exp(e.deltaY * 0.001);
    this.radius = Math.min(500, Math.max(0.5, this.radius));
  };

  private pan(dxPx: number, dyPx: number) {
    // move target in the camera's view plane, scaled with distance
    const scale = (this.radius * Math.tan(((this.fovYDeg / 2) * Math.PI) / 180) * 2) /
      this.canvas.clientHeight;
    const eye = this.position;
    const forward = vec3.normalize(vec3.sub(this.target, eye));
    const right = vec3.normalize(vec3.cross(forward, vec3.fromValues(0, 1, 0)));
    const up = vec3.cross(right, forward);
    vec3.addScaled(this.target, right, -dxPx * scale, this.target);
    vec3.addScaled(this.target, up, dyPx * scale, this.target);
  }
}
