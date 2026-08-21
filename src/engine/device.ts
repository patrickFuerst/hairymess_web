// WebGPU bootstrap + shader module creation with compilation diagnostics.

export interface GpuContext {
  adapter: GPUAdapter;
  device: GPUDevice;
  context: GPUCanvasContext;
  format: GPUTextureFormat;
  hasTimestamps: boolean;
  /** whether the ribbon path may bind the particle buffer as vertex-stage storage */
  canPullInVertexStage: boolean;
}

/**
 * WebGPU defaults `maxStorageBuffersInVertexStage` to 0, so vertex pulling has to be
 * asked for. Older implementations predate the per-stage limit entirely and simply
 * allow it under `maxStorageBuffersPerShaderStage`; a `requiredLimits` key an adapter
 * does not know is a hard error, so it is only requested when the adapter reports it.
 */
function vertexStageStorageSupport(adapter: GPUAdapter): {
  supported: boolean;
  requiredLimits: Record<string, number>;
} {
  const limits = adapter.limits as unknown as Record<string, number | undefined>;
  const reported = limits.maxStorageBuffersInVertexStage;
  if (reported === undefined) return { supported: true, requiredLimits: {} };
  if (reported < 1) return { supported: false, requiredLimits: {} };
  return { supported: true, requiredLimits: { maxStorageBuffersInVertexStage: 1 } };
}

export class WebGpuUnavailableError extends Error {}

export async function initWebGPU(canvas: HTMLCanvasElement): Promise<GpuContext> {
  if (!('gpu' in navigator) || !navigator.gpu) {
    throw new WebGpuUnavailableError(
      'navigator.gpu is not available. This demo needs WebGPU — try Chrome/Edge 113+, Safari 26+ or Firefox 141+.'
    );
  }

  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) {
    throw new WebGpuUnavailableError(
      'No suitable GPU adapter found. WebGPU may be disabled or blocklisted on this machine.'
    );
  }

  const hasTimestamps = adapter.features.has('timestamp-query');
  const requiredFeatures: GPUFeatureName[] = hasTimestamps ? ['timestamp-query'] : [];

  const vertexStorage = vertexStageStorageSupport(adapter);
  let device: GPUDevice;
  try {
    device = await adapter.requestDevice({
      requiredFeatures,
      requiredLimits: vertexStorage.requiredLimits,
    });
  } catch (err) {
    // an implementation that refuses the raised limit still gets the line hair path
    console.warn('Falling back to default limits (no vertex-stage storage buffers).', err);
    vertexStorage.supported = false;
    device = await adapter.requestDevice({ requiredFeatures });
  }
  const deviceLimits = device.limits as unknown as Record<string, number | undefined>;
  const grantedVertexStorage = deviceLimits.maxStorageBuffersInVertexStage;
  const canPullInVertexStage =
    vertexStorage.supported && (grantedVertexStorage === undefined || grantedVertexStorage >= 1);

  const context = canvas.getContext('webgpu');
  if (!context) {
    throw new WebGpuUnavailableError('Could not create a WebGPU canvas context.');
  }

  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: 'opaque' });

  return { adapter, device, context, format, hasTimestamps, canPullInVertexStage };
}

/**
 * Creates a shader module and reports WGSL diagnostics. WGSL is only validated at
 * runtime, so surface every message loudly instead of failing silently later.
 */
export function createShaderModule(
  device: GPUDevice,
  name: string,
  code: string,
  onError?: (message: string) => void
): GPUShaderModule {
  const module = device.createShaderModule({ code, label: `${name}.wgsl` });

  void module.getCompilationInfo().then((info) => {
    let hadError = false;
    for (const msg of info.messages) {
      const where = `${name}.wgsl:${msg.lineNum}:${msg.linePos}`;
      const text = `[WGSL ${msg.type}] ${where} ${msg.message}`;
      if (msg.type === 'error') {
        hadError = true;
        console.error(text);
      } else if (msg.type === 'warning') {
        console.warn(text);
      } else {
        console.info(text);
      }
    }
    if (hadError && onError) onError(`WGSL compilation failed in ${name}.wgsl — see console.`);
  });

  return module;
}
