import GUI from 'lil-gui';
import type { SimParams } from '../params';

export type ModelName = 'beast' | 'sphere';

export interface GuiHooks {
  /** which model is loaded right now — the dropdown's initial value */
  model: ModelName;
  /** switching models reloads the page with a new ?model= (simplest full reinit) */
  onModelChange: (model: ModelName) => void;
  /** re-initialises the particle buffer with the new colour scheme */
  onColorModeChange: () => void;
  /** any other parameter changed */
  onChange?: () => void;
}

// Control panel mirroring the original ofxGui setup (ofApp::createGui).
export function createGui(params: SimParams, hooks?: GuiHooks): GUI {
  const gui = new GUI({ title: 'hairy mess' });

  if (hooks) {
    const scene = { model: hooks.model };
    gui
      .add(scene, 'model', ['beast', 'sphere'] satisfies ModelName[])
      .name('model')
      .onChange((value: ModelName) => hooks.onModelChange(value));
  }

  gui.add(params, 'algorithm', ['DFTL', 'PBD']).name('algorithm');
  if (hooks) {
    gui
      .add(params, 'colorMode', ['alternating', 'gradient'])
      .name('colors')
      .onChange(() => hooks.onColorModeChange());
  }
  gui.add(params, 'useFilter').name('use filter');
  gui.add(params, 'drawBoundingBox').name('draw bounding box');
  gui.add(params, 'drawVoxelGrid').name('draw voxel grid');
  gui.add(params, 'drawFur').name('draw fur');
  gui.add(params, 'playAnimation').name('play animation');

  const shader = gui.addFolder('shader params');
  shader.add(params, 'velocityDamping', 0, 1, 0.001).name('velocityDamping');
  shader.add(params, 'numIterations', 1, 200, 1).name('numIterations');
  shader.add(params, 'stiffness', 0, 1, 0.001).name('stiffness');
  shader.add(params, 'friction', 0, 1, 0.001).name('friction');
  shader.add(params, 'repulsion', 0, 1000, 0.1).name('repulsion');
  shader.add(params, 'ftlDamping', 0, 1, 0.001).name('ftlDamping');

  const gravity = gui.addFolder('gravity');
  gravity.add(params, 'gravityX', -100, 100, 0.1).name('x');
  gravity.add(params, 'gravityY', -100, 100, 0.1).name('y');
  gravity.add(params, 'gravityZ', -100, 100, 0.1).name('z');
  gravity.close();

  if (hooks?.onChange) gui.onChange(hooks.onChange);
  return gui;
}
