# Video reference — vimeo.com/131453436 "FurryBall - hair/fur simulation with OpenGL" (2015-06)

Frame-by-frame findings from the original demo video (predates the checked-in beast model — the
whole video is the procedural sphere "FurryBall" mode). These define the **video-faithful FurryBall
mode** to be layered on top of the base port.

## Scale (from on-screen info text)
- Num Hairstrands: **80,601**, Num Particles: **1,289,616** → **16 particles per strand**
  (the checked-in code uses 8; particles-per-strand is uniform-driven in our kernels, so 16 needs
  no shader change — numStrandsPerThreadGroup becomes 64/16 = 4, strand padding multiple 4).
- Sphere ≈ radius 4 (matches `ofMesh::sphere(4, …)` and the commented sphere-collision at radius 4).

## Colors — root→tip gradient (NOT the checked-in alternating strands)
- Deep interior/roots: red-orange; mid-strand: yellow-green; tips/silhouette: light teal-cyan.
- Reads like an HSV-style lerp orange(~30°) → teal(~175°) passing through yellow/green.
  Implement as 3-stop gradient per particle j/(N−1): root #d5491d → mid #b7cf4f → tip #52c8bc
  (tune visually against frames). Keep the checked-in alternating orange/teal scheme as the second
  color mode (default for beast model).

## Animation (both lines exist commented-out in ofApp.cpp — active in the video build)
- Bounce: `translation = (0, 4 + 5·|sin(t)|, 0)`
- Roll: orientation slerp between identity and 180° about axis (1,1,0), factor sin(0.2·t)
  (`mModelOrientation.slerp(sin(0.2f*t), rot0, rot180)`), applied via postMultRotate.
- Gated by the playAnimation toggle.

## Voxel grid BB for sphere mode
Grid space is model-local (fill subtracts modelTranslation). The checked-in BB (−5,0,−5)…(5,10,5)
suits the standing beast; for the bouncing sphere use a symmetric model-local BB
(−7,−7,−7)…(7,7,7) so the whole ball stays gridded through the bounce.

## Look
- Background near-white (the circular lightGray→white gradient is subtle).
- Floor: large white diamond (45°-rotated 30×30 plane), soft fading reflection of the ball —
  clearly visible whenever the ball is airborne.
- Fur: thin 1px-ish lines, soft dense look, no glow/additive blowout → alpha/opaque lines + MSAA.
- GUI param ranges seen: friction 0–0.24, repulsion 0.61–3.4 (video build used the normalized-
  gradient repulsion scale; final checked-in shader uses non-normalized gradient with repulsion
  ~117 — keep settings.xml defaults with our ported non-normalized math).
- Debug views: voxel grid = colored points (direction-coded colors); bounding box = red/dark
  wireframe box; both toggled while fur hidden or shown.

## Reference stills
`/private/tmp/claude-502/-Users-patrickfurst-Projects-hairymess-web/0cd67925-87b1-4dbe-89d0-cb555c81f288/scratchpad/reference/vimeo_thumb.jpg`
(poster frame; more frames were reviewed at 0:20, 1:00, 1:40, 2:25, 3:05, 3:45, 4:30, 4:57).

## Mode matrix (target state)
| Mode   | Default source        | Strands | Particles/strand | Colors      | Animation        | BB                  |
|--------|-----------------------|---------|------------------|-------------|------------------|---------------------|
| beast  | beast.bin (default)   | 17,365  | 8                | alternating | walk cycle (baked)| (−5,0,−5)…(5,10,5) |
| sphere | procedural (?model=sphere or beast.bin missing) | ~80,600 | 16 | gradient | bounce + roll | (−7,−7,−7)…(7,7,7) |
