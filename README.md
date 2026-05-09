# Gravity Simulator
[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)](https://www.python.org/)
[![OpenGL](https://img.shields.io/badge/OpenGL-2.1+-red?style=flat-square)](https://pypi.org/project/PyOpenGL/3.1.10/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](https://opensource.org/licenses/MIT)
 
A real-time 3D N-body gravity simulator with a spacetime rubber-sheet visualisation, built with Python, Pygame, and OpenGL.
 
---
 
## What it looks like
 
The simulation renders a warped grid that curves and deforms in real time based on the gravitational potential of every object in the scene — just like the classic "rubber sheet" analogy for spacetime curvature. A massive central star creates a deep bowl, while orbiting planets add their own smaller depressions that shift as they move.
 
---
 
## Requirements
 
Python 3.8 or higher, plus three packages:
 
```
pip install pygame PyOpenGL PyOpenGL_accelerate numpy
```
 
---
 
## Running
 
```
python gravity_sim.py
```
 
---
 
## Controls
 
| Input | Action |
|---|---|
| Left-click drag | Orbit camera (full 360°) |
| Right-click drag | Pan camera |
| Scroll wheel | Zoom in / out |
| `P` | Pause / unpause |
| `+` | Increase simulation speed (×1.5) |
| `-` | Decrease simulation speed (÷1.5) |
| `R` | Reset camera to default position |
| `ESC` | Quit |
 
---
 
## How it works
 
### Physics
 
Each body exerts a gravitational force on every other body using Newton's law:
 
```
a = G × M / r²
```
 
Acceleration is computed in SI units (m/s²), converted back to scene units (1 unit = 1 km), then integrated using a sub-stepped Euler method (6 sub-steps per frame) for stability. A softening term prevents singularities when objects get close:
 
```
r_eff = sqrt(r² + softening²)
```
 
### Grid warp
 
The grid uses the rubber-sheet spacetime analogy. Every vertex Y position is computed by summing the contribution of each body:
 
```
y(x, z) = -A × mass / sqrt(dx² + dz² + eps²)
```
 
- `eps` is the softening radius — large enough (min 1200 scene-units) to produce a wide smooth bowl instead of a spike
- `A` scales depth so the deepest well is always `VIS_DEPTH = 3200` scene-units below the flat rim
- The grid rim is always normalised to `y = 0`; wells curve downward below it
- The warp updates every 2 frames, tracking objects as they move
### Rendering
 
- `GL_LINE_STRIP` draws each grid row and column as a continuous smooth curve rather than disconnected flat segments
- Objects are rendered as `gluSphere` calls; the central star gets a brightness boost via colour scaling
- 4× MSAA-equivalent smoothing via `GL_LINE_SMOOTH`
---
 
## Simulation parameters
 
These constants at the top of `gravity_sim.py` are easy to tune:
 
| Constant | Default | Effect |
|---|---|---|
| `GDIV` | 70 | Grid line count — higher = smoother, slower |
| `GSIZE` | 28000 | Grid width in scene units |
| `VIS_DEPTH` | 3200 | Maximum well depth (visual only) |
| `EPS_FACTOR` | 1.8 | Well width multiplier — larger = wider bowl |
| `SIM_SPEED` | 80 | Simulated seconds per real second at start |
| `SUBSTEPS` | 6 | Physics sub-steps per frame |
 
---
 
## Initial scene
 
| Body | Mass | Notes |
|---|---|---|
| Central star | 1.989 × 10²⁵ kg | ~10× solar mass, glowing yellow |
| Planet 1 | 4.0 × 10²² kg | Orbiting at 5200 scene-units, cyan |
| Planet 2 | 4.0 × 10²² kg | Counter-orbiting, cyan |
 
Planets start at 82% of the circular orbit velocity, giving elliptical orbits that precess visibly over time.
 
---
 
## File structure
 
```
gravity_sim.py   — entire simulation (single file, ~280 lines)
README.md        — this file
```
 
---
 
## Known limitations
 
- Grid warp is a visual analogy only — it does not feed back into the physics
- No collision merging; objects that overlap continue through each other
- Performance drops noticeably above ~8 bodies due to O(n²) gravity and O(n × grid²) warp computation
- Warp recomputes every 2 frames on the CPU; a GPU shader would be needed for real-time warp with very high grid density
---
