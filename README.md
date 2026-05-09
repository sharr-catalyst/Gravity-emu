# Gravity Simulator
[![Python](https://img.shields.io/badge/Python-3.8+-yellow?style=flat-square&logo=python)](https://www.python.org/)
[![OpenGL](https://img.shields.io/badge/OpenGL-2.1+-red?style=flat-square)](https://pypi.org/project/PyOpenGL/3.1.10/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](https://github.com/sharr-catalyst/Gravity-emu?tab=MIT-1-ov-file#readme)

A real-time 3D N-body gravity simulator with a spacetime rubber-sheet visualisation, built with Python, Pygame, and OpenGL.

---

## What it looks like

The simulation renders a warped grid that curves and deforms in real time based on the gravitational potential of every object in the scene — just like the classic "rubber sheet" analogy for spacetime curvature. A massive central star creates a deep bowl, while orbiting planets add their own smaller depressions that shift as they move. Bodies highlight when hovered, and can be grabbed and repositioned with the mouse in real time.

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
| Left-click drag (empty space) | Orbit camera (full 360°) |
| Left-click drag (on a body) | Pick up and reposition that body |
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

The grid uses the rubber-sheet spacetime analogy. Every vertex Y position is computed by summing two contributions per body:

**Main well:**
```
y(x, z) = -amp / sqrt(dx² + dz² + eps²) × eps
```

**Ripple ring** — a raised annulus just outside the well:
```
y(x, z) += ring_amp × exp(-((r - ring_r) / ring_w)²)
```

- `eps` is the softening radius — small bodies get a tighter `eps` for a sharper, more visible dip
- `amp` has a guaranteed floor (`MIN_AMP = 18%` of the star's depth) so even tiny bodies always visibly dent the sheet
- The ring sits at `3 × eps` from the body centre and rises to `35%` of the well depth, making small bodies clearly legible on the grid
- The grid rim is always normalised to `y = 0`; wells curve downward below it
- The warp updates every 2 frames, tracking objects as they move

### Influence rings

Non-star bodies get a glowing elliptical ring drawn on the warped grid surface. The ring samples the Y-displacement at each point so it follows the sheet contour. Rings brighten and thicken when the body is hovered or being dragged.

### Body dragging

Left-clicking a body freezes all body velocities and lets you drag it freely across the Y-plane. On release, orbiting bodies resume their pre-drag velocities and the dragged body's velocity is zeroed. The grid warp updates live during the drag.

### Rendering

- `GL_LINE_STRIP` draws each grid row and column as a continuous smooth curve
- Objects are rendered as `gluSphere` calls; the central star gets a brightness boost via colour scaling
- Hovered bodies brighten by +0.35 on each RGB channel
- 4× MSAA-equivalent smoothing via `GL_LINE_SMOOTH`

---

## Simulation parameters

These constants at the top of `gravity_sim.py` are easy to tune:

| Constant | Default | Effect |
|---|---|---|
| `GDIV` | 70 | Grid line count — higher = smoother, slower |
| `GSIZE` | 28000 | Grid width in scene units |
| `VIS_DEPTH` | 3200 | Maximum well depth (visual only) |
| `MIN_AMP` | VIS_DEPTH × 0.18 | Minimum warp amplitude for small bodies |
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
gravity_sim.py   — entire simulation (single file, ~440 lines)
README.md        — this file
```

---

## Known limitations

- Grid warp is a visual analogy only — it does not feed back into the physics
- No collision merging; objects that overlap continue through each other
- Performance drops noticeably above ~8 bodies due to O(n²) gravity and O(n × grid²) warp computation
- Warp recomputes every 2 frames on the CPU; a GPU shader would be needed for real-time warp with very high grid density
- Dragged body velocity is zeroed on release; a velocity estimate from drag motion is not yet implemented
