import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import math

# ── Constants ──────────────────────────────────────────────────────────────────
G           = 6.6743e-11
C           = 299792458.0
SIZE_RATIO  = 30000.0
SCENE_TO_M  = 1000.0

# ── Camera ─────────────────────────────────────────────────────────────────────
cam = {"yaw": 0.0, "pitch": 28.0, "dist": 13000.0,
       "cx": 0.0, "cy": 0.0, "cz": 0.0}
_drag = {"lmb": False, "rmb": False, "last": (0, 0)}


# ── Body ───────────────────────────────────────────────────────────────────────
class Body:
    def __init__(self, pos, vel, mass, density=5515,
                 color=(1.0, 1.0, 1.0, 1.0), glow=False):
        self.pos     = np.array(pos,  dtype=np.float64)
        self.vel     = np.array(vel,  dtype=np.float64)
        self.mass    = float(mass)
        self.density = density
        self.color   = color
        self.glow    = glow
        self._r()

    def _r(self):
        vol = (3 * self.mass / self.density) / (4 * math.pi)
        self.radius = (vol ** (1.0/3.0)) / SIZE_RATIO

    def draw(self):
        glPushMatrix()
        glTranslatef(*self.pos)
        r, g, b, a = self.color
        if self.glow:
            r, g, b = min(r*2.2,1), min(g*2.2,1), min(b*2.2,1)
        glColor4f(r, g, b, a)
        gluSphere(gluNewQuadric(), max(self.radius, 60.0), 32, 32)
        glPopMatrix()


# ── Grid mesh ──────────────────────────────────────────────────────────────────
GDIV  = 70          # lines each direction — more = smoother
GSIZE = 28000       # total grid width in scene units

def make_grid_xz():
    """Return xs, zs as 1-D arrays of grid line positions."""
    xs = np.linspace(-GSIZE/2, GSIZE/2, GDIV+1)
    zs = np.linspace(-GSIZE/2, GSIZE/2, GDIV+1)
    return xs, zs


def compute_warp(xs, zs, bodies):
    """
    Build a (nz, nx) Y-displacement matrix.

    Formula used by every "rubber sheet" spacetime visualisation:
        y(x,z) = -A * sum_i  mass_i / sqrt( (x-xi)^2 + (z-zi)^2 + eps^2 )

    eps (softening) is large so the well is WIDE and SMOOTH — no spike.
    A (amplitude) is chosen so the deepest point is ~VIS_DEPTH scene-units down.
    """
    nz, nx = len(zs), len(xs)
    Y = np.zeros((nz, nx), dtype=np.float64)

    REF_MASS = 1.989e25          # reference so amplitude stays consistent
    VIS_DEPTH = 3200.0           # how deep the central well goes (scene units)
    EPS_FACTOR = 1.8             # softening = EPS_FACTOR * object_radius
                                  # larger → flatter/wider bowl

    for b in bodies:
        bx, _, bz = b.pos
        # softening radius — makes the bowl wide; minimum 1200 scene-units
        eps = max(b.radius * EPS_FACTOR, 1200.0)
        # scale amplitude by mass relative to reference
        amp = VIS_DEPTH * (b.mass / REF_MASS)

        for zi, z in enumerate(zs):
            for xi, x in enumerate(xs):
                dx  = x - bx
                dz  = z - bz
                r   = math.sqrt(dx*dx + dz*dz + eps*eps)
                Y[zi, xi] -= amp / r * eps   # eps normalises units

    # Shift so the rim (edges) sits at y=0, wells go downward
    Y -= Y.max()
    return Y


def draw_grid(xs, zs, Y):
    """
    Draw the warped grid as smooth GL_LINE_STRIP lines.
    Row lines (fixed Z, vary X) and column lines (fixed X, vary Z).
    """
    nz, nx = len(zs), len(xs)

    glLineWidth(1.0)
    glColor4f(0.50, 0.62, 0.72, 0.85)

    # Row lines — each is a smooth curve along X
    for zi in range(nz):
        glBegin(GL_LINE_STRIP)
        for xi in range(nx):
            glVertex3f(float(xs[xi]), float(Y[zi, xi]), float(zs[zi]))
        glEnd()

    # Column lines — each is a smooth curve along Z
    for xi in range(nx):
        glBegin(GL_LINE_STRIP)
        for zi in range(nz):
            glVertex3f(float(xs[xi]), float(Y[zi, xi]), float(zs[zi]))
        glEnd()


# ── Physics ────────────────────────────────────────────────────────────────────
def physics_step(bodies, dt):
    n    = len(bodies)
    accs = [np.zeros(3, dtype=np.float64) for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j: continue
            d_su = bodies[j].pos - bodies[i].pos
            r_su = np.linalg.norm(d_su)
            if r_su < 0.5: continue
            r_m  = r_su * SCENE_TO_M
            soft = (bodies[i].radius + bodies[j].radius) * SCENE_TO_M * 0.3
            r_eff = math.sqrt(r_m*r_m + soft*soft)
            a    = (G * bodies[j].mass) / r_eff**2 / SCENE_TO_M
            accs[i] += (d_su / r_su) * a
    for i, b in enumerate(bodies):
        b.vel += accs[i] * dt
        b.pos += b.vel  * dt


# ── Camera ─────────────────────────────────────────────────────────────────────
def apply_cam():
    glLoadIdentity()
    glTranslatef(0, 0, -cam["dist"])
    glRotatef(-cam["pitch"], 1, 0, 0)
    glRotatef( cam["yaw"],   0, 1, 0)
    glTranslatef(-cam["cx"], -cam["cy"], -cam["cz"])


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    pygame.init()
    W, H = 1200, 800
    pygame.display.set_mode((W, H), DOUBLEBUF | OPENGL)

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glEnable(GL_LINE_SMOOTH)
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

    glMatrixMode(GL_PROJECTION); glLoadIdentity()
    gluPerspective(48, W/H, 1.0, 500000.0)
    glMatrixMode(GL_MODELVIEW)

    # ── Bodies ─────────────────────────────────────────────────────────────────
    M_star = 1.989e25
    r_orb  = 5200.0
    v_c    = math.sqrt(G * M_star / (r_orb * SCENE_TO_M)) / SCENE_TO_M * 0.82

    bodies = [
        Body([0, 0, 0], [0,0,0], M_star, 5515, (1.0, 0.92, 0.55, 1.0), True),
        Body([-r_orb, 0,  300], [0, 0,  v_c], 4.0e22, 5515, (0.3, 0.75, 1.0, 1.0)),
        Body([ r_orb, 0, -300], [0, 0, -v_c], 4.0e22, 5515, (0.3, 0.75, 1.0, 1.0)),
    ]

    xs, zs    = make_grid_xz()
    Y         = compute_warp(xs, zs, bodies)   # initial warp

    clock      = pygame.time.Clock()
    paused     = False
    SIM_SPEED  = 80.0
    SUBSTEPS   = 6
    warp_tick  = 0

    while True:
        dt_real = min(clock.tick(60) / 1000.0, 0.033)

        # ── Events ─────────────────────────────────────────────────────────────
        for ev in pygame.event.get():
            if ev.type == QUIT: pygame.quit(); return
            if ev.type == KEYDOWN:
                if ev.key == K_ESCAPE: pygame.quit(); return
                if ev.key == K_p:      paused = not paused
                if ev.key in (K_EQUALS, K_PLUS):
                    SIM_SPEED = min(SIM_SPEED * 1.5, 6000.0)
                if ev.key == K_MINUS:
                    SIM_SPEED = max(SIM_SPEED / 1.5, 1.0)
                if ev.key == K_r:
                    cam.update({"yaw":0,"pitch":28,"dist":13000,"cx":0,"cy":0,"cz":0})

            if ev.type == MOUSEBUTTONDOWN:
                if ev.button == 1: _drag["lmb"]=True;  _drag["last"]=pygame.mouse.get_pos()
                if ev.button == 3: _drag["rmb"]=True;  _drag["last"]=pygame.mouse.get_pos()
                if ev.button == 4: cam["dist"] = max(cam["dist"]*0.88, 200.0)
                if ev.button == 5: cam["dist"] = min(cam["dist"]*1.13, 200000.0)
            if ev.type == MOUSEBUTTONUP:
                if ev.button == 1: _drag["lmb"] = False
                if ev.button == 3: _drag["rmb"] = False
            if ev.type == MOUSEMOTION:
                mx, my = pygame.mouse.get_pos()
                dx = mx - _drag["last"][0]; dy = my - _drag["last"][1]
                _drag["last"] = (mx, my)
                if _drag["lmb"]:
                    cam["yaw"]   += dx * 0.35
                    cam["pitch"] += dy * 0.35
                    cam["pitch"]  = max(-89, min(89, cam["pitch"]))
                if _drag["rmb"]:
                    pan = cam["dist"] * 0.0009
                    ry  = math.radians(cam["yaw"])
                    cam["cx"] -= math.cos(ry) * dx * pan
                    cam["cz"] += math.sin(ry) * dx * pan
                    cam["cy"] -= dy * pan * 0.5

        # ── Physics ────────────────────────────────────────────────────────────
        if not paused:
            sub_dt = dt_real * SIM_SPEED / SUBSTEPS
            for _ in range(SUBSTEPS):
                physics_step(bodies, sub_dt)

            warp_tick += 1
            if warp_tick >= 2:          # recompute warp every 2 frames
                Y = compute_warp(xs, zs, bodies)
                warp_tick = 0

        # ── Render ─────────────────────────────────────────────────────────────
        glClearColor(0.01, 0.01, 0.03, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        apply_cam()
        draw_grid(xs, zs, Y)
        for b in bodies: b.draw()

        pygame.display.set_caption(
            f"Gravity Sim  |  {'PAUSED' if paused else 'RUNNING'}  "
            f"x{SIM_SPEED:.0f}  |  LMB orbit  RMB pan  Scroll zoom  +/-")
        pygame.display.flip()


if __name__ == "__main__":
    main()