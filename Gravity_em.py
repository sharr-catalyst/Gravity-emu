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

W, H = 1200, 800   # window size (referenced in unproject)


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
        vol        = (3 * self.mass / self.density) / (4 * math.pi)
        self.radius = (vol ** (1.0/3.0)) / SIZE_RATIO

    def draw(self, highlight=False):
        glPushMatrix()
        glTranslatef(*self.pos)
        r, g, b, a = self.color
        if self.glow:
            r, g, b = min(r*2.2, 1), min(g*2.2, 1), min(b*2.2, 1)
        if highlight:
            # brighten the body when it is under the cursor / being dragged
            r, g, b = min(r+0.35, 1), min(g+0.35, 1), min(b+0.35, 1)
        glColor4f(r, g, b, a)
        gluSphere(gluNewQuadric(), max(self.radius, 60.0), 32, 32)
        glPopMatrix()


# ── Grid mesh ──────────────────────────────────────────────────────────────────
GDIV  = 70
GSIZE = 28000

def make_grid_xz():
    xs = np.linspace(-GSIZE/2, GSIZE/2, GDIV+1)
    zs = np.linspace(-GSIZE/2, GSIZE/2, GDIV+1)
    return xs, zs


def compute_warp(xs, zs, bodies):
    """
    Two-term warp per body:

      1. Main well  : -amp / sqrt(r² + eps²) * eps
                      amp has a guaranteed floor so small bodies are always visible.

      2. Ripple ring: +ring_amp * exp(-((r - ring_r)/ring_w)²)
                      A raised annulus just outside the well — gives the small
                      bodies a distinctive halo-ridge that is easy to see even
                      when the central dip is shallow.

    Small bodies get tighter eps (narrower, punchy well) and a proportionally
    larger ripple, making their local distortion clearly legible.
    """
    nz, nx    = len(zs), len(xs)
    Y         = np.zeros((nz, nx), dtype=np.float64)

    REF_MASS  = 1.989e25      # normalisation anchor (star mass)
    VIS_DEPTH = 3200.0        # max well depth for the star

    # minimum visible amplitude so small bodies always dent the sheet
    MIN_AMP   = VIS_DEPTH * 0.18   # ~18 % of star depth — clearly legible

    for b in bodies:
        bx, _, bz = b.pos
        mass_ratio = b.mass / REF_MASS

        # ── Well parameters ─────────────────────────────────────────────────
        # Small bodies → tighter softening → sharper, more visible dip
        if mass_ratio < 0.01:          # "small" body
            eps = max(b.radius * 0.9, 400.0)
        else:
            eps = max(b.radius * 1.8, 1200.0)

        # Raw amplitude, then lift to the minimum floor
        amp = max(VIS_DEPTH * mass_ratio, MIN_AMP)

        # ── Ripple-ring parameters ───────────────────────────────────────────
        # Ring radius ≈ 3× the softening so it sits cleanly outside the well
        ring_r   = eps * 3.0
        ring_w   = eps * 1.4          # Gaussian half-width
        # Ring height is 35 % of the well depth — noticeable but not dominant
        ring_amp = amp * 0.35

        for zi, z in enumerate(zs):
            for xi, x in enumerate(xs):
                dx  = x - bx
                dz  = z - bz
                r2  = dx*dx + dz*dz
                r   = math.sqrt(r2)

                # main downward well
                well = math.sqrt(r2 + eps*eps)
                Y[zi, xi] -= amp / well * eps

                # upward ripple ring
                dr   = r - ring_r
                Y[zi, xi] += ring_amp * math.exp(-(dr*dr) / (ring_w*ring_w))

    Y -= Y.max()
    return Y


def draw_influence_ring(body, Y, xs, zs, highlight=False):
    """
    Draw a glowing ellipse on the warped grid surface around a body.
    Samples the Y-displacement at each ring point so it follows the sheet.
    Only drawn for non-glow (small) bodies.
    """
    if body.glow:
        return   # star doesn't need it

    bx, _, bz = body.pos
    r, g, b_c, _ = body.color

    # ring radius in scene units — large enough to be clearly visible
    ring_scene_r = max(body.radius, 60.0) * 8.0

    SEGS = 64
    alpha = 0.85 if highlight else 0.45
    width = 2.5  if highlight else 1.5

    glLineWidth(width)
    glBegin(GL_LINE_LOOP)
    for i in range(SEGS):
        angle = 2.0 * math.pi * i / SEGS
        px = bx + ring_scene_r * math.cos(angle)
        pz = bz + ring_scene_r * math.sin(angle)

        # bilinear-ish sample: find nearest grid cell Y
        xi_f = (px - xs[0]) / (xs[-1] - xs[0]) * (len(xs) - 1)
        zi_f = (pz - zs[0]) / (zs[-1] - zs[0]) * (len(zs) - 1)
        xi   = int(max(0, min(len(xs)-1, round(xi_f))))
        zi   = int(max(0, min(len(zs)-1, round(zi_f))))
        py   = float(Y[zi, xi]) + 5.0   # sit just above the mesh

        glColor4f(r, g, b_c, alpha)
        glVertex3f(px, py, pz)
    glEnd()
    glLineWidth(1.0)



    nz, nx = len(zs), len(xs)
    glLineWidth(1.0)
    glColor4f(0.50, 0.62, 0.72, 0.85)
    for zi in range(nz):
        glBegin(GL_LINE_STRIP)
        for xi in range(nx):
            glVertex3f(float(xs[xi]), float(Y[zi, xi]), float(zs[zi]))
        glEnd()
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


# ── Picking helpers ────────────────────────────────────────────────────────────
def screen_to_ray(mx, my):
    """
    Return (origin, direction) of the eye-ray through pixel (mx, my).
    Uses glUnProject with the current OpenGL matrices.
    """
    viewport = glGetIntegerv(GL_VIEWPORT)          # (x, y, w, h)
    mv       = glGetDoublev(GL_MODELVIEW_MATRIX)
    proj     = glGetDoublev(GL_PROJECTION_MATRIX)
    # flip Y: OpenGL origin is bottom-left, pygame is top-left
    wy = viewport[3] - my - 1
    near = gluUnProject(mx, wy, 0.0, mv, proj, viewport)
    far  = gluUnProject(mx, wy, 1.0, mv, proj, viewport)
    origin    = np.array(near, dtype=np.float64)
    direction = np.array(far,  dtype=np.float64) - origin
    direction /= np.linalg.norm(direction)
    return origin, direction
def draw_grid(xs, zs, Y):
    nz, nx = len(zs), len(xs)
    glLineWidth(1.0)
    glColor4f(0.50, 0.62, 0.72, 0.85)
    for zi in range(nz):
        glBegin(GL_LINE_STRIP)
        for xi in range(nx):
            glVertex3f(float(xs[xi]), float(Y[zi, xi]), float(zs[zi]))
        glEnd()
    for xi in range(nx):
        glBegin(GL_LINE_STRIP)
        for zi in range(nz):
            glVertex3f(float(xs[xi]), float(Y[zi, xi]), float(zs[zi]))
        glEnd()

def ray_sphere(origin, direction, center, radius):
    """Return distance t > 0 to closest intersection, or None."""
    oc = origin - center
    b  = 2.0 * np.dot(direction, oc)
    c  = np.dot(oc, oc) - radius * radius
    disc = b*b - 4.0*c
    if disc < 0:
        return None
    sq   = math.sqrt(disc)
    t1   = (-b - sq) / 2.0
    t2   = (-b + sq) / 2.0
    t    = t1 if t1 > 0 else t2
    return t if t > 0 else None


def hit_body(mx, my, bodies):
    """Return index of the body under cursor, or -1."""
    origin, direction = screen_to_ray(mx, my)
    best_t, best_i = 1e18, -1
    for i, b in enumerate(bodies):
        pick_r = max(b.radius, 60.0) * 2.5   # generous pick radius
        t = ray_sphere(origin, direction, b.pos, pick_r)
        if t is not None and t < best_t:
            best_t, best_i = t, i
    return best_i


def ray_y_plane(origin, direction, y=0.0):
    """Intersect ray with the horizontal plane Y=y.  Returns XZ point or None."""
    if abs(direction[1]) < 1e-9:
        return None
    t = (y - origin[1]) / direction[1]
    if t < 0:
        return None
    p = origin + direction * t
    return np.array([p[0], y, p[2]], dtype=np.float64)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    pygame.init()
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
        Body([0,      0, 0],     [0,0,0],    M_star,   5515, (1.0, 0.92, 0.55, 1.0), True),
        Body([-r_orb, 0,  300],  [0,0, v_c], 4.0e22,   5515, (0.3, 0.75, 1.0,  1.0)),
        Body([ r_orb, 0, -300],  [0,0,-v_c], 4.0e22,   5515, (0.3, 0.75, 1.0,  1.0)),
    ]

    xs, zs = make_grid_xz()
    Y      = compute_warp(xs, zs, bodies)

    clock      = pygame.time.Clock()
    paused     = False
    SIM_SPEED  = 80.0
    SUBSTEPS   = 6
    warp_tick  = 0

    # ── Drag-body state ────────────────────────────────────────────────────────
    dragged_idx   = -1      # which body is being dragged (-1 = none)
    drag_offset   = None    # 3-D offset from body centre to grab point
    hover_idx     = -1      # body under cursor (for highlight)
    saved_vels    = None    # ALL bodies' velocities saved when drag starts

    GRID_LIMIT    = GSIZE / 2 * 0.92   # keep bodies inside ~92 % of grid half-width

    while True:
        dt_real = min(clock.tick(60) / 1000.0, 0.033)
        mx, my  = pygame.mouse.get_pos()

        # ── Hover detection (every frame, cheap) ───────────────────────────────
        # Need matrices set first — do a dry apply_cam before event loop reads
        apply_cam()
        hover_idx = hit_body(mx, my, bodies) if dragged_idx == -1 else dragged_idx

        # ── Events ─────────────────────────────────────────────────────────────
        for ev in pygame.event.get():
            if ev.type == QUIT:        pygame.quit(); return
            if ev.type == KEYDOWN:
                if ev.key == K_ESCAPE: pygame.quit(); return
                if ev.key == K_p:      paused = not paused
                if ev.key in (K_EQUALS, K_PLUS):
                    SIM_SPEED = min(SIM_SPEED * 1.5, 6000.0)
                if ev.key == K_MINUS:
                    SIM_SPEED = max(SIM_SPEED / 1.5, 1.0)
                if ev.key == K_r:
                    cam.update({"yaw":0,"pitch":28,"dist":13000,
                                "cx":0,"cy":0,"cz":0})

            # ── Mouse button DOWN ───────────────────────────────────────────────
            if ev.type == MOUSEBUTTONDOWN:
                if ev.button == 1:
                    _drag["last"] = pygame.mouse.get_pos()
                    hit = hit_body(mx, my, bodies)
                    if hit >= 0:
                        # start body-drag — freeze ALL bodies so nothing slingsshots
                        dragged_idx = hit
                        saved_vels  = [b.vel.copy() for b in bodies]
                        for b in bodies:
                            b.vel = np.zeros(3, dtype=np.float64)
                        # compute grab offset on Y=body.pos[1] plane
                        origin, direction = screen_to_ray(mx, my)
                        pt = ray_y_plane(origin, direction, bodies[hit].pos[1])
                        drag_offset = (bodies[hit].pos.copy() - pt) if pt is not None \
                                      else np.zeros(3, dtype=np.float64)
                        _drag["lmb"] = False   # no camera orbit while dragging
                    else:
                        _drag["lmb"] = True
                if ev.button == 3:
                    _drag["rmb"]  = True
                    _drag["last"] = pygame.mouse.get_pos()
                if ev.button == 4:
                    cam["dist"] = max(cam["dist"] * 0.88, 200.0)
                if ev.button == 5:
                    cam["dist"] = min(cam["dist"] * 1.13, 200000.0)

            # ── Mouse button UP ─────────────────────────────────────────────────
            if ev.type == MOUSEBUTTONUP:
                if ev.button == 1:
                    if dragged_idx >= 0 and saved_vels is not None:
                        # restore every body's pre-drag velocity
                        for i, b in enumerate(bodies):
                            b.vel = saved_vels[i].copy()
                        # zero out the dragged body's velocity (it was moved manually)
                        bodies[dragged_idx].vel = np.zeros(3, dtype=np.float64)
                    dragged_idx = -1
                    drag_offset = None
                    saved_vels  = None
                    _drag["lmb"] = False
                if ev.button == 3:
                    _drag["rmb"] = False

            # ── Mouse MOTION ────────────────────────────────────────────────────
            if ev.type == MOUSEMOTION:
                cx2, cy2 = pygame.mouse.get_pos()
                dx = cx2 - _drag["last"][0]
                dy = cy2 - _drag["last"][1]
                _drag["last"] = (cx2, cy2)

                if dragged_idx >= 0:
                    # Move the body to wherever the ray hits the Y-plane
                    b = bodies[dragged_idx]
                    origin, direction = screen_to_ray(cx2, cy2)
                    pt = ray_y_plane(origin, direction, b.pos[1])
                    if pt is not None:
                        new_pos = pt + drag_offset
                        # clamp to grid so it never escapes the visible mesh
                        new_pos[0] = max(-GRID_LIMIT, min(GRID_LIMIT, new_pos[0]))
                        new_pos[2] = max(-GRID_LIMIT, min(GRID_LIMIT, new_pos[2]))
                        b.pos = new_pos
                    b.vel = np.zeros(3, dtype=np.float64)
                    # live warp update while dragging
                    Y = compute_warp(xs, zs, bodies)

                elif _drag["lmb"]:
                    cam["yaw"]   += dx * 0.35
                    cam["pitch"] += dy * 0.35
                    cam["pitch"]  = max(-89, min(89, cam["pitch"]))

                if _drag["rmb"]:
                    pan = cam["dist"] * 0.0009
                    ry  = math.radians(cam["yaw"])
                    cam["cx"] -= math.cos(ry) * dx * pan
                    cam["cz"] += math.sin(ry) * dx * pan
                    cam["cy"] -= dy * pan * 0.5

        # ── Physics — fully paused while any body is being dragged ────────────
        if not paused and dragged_idx < 0:
            sub_dt = dt_real * SIM_SPEED / SUBSTEPS
            for _ in range(SUBSTEPS):
                physics_step(bodies, sub_dt)

            warp_tick += 1
            if warp_tick >= 2:
                Y         = compute_warp(xs, zs, bodies)
                warp_tick = 0

        # ── Render ─────────────────────────────────────────────────────────────
        glClearColor(0.01, 0.01, 0.03, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        apply_cam()
        draw_grid(xs, zs, Y)
        for i, b in enumerate(bodies):
            b.draw(highlight=(i == hover_idx))
            draw_influence_ring(b, Y, xs, zs, highlight=(i == hover_idx))

        drag_hint = "  [DRAGGING]" if dragged_idx >= 0 else ""
        pygame.display.set_caption(
            f"Gravity Sim  |  {'PAUSED' if paused else 'RUNNING'}  "
            f"x{SIM_SPEED:.0f}  |  Click body to drag  RMB pan  Scroll zoom  "
            f"+/- speed  P pause  R reset{drag_hint}")
        pygame.display.flip()


if __name__ == "__main__":
    main()
