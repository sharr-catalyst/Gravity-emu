import numpy as np
import math
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Gravity_em import Body, physics_step, compute_warp, make_grid_xz, ray_sphere, ray_y_plane

# ── Body tests ────────────────────────────────────────────────────────────────

def test_body_creation():
    b = Body([0, 0, 0], [0, 0, 0], 1.989e25)
    assert b.mass == 1.989e25
    assert b.radius > 0

def test_body_radius_scales_with_mass():
    b1 = Body([0, 0, 0], [0, 0, 0], 1.0e25)
    b2 = Body([0, 0, 0], [0, 0, 0], 8.0e25)
    assert b2.radius > b1.radius

# ── Physics tests ─────────────────────────────────────────────────────────────

def test_physics_attraction():
    """Two bodies should move toward each other."""
    b1 = Body([0,   0, 0], [0, 0, 0], 1.989e25)
    b2 = Body([5000,0, 0], [0, 0, 0], 1.989e25)
    physics_step([b1, b2], dt=1.0)
    assert b1.pos[0] > 0    # b1 moved right (toward b2)
    assert b2.pos[0] < 5000 # b2 moved left  (toward b1)

def test_physics_no_self_force():
    """Single body should not move."""
    b = Body([100, 0, 0], [0, 0, 0], 1.989e25)
    physics_step([b], dt=1.0)
    assert np.allclose(b.pos, [100, 0, 0])

def test_velocity_applied():
    """Body with velocity should move even with no other bodies."""
    b = Body([0, 0, 0], [10, 0, 0], 1.989e25)
    physics_step([b], dt=1.0)
    assert b.pos[0] == 10.0

# ── Warp tests ────────────────────────────────────────────────────────────────

def test_warp_shape():
    xs, zs = make_grid_xz()
    bodies = [Body([0, 0, 0], [0, 0, 0], 1.989e25)]
    Y = compute_warp(xs, zs, bodies)
    assert Y.shape == (len(zs), len(xs))

def test_warp_max_zero():
    """Warp should always be normalised so max = 0."""
    xs, zs = make_grid_xz()
    bodies = [Body([0, 0, 0], [0, 0, 0], 1.989e25)]
    Y = compute_warp(xs, zs, bodies)
    assert math.isclose(Y.max(), 0.0, abs_tol=1e-6)

def test_warp_dips_under_massive_body():
    """Centre of grid should dip below 0 with a massive body."""
    xs, zs = make_grid_xz()
    bodies = [Body([0, 0, 0], [0, 0, 0], 1.989e25)]
    Y = compute_warp(xs, zs, bodies)
    assert Y.min() < -100

# ── Ray tests ─────────────────────────────────────────────────────────────────

def test_ray_sphere_hit():
    origin    = np.array([0.0, 0.0, -100.0])
    direction = np.array([0.0, 0.0,   1.0])
    t = ray_sphere(origin, direction, np.array([0.0, 0.0, 0.0]), 10.0)
    assert t is not None
    assert t > 0

def test_ray_sphere_miss():
    origin    = np.array([100.0, 0.0, -100.0])
    direction = np.array([0.0,   0.0,    1.0])
    t = ray_sphere(origin, direction, np.array([0.0, 0.0, 0.0]), 10.0)
    assert t is None

def test_ray_y_plane():
    origin    = np.array([0.0, 10.0, 0.0])
    direction = np.array([0.0, -1.0, 0.0])
    pt = ray_y_plane(origin, direction, y=0.0)
    assert pt is not None
    assert math.isclose(pt[1], 0.0, abs_tol=1e-9)