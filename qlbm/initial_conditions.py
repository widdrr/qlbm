import numpy as np
from numpy.typing import NDArray


# ── 1D ───────────────────────────────────────────────────────────────────────

def get_gaussian_initial_distribution(
    sites: int,
    center: float,
    width: float,
    amplitude: float = 0.9,
    background: float = 0.1,
) -> NDArray[np.float64]:
    x = np.arange(sites)
    return background + amplitude * np.exp(-((x - center)**2) / (2 * width**2))


def get_uniform_velocity_field(sites: int, velocity: float = 0.2) -> NDArray[np.float64]:
    return np.full((1, sites), velocity)


# ── 2D ───────────────────────────────────────────────────────────────────────

def get_ring_initial_distribution(
    sites: tuple[int, int],
    cylinder_radius_outer: float,
    cylinder_radius_inner: float,
    background_density: float = 0.1,
    ring_density: float = 0.4,
) -> NDArray[np.float64]:
    density = np.full(sites, background_density)

    x = np.arange(sites[0])
    y = np.arange(sites[1])
    X, Y = np.meshgrid(x, y)

    center_x = sites[0] / 2
    center_y = sites[1] / 2

    distance_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    ring_mask = (
        (distance_from_center < cylinder_radius_outer) &
        (distance_from_center > cylinder_radius_inner)
    ).T

    density[ring_mask] = ring_density
    return density


def get_shear_velocity_field(
    sites: tuple[int, int],
    u_magnitude: float = 0.2,
    v_magnitude: float = 0.1,
) -> NDArray[np.float64]:
    u = np.full(sites, u_magnitude)
    v = np.full(sites, v_magnitude)

    u[:, int(sites[1] / 2):] = -u_magnitude

    return np.stack([u, v])


def get_cross_initial_distribution(
    sites: tuple[int, int],
    line_width: int = 2,
    background_density: float = 0.1,
    cross_density: float = 0.9,
) -> NDArray[np.float64]:
    density = np.full(sites, background_density)

    center_x = sites[0] // 2
    center_y = sites[1] // 2

    density[:, center_y - line_width:center_y + line_width] = cross_density
    density[center_x - line_width:center_x + line_width, :] = cross_density

    return density / np.linalg.norm(density)


def get_vortex_velocity_field(
    sites: tuple[int, int],
    magnitude: float = 0.2,
    clockwise: bool = True,
) -> NDArray[np.float64]:
    x = np.arange(sites[0])
    y = np.arange(sites[1])
    X, Y = np.meshgrid(x, y)

    center_x = sites[0] / 2
    center_y = sites[1] / 2
    X_centered = X.T - center_x
    Y_centered = Y.T - center_y

    r = np.sqrt(X_centered**2 + Y_centered**2) + 1e-6

    direction = 1 if clockwise else -1
    v_theta = magnitude * direction

    u = -v_theta * Y_centered / r
    v = v_theta * X_centered / r

    return np.stack([u, v])


# ── 3D ───────────────────────────────────────────────────────────────────────

def get_3d_planes_initial_distribution(
    sites: tuple[int, int, int],
    plane_width: int = 2,
    background_density: float = 0.1,
    plane_density: float = 0.9,
) -> NDArray[np.float64]:
    density = np.full(sites, background_density)

    center_x = sites[0] // 2 - 1
    center_y = sites[1] // 2 - 1
    center_z = sites[2] // 2 - 1

    density[center_x, :, :] = plane_density
    density[:, center_y, :] = plane_density
    density[:, :, center_z] = plane_density

    return density


def get_3d_velocity_field(
    sites: tuple[int, int, int],
    A: float = 0.2,
    B: float = 0.2,
    C: float = 0.2,
    a: float = 1.0,
    b: float = 1.0,
    c: float = 1.0,
) -> NDArray[np.float64]:
    x = np.linspace(0, 2 * np.pi, sites[0])
    y = np.linspace(0, 2 * np.pi, sites[1])
    z = np.linspace(0, 2 * np.pi, sites[2])

    u = np.zeros(sites)
    v = np.zeros(sites)
    w = np.zeros(sites)

    for i in range(sites[0]):
        for j in range(sites[1]):
            for k in range(sites[2]):
                u[i, j, k] = A * np.cos(a * x[i]) * np.sin(b * y[j]) * np.sin(c * z[k])
                v[i, j, k] = B * np.sin(a * x[i]) * np.cos(b * y[j]) * np.sin(c * z[k])
                w[i, j, k] = C * np.sin(a * x[i]) * np.sin(b * y[j]) * np.cos(c * z[k])

    return np.stack([u, v, w])


# ── Boundary condition helpers ────────────────────────────────────────────────

def create_identity_bc_1d(grid_size: int, num_velocities: int) -> NDArray[np.float64]:
    bc = np.zeros((grid_size, num_velocities, num_velocities))
    for i in range(grid_size):
        bc[i] = np.eye(num_velocities)
    return bc


def create_bounceback_bc_1d(
    grid_size: int,
    links: list[list[int]],
) -> NDArray[np.float64]:
    num_velocities = len(links)
    bc = create_identity_bc_1d(grid_size, num_velocities)

    left_idx = stationary_idx = right_idx = None
    for i, link in enumerate(links):
        if link[0] == -1:
            left_idx = i
        elif link[0] == 0:
            stationary_idx = i
        elif link[0] == 1:
            right_idx = i

    print(f"Velocity indices: left={left_idx}, stationary={stationary_idx}, right={right_idx}")

    # Position 1 (next to left wall): bounce back left-moving particles
    bc[1] = np.zeros((num_velocities, num_velocities))
    bc[1][left_idx, left_idx] = 0.0
    bc[1][stationary_idx, stationary_idx] = 1.0
    bc[1][right_idx, right_idx] = 1.0
    bc[1][stationary_idx, left_idx] = 1.0

    # Position grid_size-2 (next to right wall): bounce back right-moving particles
    bc[-2] = np.zeros((num_velocities, num_velocities))
    bc[-2][left_idx, left_idx] = 1.0
    bc[-2][stationary_idx, stationary_idx] = 1.0
    bc[-2][right_idx, right_idx] = 0.0
    bc[-2][stationary_idx, right_idx] = 1.0

    return bc


def get_gaussian_2d_initial_distribution(
    sites: tuple[int, int],
    center: tuple[float, float],
    width: float,
    amplitude: float = 0.9,
    background: float = 0.1,
) -> NDArray[np.float64]:
    x = np.arange(sites[0])
    y = np.arange(sites[1])
    X, Y = np.meshgrid(x, y, indexing='ij')
    r2 = (X - center[0])**2 + (Y - center[1])**2
    return background + amplitude * np.exp(-r2 / (2 * width**2))


# ── 2D Boundary condition helpers ────────────────────────────────────────────

def get_gaussian_ring_initial_distribution(
    sites: tuple[int, int],
    ring_radius: float,
    ring_width: float,
    amplitude: float = 0.9,
    background: float = 0.1,
) -> NDArray[np.float64]:
    x = np.arange(sites[0])
    y = np.arange(sites[1])
    X, Y = np.meshgrid(x, y, indexing='ij')

    center_x = sites[0] / 2
    center_y = sites[1] / 2

    r = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    return background + amplitude * np.exp(-((r - ring_radius)**2) / (2 * ring_width**2))


def create_split_walls_bc_2d(
    grid_size: tuple[int, int],
    links: list[list[int]],
    reflect_frac: float = 0.2,
) -> NDArray[np.float64]:
    """Fully enclosed walls on all four sides of a 2D grid.

    Distributions heading perpendicular into a wall are split:
    *reflect_frac* is reflected back, the rest is divided equally
    among tangential directions that do not also point into a wall.
    """
    Nx, Ny = grid_size
    nv = len(links)

    bc = np.zeros((Nx, Ny, nv, nv))
    for x in range(Nx):
        for y in range(Ny):
            bc[x, y] = np.eye(nv)

    link_dirs: dict[tuple[int, ...], int] = {}
    for i, link in enumerate(links):
        link_dirs[tuple(link)] = i

    stat_idx = link_dirs.get((0, 0))

    for x in range(1, Nx - 1):
        for y in range(1, Ny - 1):
            # Collect (perp_in, perp_out) for each adjacent wall
            wall_links: list[tuple[int, int]] = []
            if x == 1 and (-1, 0) in link_dirs and (1, 0) in link_dirs:
                wall_links.append((link_dirs[(-1, 0)], link_dirs[(1, 0)]))
            if x == Nx - 2 and (1, 0) in link_dirs and (-1, 0) in link_dirs:
                wall_links.append((link_dirs[(1, 0)], link_dirs[(-1, 0)]))
            if y == 1 and (0, -1) in link_dirs and (0, 1) in link_dirs:
                wall_links.append((link_dirs[(0, -1)], link_dirs[(0, 1)]))
            if y == Ny - 2 and (0, 1) in link_dirs and (0, -1) in link_dirs:
                wall_links.append((link_dirs[(0, 1)], link_dirs[(0, -1)]))

            if not wall_links:
                continue

            wall_in_set = {wl[0] for wl in wall_links}

            for perp_in, perp_out in wall_links:
                bc[x, y, perp_in, perp_in] = 0.0
                bc[x, y, perp_out, perp_in] = reflect_frac

                safe_tangentials = [
                    t for t in range(nv)
                    if t != perp_in and t != perp_out
                    and t != stat_idx and t not in wall_in_set
                ]

                tangent_frac = 1.0 - reflect_frac
                if safe_tangentials:
                    frac_each = tangent_frac / len(safe_tangentials)
                    for t in safe_tangentials:
                        bc[x, y, t, perp_in] = frac_each
                else:
                    # No safe tangentials (e.g. narrow corridor) — full reflection
                    bc[x, y, perp_out, perp_in] = 1.0

    return bc
