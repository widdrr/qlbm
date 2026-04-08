import numpy as np
from numpy.typing import NDArray
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML


def boundary_mask(boundary_conditions: NDArray) -> NDArray[np.bool_]:
    """Derive a boolean mask of boundary nodes from a BC array.

    A site is a boundary node if its local BC matrix differs from the identity.
    Works for any dimensionality: 1D shape (N, nv, nv), 2D shape (Nx, Ny, nv, nv), etc.
    """
    nv = boundary_conditions.shape[-1]
    spatial_shape = boundary_conditions.shape[:-2]
    eye = np.eye(nv)
    flat = boundary_conditions.reshape(-1, nv, nv)
    is_bc = ~np.all(np.isclose(flat, eye), axis=(1, 2))
    return is_bc.reshape(spatial_shape)


# ── 1D helpers ────────────────────────────────────────────────────────────────

def save_simulation_snapshots_1d(
    classical_file: str,
    quantum_file: str,
    iterations: list[int],
    output_path: str,
) -> None:
    df_classical = pd.read_csv(classical_file, header=None)
    df_quantum = pd.read_csv(quantum_file, header=None)

    if max(iterations) >= len(df_classical):
        raise ValueError(f"Max iteration {max(iterations)} exceeds data length")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Classical vs Quantum Simulation Comparison', fontsize=16)

    vmin = min(df_classical.values.min(), df_quantum.values.min())
    vmax = max(df_classical.values.max(), df_quantum.values.max())

    for idx, iter_num in enumerate(iterations):
        ax = axes[idx // 2, idx % 2]
        x = np.arange(len(df_classical.iloc[0]))

        ax.plot(x, df_classical.iloc[iter_num], 'b-', label='Classical', alpha=0.7)
        ax.plot(x, df_quantum.iloc[iter_num], 'r--', label='Quantum', alpha=0.7)

        ax.set_title(f't = {iter_num}', pad=10)
        ax.set_xlabel('Position')
        ax.set_ylabel('Density')
        ax.set_ylim(vmin, vmax)
        ax.grid(True)
        if idx == 0:
            ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Snapshots saved to {output_path}")


def visualize_snapshots_1d(
    filename: str,
    grid_size: int,
    iterations: list[int] = None,
    title: str = 'Simulation Snapshots',
    boundary_nodes: NDArray[np.bool_] | None = None,
) -> None:
    df = pd.read_csv(filename, header=None)

    if iterations is None:
        n = len(df)
        iterations = [0, n // 3, 2 * n // 3, n - 1]

    all_frames = [np.array(df.iloc[i]) for i in iterations]
    vmin = min(frame.min() for frame in all_frames)
    vmax = max(frame.max() for frame in all_frames)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=16)

    x = np.arange(grid_size)

    for idx, iter_num in enumerate(iterations):
        ax = axes[idx // 2, idx % 2]
        frame_data = np.array(df.iloc[iter_num])
        ax.plot(x, frame_data, linewidth=2, marker='o', markersize=4)
        ax.set_ylim(vmin * 0.9, vmax * 1.1)
        ax.set_xlabel('Position')
        ax.set_ylabel('Density')
        ax.set_title(f'Iteration {iter_num}')
        ax.grid(True, alpha=0.3)

    if boundary_nodes is not None:
        labeled = False
        for i in np.where(boundary_nodes)[0]:
            label = 'Boundary' if not labeled else ''
            ax.axvspan(i - 0.5, i + 0.5, color='r', alpha=0.3, label=label)
            labeled = True
        if labeled:
            ax.legend()

    plt.tight_layout()
    plt.show()


def animate_simulation_1d(
    filename: str,
    grid_size: int,
    interval: int = 100,
    title: str = 'Simulation Animation',
    velocity_configs: list = None,
    boundary_nodes: NDArray[np.bool_] | None = None,
) -> HTML:
    df = pd.read_csv(filename, header=None)

    all_data = np.array([df.iloc[i].to_numpy() for i in range(len(df))])
    vmin = all_data.min()
    vmax = all_data.max()

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(grid_size)

    line, = ax.plot([], [], linewidth=2, marker='o', markersize=4)
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(vmin * 0.9, vmax * 1.1)
    ax.set_xlabel('Position')
    ax.set_ylabel('Density')
    ax.grid(True, alpha=0.3)

    if boundary_nodes is not None:
        labeled = False
        for i in np.where(boundary_nodes)[0]:
            label = 'Boundary' if not labeled else ''
            ax.axvspan(i - 0.5, i + 0.5, color='r', alpha=0.3, label=label)
            labeled = True

    quiver = None
    vel_text = None

    ax.legend()
    title_text = ax.set_title(f'{title} - Iteration 0')

    def get_velocity_for_iter(frame):
        if velocity_configs is None:
            return None
        for start, end, vel in velocity_configs:
            if start <= frame < end:
                return vel
        return 0.0

    def init():
        line.set_data([], [])
        return line, title_text

    def animate(frame):
        nonlocal quiver, vel_text

        y = df.iloc[frame].to_numpy()
        line.set_data(x, y)
        title_text.set_text(f'{title} - Iteration {frame}')

        if velocity_configs is not None:
            vel = get_velocity_for_iter(frame)

            if quiver is not None:
                quiver.remove()
            if vel_text is not None:
                vel_text.remove()

            if vel != 0:
                arrow_positions = np.arange(2, grid_size - 2, 3)
                arrow_x = arrow_positions
                arrow_y = np.full_like(arrow_x, vmax * 0.85, dtype=float)
                u = np.full_like(arrow_x, vel * 10, dtype=float)
                v = np.zeros_like(arrow_x, dtype=float)

                quiver = ax.quiver(
                    arrow_x, arrow_y, u, v,
                    color='green', alpha=0.7,
                    scale=1, scale_units='xy',
                    width=0.003, headwidth=4, headlength=5,
                )
                vel_text = ax.text(
                    0.98, 0.95, f'v = {vel:.2f}',
                    transform=ax.transAxes,
                    fontsize=12, color='green',
                    ha='right', va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                )
            else:
                vel_text = ax.text(
                    0.98, 0.95, 'v = 0.00',
                    transform=ax.transAxes,
                    fontsize=12, color='gray',
                    ha='right', va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                )
                quiver = None

        return line, title_text

    anim = FuncAnimation(
        fig, animate, init_func=init,
        frames=len(df), interval=interval,
        blit=False, repeat=True,
    )

    plt.close()
    return HTML(anim.to_jshtml())


# ── 2D helpers ────────────────────────────────────────────────────────────────

def save_simulation_snapshots(
    filename: str,
    dimensions: tuple[int, ...],
    output_path: str,
    iterations: list[int] = None,
) -> None:
    df = pd.read_csv(filename, header=None)

    if iterations is None:
        n = len(df)
        iterations = [0, n // 3, 2 * n // 3, n - 1]

    if max(iterations) >= len(df):
        raise ValueError(f"Max iteration {max(iterations)} exceeds data length {len(df)}")

    all_frames = [np.array(df.iloc[i]).reshape(dimensions, order='F') for i in iterations]
    vmin = min(frame.min() for frame in all_frames)
    vmax = max(frame.max() for frame in all_frames)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Simulation Snapshots', fontsize=16)

    for idx, iter_num in enumerate(iterations):
        ax = axes[idx // 2, idx % 2]
        frame_data = np.array(df.iloc[iter_num]).reshape(dimensions, order='F')
        im = ax.imshow(frame_data.T, cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, label='Density', fraction=0.046, pad=0.04)
        ax.set_title(f't = {iter_num}', pad=10)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Snapshots saved to {output_path}")


def animate_density_evolution(
    dimensions: tuple[int, int],
    filename: str,
    interval: int = 100,
    title: str = 'Density Evolution',
    boundary_nodes: NDArray[np.bool_] | None = None,
) -> HTML:
    df = pd.read_csv(filename, header=None)

    vmin = df.values.min()
    vmax = df.values.max()

    fig, ax = plt.subplots(figsize=(8, 6))
    img = ax.imshow(
        np.zeros(dimensions).T, cmap='viridis', origin='lower',
        vmin=vmin, vmax=vmax,
    )
    plt.colorbar(img, ax=ax, label='Density')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    title_text = ax.set_title(f'{title} - Iteration 0')

    if boundary_nodes is not None:
        overlay = np.full((*dimensions, 4), 0.0)  # RGBA
        overlay[boundary_nodes, :] = [1.0, 0.0, 0.0, 0.3]  # red with alpha
        ax.imshow(overlay.transpose(1, 0, 2), origin='lower', extent=(-0.5, dimensions[0] - 0.5, -0.5, dimensions[1] - 0.5))
        # Invisible artist for the legend entry
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(facecolor='red', alpha=0.3, label='Boundary')], loc='upper right', fontsize=8)

    def init():
        img.set_array(np.zeros(dimensions).T)
        return [img, title_text]

    def update(frame):
        current_density = df.iloc[frame].values.reshape(dimensions, order='F').T
        img.set_array(current_density)
        title_text.set_text(f'{title} - Iteration {frame}')
        return [img, title_text]

    anim = FuncAnimation(
        fig, update, frames=len(df),
        init_func=init, blit=False,
        interval=interval, repeat=True,
    )

    plt.close()
    return HTML(anim.to_jshtml())


# ── 3D helpers ────────────────────────────────────────────────────────────────

def save_isosurface_plots(
    data_file: str,
    sites: tuple[int, int, int],
    output_fig: str,
    timesteps: list[int],
    n_surfaces: int = 3,
    figsize: tuple[int, int] = (15, 12),
    view_angles: tuple[float, float] = (45, 45),
) -> None:
    from skimage.measure import marching_cubes
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    df = pd.read_csv(data_file, header=None)
    if max(timesteps) >= len(df):
        raise ValueError(f"Max timestep {max(timesteps)} exceeds data length {len(df)}")

    fig = plt.figure(figsize=figsize)
    gs = plt.GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 0.15])

    timestep_data = [df.iloc[t].values for t in timesteps]
    data_mins = [data.min() for data in timestep_data]
    data_maxs = [data.max() for data in timestep_data]
    global_min = min(data_mins)
    global_max = max(data_maxs)

    isovalues = np.logspace(np.log10(global_min), np.log10(global_max), n_surfaces + 2)[1:-1]

    cmap = plt.cm.viridis
    colors = [cmap(i) for i in np.linspace(0, 1, n_surfaces)]
    alphas = np.linspace(0.2, 0.8, n_surfaces)

    for idx, timestep in enumerate(timesteps):
        print(f"Time step {timestep}:")
        print(f"  Density range: [{data_mins[idx]:.3f}, {data_maxs[idx]:.3f}]")
        print(f"  Using isosurface levels: {[f'{v:.3f}' for v in isovalues]}")

        row = idx // 2
        col = idx % 2

        ax = fig.add_subplot(gs[row, col], projection='3d')

        density_data = df.iloc[timestep].values
        try:
            density_grid = density_data.reshape(sites, order='F')
        except ValueError as e:
            raise ValueError(f"Could not reshape data of length {len(density_data)} into grid of shape {sites}") from e

        for isovalue, color, alpha in zip(isovalues, colors, alphas):
            try:
                verts, faces, _, _ = marching_cubes(density_grid, isovalue, spacing=(1.0, 1.0, 1.0))
                if len(faces) > 0:
                    mesh = Poly3DCollection(verts[faces])
                    mesh.set_edgecolor('none')
                    mesh.set_facecolor(color)
                    mesh.set_alpha(alpha)
                    ax.add_collection3d(mesh)
            except (ValueError, RuntimeError) as e:
                print(f"  Warning: Could not generate isosurface at {isovalue:.3f}: {e}")
                continue

        ax.set_xlim(0, sites[0])
        ax.set_ylim(0, sites[1])
        ax.set_zlim(0, sites[2])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=view_angles[0], azim=view_angles[1])
        ax.set_title(f'Timestep {timestep}')

    cbar_ax = fig.add_subplot(gs[:, -1])
    norm = plt.Normalize(global_min, global_max)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    ticks = np.linspace(global_min, global_max, 6)
    plt.colorbar(sm, cax=cbar_ax, label='Density', format='%.3f', ticks=ticks)

    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.savefig(output_fig, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
