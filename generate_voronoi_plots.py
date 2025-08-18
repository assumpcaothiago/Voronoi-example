"""
Generate animation of Voronoi tessellation for a moving particle.

Author: Thiago Assumpção
"""

# Import necessary modules
from __future__ import annotations
import os
import itertools
from pathlib import Path
from typing import Optional, Union

# Use a non-interactive backend for multiprocessing-safe rendering
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import imageio.v2 as imageio
from scipy.spatial import Voronoi, voronoi_plot_2d

# Select domain size, i.e., x, y in (-domain_size, domain_size)
domain_size = 5.0

# Set speed of particle motion in x, y directions
speed = 0.05

# Generate linear points
npts = 12  # number of points in each dimension
X = np.linspace(-domain_size, domain_size, npts)
Y = np.linspace(-domain_size, domain_size, npts)
lin_points = np.array([[_x, _y] for _x in X for _y in Y])

# Generate random points
rng = np.random.default_rng(seed=1)
rand_points = rng.uniform(-domain_size / 15.0, domain_size / 15.0, (npts * npts, 2))

# Generate sum of random and linear points
points = lin_points + rand_points

# Add corner point for visualization point
corner_point = np.array([-domain_size, -domain_size])

# Overwrite first point
points[0] = corner_point


def generate_voronoi_plot(
    points: np.ndarray,
    n_step: int = 0,
    save_path: Optional[Union[str, os.PathLike]] = None,
) -> Optional[Figure]:
    """
    Generate a Voronoi tessellation plot for a given step.

    Frames are saved with fixed figure size and fixed axis limits so that all
    images have identical pixel dimensions (avoiding GIF assembly errors).

    :param points: Array of shape (N, 2) containing the point coordinates.
    :param n_step: Nonnegative integer time/step index controlling the motion of the first point.
    :param save_path: Optional path to save the rendered figure. If provided, the figure is closed.
    :return: The matplotlib Figure if not saving to disk; otherwise None.
    """
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must be a 2D array with shape (N, 2).")
    if points.shape[0] < 4:
        raise ValueError(
            "points must contain at least 4 points for a stable Voronoi diagram."
        )

    # Add corner point for visualization point
    corner_point = np.array([-domain_size, -domain_size])

    # Set up "velocity" of first point
    vel = np.array([speed, speed])

    # Overwrite first point (operate on the passed-in array)
    points[0] = corner_point + vel * n_step

    # Generate Voronoi tessellation
    vor = Voronoi(points)

    # Use a fixed figure size and dpi so saved PNGs have identical pixel shape
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, point_size=8.0, line_colors="red")

    # Lock axis limits (pad a bit so edges are always visible)
    pad = 0.5
    ax.set_xlim(-domain_size - pad, domain_size + pad)
    ax.set_ylim(-domain_size - pad, domain_size + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"Voronoi tessellation — n_step = {n_step}")

    # Disable autoscale after setting limits (belt & suspenders)
    ax.set_autoscale_on(False)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return None

    return fig


if __name__ == "__main__":
    # Build an animation by rendering frames in parallel
    from concurrent.futures import ProcessPoolExecutor

    out_dir = Path("voronoi_frames")
    out_dir.mkdir(parents=True, exist_ok=True)

    max_steps = round(domain_size * 2.0 / speed)
    steps = list(range(0, max_steps + 1))
    frame_paths = [out_dir / f"frame_{n:03d}.png" for n in steps]

    # Use multiprocessing to render frames in parallel.
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        list(
            executor.map(
                generate_voronoi_plot,
                itertools.repeat(points),
                steps,
                frame_paths,
            )
        )

    # Assemble GIF after frames are rendered (all frames now same shape)
    images = [imageio.imread(p) for p in frame_paths]
    imageio.mimsave("voronoi_animation.gif", images, duration=0.08, loop=0)

    # Optional: also save as MP4 if you have ffmpeg installed
    import matplotlib.animation as animation

    writer = animation.FFMpegWriter(fps=12)
    fig = plt.figure()
    with writer.saving(fig, "voronoi_animation.mp4", dpi=150):
        for p in frame_paths:
            img = plt.imread(p)
            plt.imshow(img)
            plt.axis("off")
            writer.grab_frame()
            plt.cla()
