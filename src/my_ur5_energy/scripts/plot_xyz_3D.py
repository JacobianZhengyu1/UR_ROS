#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D visualization script: Read Bunny_10_parsed.csv and plot point cloud.

Usage:
    python plot_xyz_fixed.py

Output:
    - Interactive 3D visualization window (rotatable)
    - Automatically saves image: xyz_3d_scatter.png
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ======== Configuration ========
PATH_CSV = Path("/home/zhengyu/ur_ws/src/my_ur5_energy/scripts/Bunny_10_parsed.csv")  # Path to your CSV file
OUTPUT_IMG = PATH_CSV.parent / "xyz_3d_scatter.png"
DS = 0                 # Down-sampling ratio (0 means no down-sampling)
POINT_SIZE = 1.0       # Scatter point size
TITLE = "3D Point Cloud Visualization"  # Plot title

# ======== Utility Functions ========
def pick_xyz_columns(df: pd.DataFrame):
    """Automatically detect X/Y/Z column names."""
    candidates = {c.lower(): c for c in df.columns}

    def find_col(name):
        for key, orig in candidates.items():
            if name in key:
                return orig
        return None

    x = find_col("x") or list(df.columns)[0]
    y = find_col("y") or list(df.columns)[1]
    z = find_col("z") or list(df.columns)[2]
    return x, y, z

def set_axes_equal(ax):
    """Force the 3D axes to use equal scaling."""
    xs, ys, zs = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
    max_range = max(xs[1]-xs[0], ys[1]-ys[0], zs[1]-zs[0])
    mid_x, mid_y, mid_z = np.mean(xs), np.mean(ys), np.mean(zs)
    ax.set_xlim3d(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim3d(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim3d(mid_z - max_range/2, mid_z + max_range/2)

# ======== Main Function ========
def main():
    if not PATH_CSV.exists():
        raise FileNotFoundError(f"❌ File not found: {PATH_CSV}")

    df = pd.read_csv(PATH_CSV)
    x_col, y_col, z_col = pick_xyz_columns(df)
    xyz = df[[x_col, y_col, z_col]].copy()

    if DS and DS > 1:
        xyz = xyz.iloc[::DS]

    print(f"✅ Successfully loaded file: {PATH_CSV}")
    print(f"Using columns: {x_col}, {y_col}, {z_col}")
    print(f"Number of points: {len(xyz)}")

    # Plot 3D scatter
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xyz[x_col], xyz[y_col], xyz[z_col], s=POINT_SIZE, color="orange", alpha=0.8)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_zlabel(z_col)
    ax.set_title(TITLE)
    set_axes_equal(ax)

    plt.tight_layout()
    plt.savefig(OUTPUT_IMG, dpi=200)
    print(f"📸 Image saved to: {OUTPUT_IMG}")
    plt.show()

if __name__ == "__main__":
    main()
