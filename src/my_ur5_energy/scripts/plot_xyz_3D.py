#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D 可视化脚本：读取 Bunny_10_parsed.csv 并绘制点云

运行方式：
    python plot_xyz_fixed.py

输出：
    - 弹出交互式 3D 图窗口（可旋转）
    - 自动保存图片 xyz_3d_scatter.png
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ======== 配置区 ========
PATH_CSV = Path("/home/zhengyu/ur_ws/src/my_ur5_energy/scripts/Bunny_10_parsed.csv")  # 你的文件路径
OUTPUT_IMG = PATH_CSV.parent / "xyz_3d_scatter.png"
DS = 0                 # 下采样比例（0 表示不下采样）
POINT_SIZE = 1.0       # 散点大小
TITLE = "3D 点云可视化"  # 图标题

# ======== 工具函数 ========
def pick_xyz_columns(df: pd.DataFrame):
    """自动识别 X/Y/Z 列名"""
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
    """让坐标轴等比例显示"""
    xs, ys, zs = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
    max_range = max(xs[1]-xs[0], ys[1]-ys[0], zs[1]-zs[0])
    mid_x, mid_y, mid_z = np.mean(xs), np.mean(ys), np.mean(zs)
    ax.set_xlim3d(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim3d(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim3d(mid_z - max_range/2, mid_z + max_range/2)

# ======== 主函数 ========
def main():
    if not PATH_CSV.exists():
        raise FileNotFoundError(f"❌ 找不到文件：{PATH_CSV}")

    df = pd.read_csv(PATH_CSV)
    x_col, y_col, z_col = pick_xyz_columns(df)
    xyz = df[[x_col, y_col, z_col]].copy()

    if DS and DS > 1:
        xyz = xyz.iloc[::DS]

    print(f"✅ 成功读取文件: {PATH_CSV}")
    print(f"使用列名: {x_col}, {y_col}, {z_col}")
    print(f"点数量: {len(xyz)}")

    # 绘制 3D 散点图
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
    print(f"📸 已保存图片到：{OUTPUT_IMG}")
    plt.show()

if __name__ == "__main__":
    main()
