# -*- coding: utf-8 -*-
"""把托盘裁到 X=330，并沿 Y 向中缝拆成两块，接缝做梳齿榫"""
import trimesh, numpy as np

SRC = r"E:/智能分拣/docs/tray_flat_printable.stl"
OUT_A = r"E:/智能分拣/docs/tray_A_bottom.stl"
OUT_B = r"E:/智能分拣/docs/tray_B_top.stl"

TARGET_X = 330.0     # 打印机 X 上限
SEAM = 244.0         # 接缝位置（围边止于 241.5，大孔凸台起于 252，落在空隙里）
DEPTH = 8.0          # 榫齿伸入深度 244~252
WIDTH = 11.0         # 榫齿宽度
GAP = 11.0           # 齿间空隙
CLEAR = 0.15         # 单边装配间隙

m = trimesh.load(SRC, process=True)
print("原始: 包围盒 %s, 体积 %.2f cm3, 水密=%s" %
      (np.round(m.bounds[1]-m.bounds[0], 2), m.volume/1000, m.is_watertight))

# --- 1. X 向裁到 330 ---
cx, cy = 170.0, 170.0
trimbox = trimesh.creation.box(extents=[TARGET_X, 400, 60])
trimbox.apply_translation([cx, cy, 10])
t = trimesh.boolean.intersection([m, trimbox], engine="manifold")
print("裁 X 后: 包围盒 %s, 体积 %.2f cm3, 水密=%s" %
      (np.round(t.bounds[1]-t.bounds[0], 2), t.volume/1000, t.is_watertight))

# --- 2. 生成梳齿区域（A 的榫齿，x 区间列表）---
tabs = []
x = 0.0
on = True
while x < TARGET_X - 0.01:
    w = WIDTH if on else GAP
    if TARGET_X - x < w:
        w = TARGET_X - x
    if on:
        tabs.append((x, x + w))
    x += w
    on = not on
# 保证两端都是榫齿，避免外缘出现缺口
if tabs[-1][1] < TARGET_X - 0.01:
    tabs.append((TARGET_X - WIDTH, TARGET_X))
print("榫齿数量: %d, 首尾: %s ... %s" % (len(tabs), tabs[0], tabs[-1]))

def strip_boxes(ranges, y0, y1, extra=0.0):
    """把 x 区间列表变成一组盒子并合并"""
    bs = []
    for a, b in ranges:
        w = (b - a) + 2*extra
        box = trimesh.creation.box(extents=[w, y1 - y0, 40])
        box.apply_translation([(a+b)/2, (y0+y1)/2, 10])
        bs.append(box)
    return trimesh.boolean.union(bs, engine="manifold")

# --- 3. A 件：y <= SEAM 全部 + 榫齿（y 到 SEAM+DEPTH）---
Y_LO = -50.0
boxA = trimesh.creation.box(extents=[400, SEAM - Y_LO, 60])
boxA.apply_translation([cx, (SEAM + Y_LO) / 2, 10])
A_base = trimesh.boolean.intersection([t, boxA], engine="manifold")

tabSolid = strip_boxes(tabs, SEAM - 0.5, SEAM + DEPTH)
A = trimesh.boolean.union([A_base, trimesh.boolean.intersection([t, tabSolid], engine="manifold")],
                          engine="manifold")

# --- 4. B 件：y >= SEAM 全部，减去榫齿位置（留装配间隙）---
Y_HI = 400.0
boxB = trimesh.creation.box(extents=[400, Y_HI - SEAM, 60])
boxB.apply_translation([cx, (SEAM + Y_HI) / 2, 10])
B_base = trimesh.boolean.intersection([t, boxB], engine="manifold")

notchSolid = strip_boxes(tabs, SEAM - 1.0, SEAM + DEPTH, extra=CLEAR)
B = trimesh.boolean.difference([B_base, notchSolid], engine="manifold")

# --- 5. 归零并校验 ---
for name, part, out in (("A(下)", A, OUT_A), ("B(上)", B, OUT_B)):
    part.apply_translation([-part.bounds[0][0], -part.bounds[0][1], -part.bounds[0][2]])
    sz = part.bounds[1] - part.bounds[0]
    print("%s: %.1f × %.1f × %.1f mm, 体积 %.2f cm3, 水密=%s, 面数=%d" % (
        name, sz[0], sz[1], sz[2], part.volume/1000, part.is_watertight, len(part.faces)))
    part.export(out)
    print("   已写出:", out)

print("\n两块体积合计: %.2f cm3 (裁剪后整体应为 %.2f cm3)" % ((A.volume + B.volume)/1000, t.volume/1000))
print("差值 %.2f cm3 = 榫齿装配间隙" % ((t.volume - A.volume - B.volume)/1000))
