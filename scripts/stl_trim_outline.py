# -*- coding: utf-8 -*-
"""按指定尺寸裁掉外轮廓（孔位绝对坐标不变），输出可打印 STL"""
import sys, trimesh, numpy as np

src = sys.argv[1]
dst = sys.argv[2]
target = float(sys.argv[3]) if len(sys.argv) > 3 else 334.0

m = trimesh.load(src, process=True)
print("原始: %d 面, 水密=%s, 包围盒=%s" % (len(m.faces), m.is_watertight,
      np.round(m.bounds[1] - m.bounds[0], 2)))
print("原始体积: %.2f cm3" % (m.volume / 1000))

lo, hi = m.bounds
cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
half = target / 2.0

# 裁剪盒：XY 收到 target，Z 留足余量
box = trimesh.creation.box(extents=[target, target, (hi[2] - lo[2]) + 20])
box.apply_translation([cx, cy, (lo[2] + hi[2]) / 2])

cut = trimesh.boolean.intersection([m, box], engine="manifold")
cut.apply_translation([-(cx - half), -(cy - half), -lo[2]])

print("\n裁后: %d 面, 水密=%s" % (len(cut.faces), cut.is_watertight))
print("裁后包围盒: %s" % np.round(cut.bounds[1] - cut.bounds[0], 2))
print("裁后体积: %.2f cm3 (损失 %.2f cm3, %.1f%%)" % (
    cut.volume / 1000, (m.volume - cut.volume) / 1000,
    100 * (m.volume - cut.volume) / m.volume))

cut.export(dst)
print("已写出:", dst)
