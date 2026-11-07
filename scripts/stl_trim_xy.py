# -*- coding: utf-8 -*-
"""按 X/Y 分别裁外轮廓（孔位绝对坐标不变），输出可打印 STL
用法: python stl_trim_xy.py 源.stl 目标.stl 330 320
"""
import sys, trimesh, numpy as np

src = sys.argv[1]
dst = sys.argv[2]
tx = float(sys.argv[3])
ty = float(sys.argv[4])

m = trimesh.load(src, process=True)
lo, hi = m.bounds
print("原始: 包围盒 %s, 体积 %.2f cm3, 水密=%s" %
      (np.round(hi - lo, 2), m.volume / 1000, m.is_watertight))

cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
box = trimesh.creation.box(extents=[tx, ty, (hi[2] - lo[2]) + 20])
box.apply_translation([cx, cy, (lo[2] + hi[2]) / 2])

cut = trimesh.boolean.intersection([m, box], engine="manifold")
cut.apply_translation([-(cx - tx / 2), -(cy - ty / 2), -lo[2]])

print("\n裁后: 包围盒 %s, 体积 %.2f cm3, 水密=%s, 面数=%d" %
      (np.round(cut.bounds[1] - cut.bounds[0], 2), cut.volume / 1000,
       cut.is_watertight, len(cut.faces)))
print("材料损失: %.2f cm3 (%.1f%%)" % ((m.volume - cut.volume) / 1000,
      100 * (m.volume - cut.volume) / m.volume))

cut.export(dst)
print("已写出:", dst)
