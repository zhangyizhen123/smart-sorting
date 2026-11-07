# -*- coding: utf-8 -*-
"""按 长×宽 重新裁板：堵旧孔 -> 裁轮廓 -> 按新轮廓重钻 8 孔"""
import sys, trimesh, numpy as np

src = sys.argv[1]
dst = sys.argv[2]
LX = float(sys.argv[3])   # 长（X）
WY = float(sys.argv[4])   # 宽（Y）
D_HOLE = 5.0
EDGE = 10.5               # 孔心距边
CORNER = 41.0             # 孔距相邻角

m = trimesh.load(src, process=True)
print("原始: %d 面, 水密=%s, 尺寸=%s, 体积=%.2f cm3" % (
    len(m.faces), m.is_watertight,
    np.round(m.bounds[1] - m.bounds[0], 2), m.volume/1000))

OX, OY = m.bounds[0][0], m.bounds[0][1]
SX, SY = m.bounds[1][0] - OX, m.bounds[1][1] - OY

def cyl(cx, cy, d, h=40, zc=5.0):
    c = trimesh.creation.cylinder(radius=d/2, height=h, sections=48)
    c.apply_translation([cx, cy, zc])
    return c

# 旧孔位置（按原板四边偏移）
old = []
for e in (EDGE, SX-EDGE):
    for o in (CORNER, SY-CORNER):
        old.append((e, o))
for e in (EDGE, SY-EDGE):
    for o in (CORNER, SX-CORNER):
        old.append((o, e))
old = list(dict.fromkeys(old))
print("旧孔 %d 个" % len(old))

# 1) 堵旧孔
plugs = [cyl(x, y, D_HOLE) for x, y in old]
filled = trimesh.boolean.union([m] + plugs, engine="manifold")
print("堵孔后: 水密=%s, 体积=%.2f cm3" % (filled.is_watertight, filled.volume/1000))

# 2) 裁轮廓
box = trimesh.creation.box(extents=[LX, WY, 60])
box.apply_translation([OX + SX/2, OY + SY/2, 10])
trim = trimesh.boolean.intersection([filled, box], engine="manifold")
trim.apply_translation([-trim.bounds[0][0], -trim.bounds[0][1], 0])
print("裁后: 水密=%s, 尺寸=%s" % (trim.is_watertight, np.round(trim.bounds[1]-trim.bounds[0], 2)))

# 3) 新孔位置
new = []
for e in (EDGE, LX-EDGE):
    for o in (CORNER, WY-CORNER):
        new.append((e, o))
for e in (EDGE, WY-EDGE):
    for o in (CORNER, LX-CORNER):
        new.append((o, e))
new = list(dict.fromkeys(new))

drills = [cyl(x, y, D_HOLE, zc=trim.bounds[1][2]/2) for x, y in new]
res = trimesh.boolean.difference([trim] + drills, engine="manifold")

print("\n最终: %d 面, 水密=%s" % (len(res.faces), res.is_watertight))
print("尺寸: %s" % np.round(res.bounds[1]-res.bounds[0], 2))
print("体积: %.2f cm3 (原 %.2f, 变化 %+.1f%%)" % (
    res.volume/1000, m.volume/1000, 100*(res.volume-m.volume)/m.volume))
print("孔位:", [(round(x,1), round(y,1)) for x, y in new])

res.export(dst)
print("已写出:", dst)
