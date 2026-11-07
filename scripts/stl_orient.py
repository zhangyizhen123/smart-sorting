# -*- coding: utf-8 -*-
"""分析 STL 的朝向、平面法向分布、截面轮廓，判断是否适合直接打印"""
import struct, sys, math
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else r"E:/智能分拣/托盘（改）.STL"
raw = open(path, "rb").read()
n = struct.unpack("<I", raw[80:84])[0]
tris = []
off = 84
for i in range(n):
    v = struct.unpack("<12fH", raw[off:off+50])
    tris.append((v[3:6], v[6:9], v[9:12]))
    off += 50

# 面积按法向主轴归类
area_by_axis = defaultdict(float)
total = 0.0
for a, b, c in tris:
    u = [b[i]-a[i] for i in range(3)]
    v = [c[i]-a[i] for i in range(3)]
    cx = u[1]*v[2]-u[2]*v[1]; cy = u[2]*v[0]-u[0]*v[2]; cz = u[0]*v[1]-u[1]*v[0]
    m = math.sqrt(cx*cx+cy*cy+cz*cz)
    if m == 0: continue
    ar = 0.5*m
    total += ar
    nx, ny, nz = abs(cx)/m, abs(cy)/m, abs(cz)/m
    if nz > 0.9: area_by_axis['Z'] += ar
    elif ny > 0.9: area_by_axis['Y'] += ar
    elif nx > 0.9: area_by_axis['X'] += ar
    else: area_by_axis['斜面'] += ar

print("总面积 %.1f mm^2" % total)
for k in ('X','Y','Z','斜面'):
    if area_by_axis[k]:
        print("  法向≈%s 的面积: %9.1f mm^2  (%5.1f%%)" % (k, area_by_axis[k], 100*area_by_axis[k]/total))

# 各轴切片：统计每个坐标层上的顶点数，判断厚度方向
print("\n--- 各轴分层（顶点在该轴上取值的分布）---")
for ax, name in ((0,'X'),(1,'Y'),(2,'Z')):
    vals = sorted(set(round(v[ax],3) for t in tris for v in t))
    print("%s: 不同坐标层数=%d, 前10层=%s" % (name, len(vals), [round(x,2) for x in vals[:10]]))

# 沿 Y 轴（20mm 方向）的截面分析：每个 Y 层上的 X/Z 范围
print("\n--- 沿 Y 分层截面（判断是否为薄板/托盘）---")
layers = defaultdict(list)
for t in tris:
    for v in t:
        layers[round(v[1],2)].append((v[0], v[2]))
for y in sorted(layers):
    pts = layers[y]
    xs = [p[0] for p in pts]; zs = [p[1] for p in pts]
    print("Y=%6.2f  点数=%4d  X范围[%7.2f,%7.2f]  Z范围[%7.2f,%7.2f]" % (y, len(pts), min(xs), max(xs), min(zs), max(zs)))
