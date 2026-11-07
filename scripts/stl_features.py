# -*- coding: utf-8 -*-
"""提取 STL 中的垂直壁（孔、槽、外轮廓、围边），输出每个特征的尺寸与位置"""
import struct, sys, math
from collections import defaultdict

path = sys.argv[1]
raw = open(path, "rb").read()
n = struct.unpack("<I", raw[80:84])[0]
tris = []
off = 84
for i in range(n):
    v = struct.unpack("<12fH", raw[off:off+50])
    tris.append((v[3:6], v[6:9], v[9:12]))
    off += 50

# 收集垂直边（XY 相同、Z 不同）
def key(p):
    return (round(p[0], 3), round(p[1], 3), round(p[2], 3))

vedges = []
for t in tris:
    for i in range(3):
        a, b = t[i], t[(i+1) % 3]
        if abs(a[0]-b[0]) < 1e-4 and abs(a[1]-b[1]) < 1e-4 and abs(a[2]-b[2]) > 1e-4:
            vedges.append((a, b))

# 按 XY 位置聚类（同一条垂直边的上下端）
pts = defaultdict(list)
for a, b in vedges:
    pts[(round(a[0], 2), round(a[1], 2))].append((min(a[2], b[2]), max(a[2], b[2])))

# 连通聚类：XY 距离 < 3mm 视为同一特征
keys = list(pts.keys())
idx = {k: i for i, k in enumerate(keys)}
parent = list(range(len(keys)))
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x
def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb: parent[ra] = rb

for i in range(len(keys)):
    for j in range(i+1, len(keys)):
        if math.dist(keys[i], keys[j]) < 3.0:
            union(i, j)

groups = defaultdict(list)
for i, k in enumerate(keys):
    groups[find(i)].append(k)

feats = []
for g, ks in groups.items():
    xs = [k[0] for k in ks]; ys = [k[1] for k in ks]
    w = max(xs)-min(xs); h = max(ys)-min(ys)
    zlo = min(min(pts[k])[0] for k in ks)
    zhi = max(max(pts[k])[1] for k in ks)
    cx = (max(xs)+min(xs))/2; cy = (max(ys)+min(ys))/2
    feats.append(dict(cx=cx, cy=cy, w=w, h=h, zlo=zlo, zhi=zhi, n=len(ks)))

feats.sort(key=lambda f: -max(f['w'], f['h']))
print("共 %d 个垂直壁特征\n" % len(feats))
print("%-4s %-10s %-10s %-9s %-9s %-8s" % ("序", "中心XY", "尺寸(W×H)", "Z范围", "类型", "顶点数"))
for i, f in enumerate(feats):
    if f['w'] > 200 and f['h'] > 200:
        t = "外轮廓"
    elif abs(f['w']-f['h']) < 2 and 3 < f['w'] < 70:
        t = "圆孔 Ø%.1f" % f['w']
    elif f['w'] < 20 and f['h'] > 50:
        t = "长槽"
    elif f['w'] < 20 and f['h'] < 20:
        t = "小孔"
    else:
        t = "其他"
    print("%-4d (%6.1f,%6.1f) %5.1f×%-6.1f %4.1f~%-4.1f %-10s %d" % (
        i, f['cx'], f['cy'], f['w'], f['h'], f['zlo'], f['zhi'], t, f['n']))

# 外轮廓边界
out = [f for f in feats if f['w'] > 200 and f['h'] > 200]
if out:
    o = out[0]
    X0 = o['cx'] - o['w']/2; X1 = o['cx'] + o['w']/2
    Y0 = o['cy'] - o['h']/2; Y1 = o['cy'] + o['h']/2
    print("\n外轮廓: X %.1f~%.1f  Y %.1f~%.1f" % (X0, X1, Y0, Y1))
    print("\n各特征到外轮廓的最小距离:")
    for i, f in enumerate(feats):
        if f is o: continue
        if f['w'] > 200: continue
        d = min(f['cx']-f['w']/2-X0, X1-(f['cx']+f['w']/2),
                f['cy']-f['h']/2-Y0, Y1-(f['cy']+f['h']/2))
        print("  %2d (%6.1f,%6.1f) %-10s 距边 %5.1f mm" % (i, f['cx'], f['cy'],
              "Ø%.1f" % f['w'] if abs(f['w']-f['h'])<2 else "%.0f×%.0f"%(f['w'],f['h']), d))
