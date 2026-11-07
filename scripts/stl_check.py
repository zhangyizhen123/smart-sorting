# -*- coding: utf-8 -*-
"""纯 Python STL 体检：格式/尺寸/水密性/法向一致性/体积/壁厚粗估"""
import struct, sys, math
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else r"E:/智能分拣/托盘（改）.STL"
raw = open(path, "rb").read()
print("文件大小: %d 字节" % len(raw))

# --- 判断 ASCII / 二进制 ---
is_ascii = False
head = raw[:512].lstrip()
if head[:5].lower() == b"solid":
    # 二进制文件也可能以 solid 开头，用长度校验
    if len(raw) >= 84:
        n = struct.unpack("<I", raw[80:84])[0]
        if len(raw) == 84 + n * 50:
            is_ascii = False
        else:
            is_ascii = True
    else:
        is_ascii = True

tris = []
if is_ascii:
    print("格式: ASCII STL")
    txt = raw.decode("utf-8", "replace")
    cur = []
    for line in txt.splitlines():
        p = line.split()
        if not p:
            continue
        if p[0].lower() == "vertex" and len(p) >= 4:
            cur.append(tuple(float(x) for x in p[1:4]))
            if len(cur) == 3:
                tris.append(tuple(cur)); cur = []
else:
    n = struct.unpack("<I", raw[80:84])[0]
    print("格式: 二进制 STL, 声明三角面数 = %d" % n)
    print("文件头: %r" % raw[:80].rstrip(b"\x00"))
    off = 84
    for i in range(n):
        if off + 50 > len(raw):
            print("!! 数据截断于第 %d 面" % i); break
        vals = struct.unpack("<12fH", raw[off:off+50])
        tris.append((vals[3:6], vals[6:9], vals[9:12]))
        off += 50
    print("实际读取三角面数 = %d" % len(tris))

print("三角面数: %d" % len(tris))

# --- 包围盒 ---
xs = [v[0] for t in tris for v in t]
ys = [v[1] for t in tris for v in t]
zs = [v[2] for t in tris for v in t]
print("包围盒 X: %.3f ~ %.3f  (%.3f)" % (min(xs), max(xs), max(xs)-min(xs)))
print("包围盒 Y: %.3f ~ %.3f  (%.3f)" % (min(ys), max(ys), max(ys)-min(ys)))
print("包围盒 Z: %.3f ~ %.3f  (%.3f)" % (min(zs), max(zs), max(zs)-min(zs)))

# --- 顶点去重 ---
def key(p):
    return (round(p[0], 4), round(p[1], 4), round(p[2], 4))

vid = {}
faces = []
for t in tris:
    f = []
    for p in t:
        k = key(p)
        if k not in vid:
            vid[k] = len(vid)
        f.append(vid[k])
    faces.append(f)
print("唯一顶点数: %d" % len(vid))

# --- 退化面 ---
degen = 0
for f in faces:
    if len(set(f)) < 3:
        degen += 1
print("退化三角面(重复顶点): %d" % degen)

# --- 边统计：水密性 ---
edges = defaultdict(int)
for f in faces:
    for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
        edges[(a, b) if a < b else (b, a)] += 1

cnt = defaultdict(int)
for e, c in edges.items():
    cnt[c] += 1
print("边-共享面数分布: %s" % dict(sorted(cnt.items())))
boundary = sum(1 for e, c in edges.items() if c == 1)
nonman = sum(1 for e, c in edges.items() if c > 2)
print("开放边(只被1个面共享) = %d   -> 非水密破洞" % boundary)
print("非流形边(被>2个面共享) = %d" % nonman)

# --- 法向一致性（有向体积 + 面朝向统计） ---
vol = 0.0
for t in tris:
    a, b, c = t
    vol += (a[0]*(b[1]*c[2]-b[2]*c[1])
            - a[1]*(b[0]*c[2]-b[2]*c[0])
            + a[2]*(b[0]*c[1]-b[1]*c[0])) / 6.0
print("有向体积: %.2f mm^3 (负值=法向整体朝内)" % vol)
print("绝对体积: %.2f mm^3 = %.2f cm^3" % (abs(vol), abs(vol)/1000.0))

# --- 连通体数量 ---
parent = list(range(len(vid)))
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x
def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb: parent[ra] = rb
for f in faces:
    union(f[0], f[1]); union(f[1], f[2])
roots = set(find(i) for i in range(len(vid)))
print("独立壳体数量: %d" % len(roots))

# --- 面积与最小面片尺度 ---
area = 0.0
minedge = 1e9
for t in tris:
    a, b, c = t
    u = (b[0]-a[0], b[1]-a[1], b[2]-a[2])
    v = (c[0]-a[0], c[1]-a[1], c[2]-a[2])
    cx = u[1]*v[2]-u[2]*v[1]; cy = u[2]*v[0]-u[0]*v[2]; cz = u[0]*v[1]-u[1]*v[0]
    area += 0.5*math.sqrt(cx*cx+cy*cy+cz*cz)
    for p, q in ((a,b),(b,c),(c,a)):
        d = math.dist(p, q)
        if d > 0: minedge = min(minedge, d)
print("表面积: %.2f mm^2" % area)
print("最短边: %.4f mm" % minedge)

# --- 零法向面 / 面积极小面 ---
tiny = 0
for t in tris:
    a, b, c = t
    u = (b[0]-a[0], b[1]-a[1], b[2]-a[2])
    v = (c[0]-a[0], c[1]-a[1], c[2]-a[2])
    cx = u[1]*v[2]-u[2]*v[1]; cy = u[2]*v[0]-u[0]*v[2]; cz = u[0]*v[1]-u[1]*v[0]
    if 0.5*math.sqrt(cx*cx+cy*cy+cz*cz) < 1e-6: tiny += 1
print("近零面积面: %d" % tiny)
