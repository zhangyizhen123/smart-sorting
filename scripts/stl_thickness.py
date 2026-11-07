# -*- coding: utf-8 -*-
"""沿 Y 轴射线投射，统计厚度分布，找出薄壁/零厚区域"""
import struct, sys, math

path = sys.argv[1] if len(sys.argv) > 1 else r"E:/智能分拣/托盘（改）.STL"
raw = open(path, "rb").read()
n = struct.unpack("<I", raw[80:84])[0]
tris = []
off = 84
for i in range(n):
    v = struct.unpack("<12fH", raw[off:off+50])
    tris.append((v[3:6], v[6:9], v[9:12]))
    off += 50

xs = [v[0] for t in tris for v in t]; ys = [v[1] for t in tris for v in t]; zs = [v[2] for t in tris for v in t]
X0,X1 = min(xs),max(xs); Y0,Y1 = min(ys),max(ys); Z0,Z1 = min(zs),max(zs)

def cast(x, z):
    """返回沿 +Y 方向所有交点的 y 值（已排序）"""
    hits = []
    for a, b, c in tris:
        # 用 X-Z 平面上的重心坐标判断点是否在三角形投影内
        d = (b[0]-a[0])*(c[2]-a[2]) - (c[0]-a[0])*(b[2]-a[2])
        if abs(d) < 1e-12: continue
        l1 = ((b[0]-x)*(c[2]-z) - (c[0]-x)*(b[2]-z)) / d
        l2 = ((c[0]-x)*(a[2]-z) - (a[0]-x)*(c[2]-z)) / d
        l3 = 1 - l1 - l2
        if l1 < -1e-9 or l2 < -1e-9 or l3 < -1e-9: continue
        hits.append(l1*a[1] + l2*b[1] + l3*c[1])
    hits.sort()
    # 合并重合交点
    merged = []
    for h in hits:
        if not merged or abs(h-merged[-1]) > 1e-4:
            merged.append(h)
    return merged

N = 34  # 34x34 采样
thick = {}
gaps = []
for i in range(N):
    x = X0 + (X1-X0)*(i+0.5)/N
    for j in range(N):
        z = Z0 + (Z1-Z0)*(j+0.5)/N
        h = cast(x, z)
        if len(h) < 2:
            thick[(i,j)] = 0.0
            gaps.append((x, z))
            continue
        t = 0.0
        for k in range(0, len(h)-1, 2):
            t += h[k+1]-h[k]
        thick[(i,j)] = t

vals = sorted(v for v in thick.values() if v > 1e-6)
print("采样点 %d 个，其中实心 %d 个，空(无料) %d 个" % (N*N, len(vals), N*N-len(vals)))
if vals:
    print("厚度 min %.2f / 中位 %.2f / 平均 %.2f / max %.2f mm" % (
        vals[0], vals[len(vals)//2], sum(vals)/len(vals), vals[-1]))

# 厚度直方图
bins = {}
for v in vals:
    b = round(v*2)/2
    bins[b] = bins.get(b,0)+1
print("\n厚度分布(mm -> 采样点数):")
for b in sorted(bins):
    print("  %5.1f : %3d %s" % (b, bins[b], '#'*min(50,bins[b]//2)))

# 打印 ASCII 厚度图（俯视 X-Z）
print("\n俯视厚度图（字符=厚度档位，. 为空，1~9 递增）")
for j in range(N-1, -1, -1):
    row = ""
    for i in range(N):
        t = thick[(i,j)]
        if t <= 1e-6: row += "."
        else:
            row += str(min(9, max(1, int(t/3)+1)))
    print("  " + row)
