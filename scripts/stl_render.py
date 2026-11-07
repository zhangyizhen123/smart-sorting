# -*- coding: utf-8 -*-
"""纯 Python 正交投影渲染 STL -> PNG（三视图 + 等轴测）"""
import struct, sys, math, zlib

path = sys.argv[1] if len(sys.argv) > 1 else r"E:/智能分拣/托盘（改）.STL"
out = sys.argv[2] if len(sys.argv) > 2 else r"E:/智能分拣/docs/tray_preview.png"
raw = open(path, "rb").read()
n = struct.unpack("<I", raw[80:84])[0]
tris = []
off = 84
for i in range(n):
    v = struct.unpack("<12fH", raw[off:off+50])
    tris.append((v[3:6], v[6:9], v[9:12]))
    off += 50

W, H = 1000, 1000
PANEL = 500

def render(tris, ax_u, ax_v, ax_d, sign, w, h, ox, oy, scale, u0, v0, d0):
    img = [[(250,250,252)]*w for _ in range(h)]
    zbuf = [[-1e18]*w for _ in range(h)]
    # 光源方向
    L = (0.4, 0.75, -0.5)
    ln = math.sqrt(sum(x*x for x in L)); L = tuple(x/ln for x in L)
    for a, b, c in tris:
        u = [b[i]-a[i] for i in range(3)]
        v = [c[i]-a[i] for i in range(3)]
        nx = u[1]*v[2]-u[2]*v[1]; ny = u[2]*v[0]-u[0]*v[2]; nz = u[0]*v[1]-u[1]*v[0]
        m = math.sqrt(nx*nx+ny*ny+nz*nz)
        if m == 0: continue
        nx, ny, nz = nx/m, ny/m, nz/m
        shade = abs(nx*L[0]+ny*L[1]+nz*L[2])
        shade = 0.35 + 0.65*shade
        base = (60, 120, 200)
        col = (int(base[0]*shade+200*(1-shade)*0.35),
               int(base[1]*shade+200*(1-shade)*0.35),
               int(base[2]*shade+200*(1-shade)*0.35))
        pts = []
        for p in (a, b, c):
            U = (p[ax_u]-u0)*scale + ox
            V = (p[ax_v]-v0)*scale + oy
            D = (p[ax_d]-d0)*sign
            pts.append((U, V, D))
        minx = max(0, int(min(p[0] for p in pts)))
        maxx = min(w-1, int(max(p[0] for p in pts))+1)
        miny = max(0, int(min(p[1] for p in pts)))
        maxy = min(h-1, int(max(p[1] for p in pts))+1)
        (x1,y1,z1),(x2,y2,z2),(x3,y3,z3) = pts
        den = (y2-y3)*(x1-x3) + (x3-x2)*(y1-y3)
        if abs(den) < 1e-12: continue
        for yy in range(miny, maxy+1):
            for xx in range(minx, maxx+1):
                l1 = ((y2-y3)*(xx-x3) + (x3-x2)*(yy-y3))/den
                l2 = ((y3-y1)*(xx-x3) + (x1-x3)*(yy-y3))/den
                l3 = 1-l1-l2
                if l1 < -0.002 or l2 < -0.002 or l3 < -0.002: continue
                z = l1*z1 + l2*z2 + l3*z3
                if z > zbuf[yy][xx]:
                    zbuf[yy][xx] = z
                    img[yy][xx] = col
    return img

def compose(panels, w=W, h=H):
    img = [[(255,255,255)]*w for _ in range(h)]
    for (pi, pj), p in panels:
        for y in range(PANEL):
            for x in range(PANEL):
                img[pj*PANEL+y][pi*PANEL+x] = p[y][x]
    return img

def save_png(img, path):
    h = len(img); w = len(img[0])
    raw_data = b''
    for row in img:
        raw_data += b'\x00' + b''.join(bytes(px) for px in row)
    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag+data) & 0xffffffff)
    png = b'\x89PNG\r\n\x1a\n'
    png += chunk(b'IHDR', struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    png += chunk(b'IDAT', zlib.compress(raw_data, 6))
    png += chunk(b'IEND', b'')
    open(path, "wb").write(png)

# 包围盒
xs = [v[0] for t in tris for v in t]; ys = [v[1] for t in tris for v in t]; zs = [v[2] for t in tris for v in t]
print("bbox X %.2f-%.2f  Y %.2f-%.2f  Z %.2f-%.2f" % (min(xs),max(xs),min(ys),max(ys),min(zs),max(zs)))

P = PANEL - 40
panels = []

# 左上：主视 X-Z（当前"正面"）
s1 = P/max(max(xs)-min(xs), max(zs)-min(zs))
panels.append(((0,0), render(tris, 0, 2, 1, 1, PANEL, PANEL, 20, 20, s1, min(xs), min(zs), max(ys))))

# 右上：俯视 X-Y
s2 = P/max(max(xs)-min(xs), max(ys)-min(ys))
panels.append(((1,0), render(tris, 0, 1, 2, 1, PANEL, PANEL, 20, 20, s2, min(xs), min(ys), max(zs))))

# 左下：侧视 Z-Y
s3 = P/max(max(zs)-min(zs), max(ys)-min(ys))
panels.append(((0,1), render(tris, 2, 1, 0, -1, PANEL, PANEL, 20, 20, s3, min(zs), min(ys), max(xs))))

# 右下：等轴测（绕Y轴转35°再俯视30°）
ca, sa = math.cos(math.radians(35)), math.sin(math.radians(35))
cb, sb = math.cos(math.radians(28)), math.sin(math.radians(28))
rot = []
for a, b, c in tris:
    t = []
    for p in (a, b, c):
        x = p[0]-170; y = p[1]-10; z = p[2]-170
        x1 =  x*ca - z*sa
        z1 =  x*sa + z*ca
        y1 =  y*cb - z1*sb
        z2 =  y*sb + z1*cb
        t.append((x1, y1, z2))
    rot.append(tuple(t))
rx = [p[0] for t in rot for p in t]; ry = [p[1] for t in rot for p in t]; rz = [p[2] for t in rot for p in t]
s4 = P/max(max(rx)-min(rx), max(rz)-min(rz))
panels.append(((1,1), render(rot, 0, 2, 1, 1, PANEL, PANEL, 20, 20, s4, min(rx), min(rz), max(ry))))

save_png(compose(panels), out)
print("已输出:", out)
