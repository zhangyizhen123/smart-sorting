# -*- coding: utf-8 -*-
"""把 STL 绕 X 轴旋转 90°，使厚度方向朝上（Z），便于切片软件识别为平放"""
import struct, sys

src = sys.argv[1]
dst = sys.argv[2]
raw = open(src, "rb").read()
n = struct.unpack("<I", raw[80:84])[0]

hdr = b"tray_v2_rotated_flat"
out = bytearray()
out += hdr + b"\x00" * (80 - len(hdr))
out += struct.pack("<I", n)

off = 84
zmax = 0.0
for i in range(n):
    v = struct.unpack("<12fH", raw[off:off+50])
    nrm = v[0:3]
    pts = [v[3:6], v[6:9], v[9:12]]
    # 绕 X 轴 +90°: (x,y,z) -> (x,-z,y)，再沿 Y 平移 340 归零
    def tf(p):
        return (p[0], -p[2] + 340.0, p[1])
    def tfn(p):
        return (p[0], -p[2], p[1])
    nn = tfn(nrm)
    q = [tf(p) for p in pts]
    out += struct.pack("<12fH", nn[0], nn[1], nn[2],
                       q[0][0], q[0][1], q[0][2],
                       q[1][0], q[1][1], q[1][2],
                       q[2][0], q[2][1], q[2][2], v[12])
    off += 50

open(dst, "wb").write(bytes(out))
print("已写出:", dst, len(out), "字节")
