"""夹爪可抓性几何核查（桌面估算，不代替实物夹持测试）。

对 9 种积木的常见静止姿态和 100/75/50/25% 四档，计算平行两指夹爪的
所需闭合量、所需张开量和斜面接触所需摩擦系数，并逐个判断候选夹爪能否抓取。
尺寸来自 models/blocks_40mm 的设计定义（README 与 validation_report.json）。

用法：python scripts/gripper_grasp_check.py [--zb 1.0] [--margin 5] [--squeeze 1]
"""
import argparse
import math

# 姿态（100% 尺寸）：名称、高 H、夹持宽 w（夹爪已转到最佳方向）、
# 下降时该方向最大宽度、接触面相对竖直的倾角（度）、说明
POSES = [
    ("圆柱 立放", 40, 40, 40, 0, ""),
    ("圆柱 侧躺", 40, 40, 40, 0, "夹两端面或赤道"),
    ("正方体", 40, 40, 40, 0, ""),
    ("长方体 平放", 20, 30, 30, 0, "夹 30 向"),
    ("长方体 侧立", 30, 40, 40, 0, "夹 40 向"),
    ("长方体 竖立", 40, 30, 30, 0, "夹 30 向"),
    ("三棱柱 立放", 40, 30, 30, 0, "一面一棱"),
    ("三棱柱 侧躺", 30, 40, 40, 0, "夹两端面"),
    ("五棱柱 立放", 40, 36.18, 36.18, 0, "一面一棱"),
    ("五棱柱 侧躺", 36.18, 38.04, 38.04, 0, "夹两侧棱"),
    ("六棱柱 立放", 40, 34.64, 34.64, 0, "对边"),
    ("六棱柱 侧躺", 34.64, 40, 40, 0, "夹两侧棱或两端面"),
    ("圆锥 立放", 40, None, 40, math.degrees(math.atan(20 / 40)), "斜面，贴底夹"),
    ("正四面体", 32.66, None, 34.64, math.degrees(math.atan(1 / (2 * math.sqrt(2)))), "斜面，贴底夹"),
    ("球", 40, 40, 40, 0, "夹赤道"),
]
SCALES = [1.0, 0.75, 0.5, 0.25]

# 候选夹爪：名称、最小闭合间距、最大张开间距（mm）
GRIPPERS = [
    ("A 原版标准爪", 2, 44),
    ("A 宽口爪", 22, 70),
    ("B 改 MG996R(150°)", 4, 51),
    ("B 改 MG996R(180°)", 4, 60),
    ("自设 m1 z22(150°)", 4, 61.6),
]


def cases(zb, margin, squeeze):
    for name, h100, w100, wo100, slope, _ in POSES:
        for s in SCALES:
            h = h100 * s
            if name.startswith("圆锥"):
                w = 40 * s * (1 - zb / h)
            elif name.startswith("正四面体"):
                w = 34.64 * s * (1 - zb / h)
            else:
                w = w100 * s
            yield dict(name=name, scale=s, h=h, w=w,
                       need_open=wo100 * s + 2 * margin,
                       need_close=w - 2 * squeeze,
                       mu=math.tan(math.radians(slope)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zb", type=float, default=1.0, help="指尖下缘离托盘高度 mm")
    ap.add_argument("--margin", type=float, default=5.0, help="单侧余量 mm（定位误差+间隙）")
    ap.add_argument("--squeeze", type=float, default=1.0, help="单侧软垫压缩 mm")
    a = ap.parse_args()
    rows = list(cases(a.zb, a.margin, a.squeeze))
    head = "| 姿态 | 档位 | 高 | 夹持宽 | 需张开≥ | 需闭合≤ | 需 μ≥ | " + " | ".join(g[0] for g in GRIPPERS) + " |"
    print(head)
    print("|" + "---|" * (7 + len(GRIPPERS)))
    for r in rows:
        marks = ["✓" if (c <= r["need_close"] and o >= r["need_open"]) else "✗" for _, c, o in GRIPPERS]
        print(f"| {r['name']} | {int(r['scale'] * 100)}% | {r['h']:.1f} | {r['w']:.1f} | "
              f"{r['need_open']:.1f} | {r['need_close']:.1f} | {r['mu']:.2f} | " + " | ".join(marks) + " |")
    print()
    for g, c, o in GRIPPERS:
        ok = [r for r in rows if c <= r["need_close"] and o >= r["need_open"]]
        # 最大物块（40 mm）下可容忍的单侧定位误差
        tol = (o - 40) / 2 - (a.margin - 3)
        print(f"{g}: 行程 {c}–{o} mm，通过 {len(ok)}/{len(rows)}；40 mm 物块单侧可容忍定位误差约 {tol:.1f} mm")


if __name__ == "__main__":
    main()
