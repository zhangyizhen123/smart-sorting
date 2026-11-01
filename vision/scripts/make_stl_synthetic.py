#!/usr/bin/env python3
"""Render a balanced YOLO-seg dataset from the project's binary STL blocks."""

from __future__ import annotations

import argparse
import colorsys
import csv
import math
import random
import struct
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models" / "blocks_40mm"
CLASSES = [
    "cylinder", "cube", "cuboid", "tri_prism", "pent_prism",
    "hex_prism", "cone", "tetrahedron", "sphere",
]
AA = 2


def load_binary_stl(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    if len(raw) < 84:
        raise ValueError(f"Not a binary STL: {path}")
    count = struct.unpack_from("<I", raw, 80)[0]
    if len(raw) != 84 + count * 50:
        raise ValueError(f"Unexpected STL length: {path}")
    dtype = np.dtype([
        ("normal", "<f4", (3,)),
        ("vertices", "<f4", (9,)),
        ("attribute", "<u2"),
    ])
    records = np.frombuffer(raw, dtype=dtype, offset=84)
    triangles = records["vertices"].reshape(-1, 3, 3).astype(np.float64)
    if not np.isfinite(triangles).all() or len(triangles) == 0:
        raise ValueError(f"Invalid mesh coordinates: {path}")
    return triangles


def make_sphere_proxy(triangles: np.ndarray) -> np.ndarray:
    """Use STL dimensions to render a smooth, rotationally invariant sphere."""
    vertices = triangles.reshape(-1, 3)
    lower, upper = vertices.min(axis=0), vertices.max(axis=0)
    center = (lower + upper) / 2.0
    radius = float((upper - lower).max() / 2.0)
    latitudes, longitudes = 24, 48
    points = []
    for j in range(latitudes + 1):
        theta = math.pi * j / latitudes
        for i in range(longitudes):
            phi = 2 * math.pi * i / longitudes
            points.append(center + radius * np.array([
                math.sin(theta) * math.cos(phi),
                math.sin(theta) * math.sin(phi),
                math.cos(theta),
            ]))
    points = np.asarray(points)
    faces = []
    for j in range(latitudes):
        for i in range(longitudes):
            a = j * longitudes + i
            b = j * longitudes + (i + 1) % longitudes
            c = (j + 1) * longitudes + (i + 1) % longitudes
            d = (j + 1) * longitudes + i
            if j > 0:
                faces.append((points[a], points[b], points[c]))
            if j < latitudes - 1:
                faces.append((points[a], points[c], points[d]))
    return np.asarray(faces, dtype=np.float64)


def rotation_matrix(rng: random.Random) -> np.ndarray:
    ax, ay, az = [math.radians(rng.uniform(-180, 180)) for _ in range(3)]
    sx, cx = math.sin(ax), math.cos(ax)
    sy, cy = math.sin(ay), math.cos(ay)
    sz, cz = math.sin(az), math.cos(az)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx


def convex_hull(points: np.ndarray) -> list[tuple[float, float]]:
    unique = sorted({(float(x), float(y)) for x, y in points})
    if len(unique) <= 2:
        return unique

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def random_material(rng: random.Random) -> tuple[int, int, int]:
    if rng.random() < 0.22:
        value = rng.randint(55, 235)
        return value, value, value
    hue = rng.random()
    saturation = rng.uniform(0.28, 0.92)
    value = rng.uniform(0.38, 0.96)
    return tuple(round(channel * 255) for channel in colorsys.hsv_to_rgb(hue, saturation, value))


def camera_basis(elevation: float, azimuth: float):
    elev, az = math.radians(elevation), math.radians(azimuth)
    right = np.array([-math.sin(az), math.cos(az), 0.0])
    up = np.array([-math.sin(elev) * math.cos(az), -math.sin(elev) * math.sin(az), math.cos(elev)])
    toward_camera = np.array([math.cos(elev) * math.cos(az), math.cos(elev) * math.sin(az), math.sin(elev)])
    return right, up, toward_camera


def render_sphere(
    radius: float,
    center: np.ndarray,
    scale: float,
    margin: int,
    right: np.ndarray,
    up: np.ndarray,
    toward_camera: np.ndarray,
    light: np.ndarray,
    material: tuple[int, int, int],
) -> tuple[Image.Image, list[tuple[float, float]]]:
    extent = radius * scale
    side = math.ceil(extent * 2 + 2 * margin)
    yy, xx = np.mgrid[0:side * AA, 0:side * AA].astype(np.float64)
    u = (xx / AA - side / 2) / scale
    v = (side / 2 - yy / AA) / scale
    r2 = u * u + v * v
    inside = r2 <= radius * radius
    depth = np.sqrt(np.maximum(0, radius * radius - r2))
    normals = (
        (u / radius)[..., None] * right
        + (v / radius)[..., None] * up
        + (depth / radius)[..., None] * toward_camera
    )
    diffuse = np.maximum(0, normals @ light)
    shade = 0.48 + 0.52 * diffuse
    rgb = np.zeros((side * AA, side * AA, 4), dtype=np.uint8)
    for channel, value in enumerate(material):
        rgb[..., channel] = np.clip(value * shade, 0, 255).astype(np.uint8)
    rgb[..., 3] = inside.astype(np.uint8) * 255
    layer = Image.fromarray(rgb, "RGBA").resize((side, side), Image.Resampling.LANCZOS)
    hull = [
        (side / 2 + extent * math.cos(2 * math.pi * i / 48),
         side / 2 - extent * math.sin(2 * math.pi * i / 48))
        for i in range(48)
    ]
    return layer, hull


def render_instance(
    name: str,
    triangles: np.ndarray,
    rng: random.Random,
    target_side: float,
    elevation: float,
    azimuth: float,
    pixels_per_mm: float | None = None,
    rotation: np.ndarray | None = None,
    material_override: tuple[int, int, int] | None = None,
    light_override: np.ndarray | None = None,
) -> tuple[Image.Image, list[tuple[float, float]]]:
    rotation = rotation_matrix(rng) if rotation is None else rotation
    transformed = triangles @ rotation.T
    transformed[..., 0] -= (transformed[..., 0].min() + transformed[..., 0].max()) / 2
    transformed[..., 1] -= (transformed[..., 1].min() + transformed[..., 1].max()) / 2
    transformed[..., 2] -= transformed[..., 2].min()

    right, up, toward_camera = camera_basis(elevation, azimuth)
    all_vertices = transformed.reshape(-1, 3)
    u = all_vertices @ right
    v = all_vertices @ up
    depth = all_vertices @ toward_camera
    if name == "sphere":
        lower, upper = all_vertices.min(axis=0), all_vertices.max(axis=0)
        radius = float((upper - lower).max() / 2)
        # Sphere orientation does not affect its projection; use the STL-derived center and radius.
        center = (lower + upper) / 2
        center_depth = float(center @ toward_camera)
        scale = pixels_per_mm if pixels_per_mm is not None else target_side / (2 * radius)
        light = np.asarray([0.25, -0.45, 0.86] if light_override is None else light_override, dtype=np.float64)
        light /= np.linalg.norm(light)
        layer, hull = render_sphere(
            radius, center, scale, 7, right, up, toward_camera, light, material_override if material_override is not None else random_material(rng)
        )
        return layer, hull

    min_u, max_u = float(u.min()), float(u.max())
    min_v, max_v = float(v.min()), float(v.max())
    span = max(max_u - min_u, max_v - min_v)
    scale = pixels_per_mm if pixels_per_mm is not None else target_side / max(span, 1e-6)
    margin = 7
    width = math.ceil((max_u - min_u) * scale + margin * 2)
    height = math.ceil((max_v - min_v) * scale + margin * 2)
    layer = Image.new("RGBA", (width * AA, height * AA), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    face_normals = np.cross(transformed[:, 1] - transformed[:, 0], transformed[:, 2] - transformed[:, 0])
    norm = np.linalg.norm(face_normals, axis=1)
    face_normals /= np.maximum(norm[:, None], 1e-12)
    visible = face_normals @ toward_camera > 1e-8
    visible_ids = np.flatnonzero(visible)
    order = visible_ids[np.argsort(depth.reshape(-1, 3).mean(axis=1)[visible_ids])]
    material = np.asarray(material_override if material_override is not None else random_material(rng), dtype=np.float64)
    light = np.asarray([0.25, -0.45, 0.86] if light_override is None else light_override, dtype=np.float64)
    light /= np.linalg.norm(light)
    for face_id in order:
        face = transformed[face_id]
        face_u, face_v = face @ right, face @ up
        xy = [
            (round((x - min_u) * scale + margin) * AA,
             round((max_v - y) * scale + margin) * AA)
            for x, y in zip(face_u, face_v)
        ]
        diffuse = max(0.0, float(face_normals[face_id] @ light))
        shade = 0.46 + 0.54 * diffuse
        color = tuple(int(np.clip(channel * shade, 0, 255)) for channel in material)
        draw.polygon(xy, fill=(*color, 255))

    layer = layer.resize((width, height), Image.Resampling.LANCZOS)
    projected = np.column_stack((
        (u - min_u) * scale + margin,
        (max_v - v) * scale + margin,
    ))
    hull = convex_hull(projected)
    return layer, hull


def make_background(size: int, rng: random.Random) -> Image.Image:
    palettes = [
        (205, 207, 202), (224, 220, 205), (166, 177, 179),
        (82, 91, 91), (192, 207, 196), (195, 204, 219),
        (236, 235, 229), (133, 117, 102),
    ]
    base = np.asarray(rng.choice(palettes), dtype=np.float32)
    small = np.random.default_rng(rng.randrange(2**32)).normal(0, 1, (18, 18, 3)).astype(np.float32)
    texture = np.asarray(
        Image.fromarray(np.uint8(np.clip((small + 4) * 28, 0, 255))).resize((size, size), Image.Resampling.BILINEAR),
        dtype=np.float32,
    ) / 28 - 4
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    gradient = ((xx / max(1, size - 1) - 0.5) * rng.uniform(-16, 16)
                + (yy / max(1, size - 1) - 0.5) * rng.uniform(-16, 16))
    fine = np.random.default_rng(rng.randrange(2**32)).normal(0, 1.8, (size, size, 1))
    pixels = base[None, None, :] + texture + gradient[..., None] + fine
    return Image.fromarray(np.uint8(np.clip(pixels, 0, 255)), "RGB").convert("RGBA")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--models", type=Path, default=MODEL_DIR)
    parser.add_argument("--per-class", type=int, default=120)
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--instances-min", type=int, default=1)
    parser.add_argument("--instances-max", type=int, default=4)
    parser.add_argument("--elevation-min", type=float, default=35)
    parser.add_argument("--elevation-max", type=float, default=65)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.per_class < 1 or args.image_size < 128:
        raise ValueError("per-class must be positive and image-size must be at least 128")
    if not 1 <= args.instances_min <= args.instances_max <= 4:
        raise ValueError("instances must be between 1 and 4")
    if not 10 <= args.elevation_min <= args.elevation_max <= 85:
        raise ValueError("camera elevations must satisfy 10 <= min <= max <= 85")
    rng = random.Random(args.seed)
    meshes = {name: load_binary_stl(args.models / f"{name}.stl") for name in CLASSES}
    meshes["sphere"] = make_sphere_proxy(meshes["sphere"])

    image_dir = args.out / "images"
    label_dir = args.out / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = args.out / "manifest.csv"
    rows = []
    scene_id = 0

    for target_class_id, target_class in enumerate(CLASSES):
        for _ in range(args.per_class):
            count = rng.randint(args.instances_min, args.instances_max)
            distractors = [i for i in range(len(CLASSES)) if i != target_class_id]
            rng.shuffle(distractors)
            class_ids = [target_class_id] + distractors[:count - 1]
            rng.shuffle(class_ids)
            scene = make_background(args.image_size, rng)
            cols = 1 if count == 1 else 2
            rows_count = math.ceil(count / cols)
            cell_w, cell_h = args.image_size / cols, args.image_size / rows_count
            labels = []
            elevation = rng.uniform(args.elevation_min, args.elevation_max)
            azimuth = rng.uniform(0, 360)

            for slot, class_id in enumerate(class_ids):
                class_name = CLASSES[class_id]
                if count == 1:
                    target_side = rng.uniform(args.image_size * 0.30, args.image_size * 0.50)
                else:
                    target_side = rng.uniform(min(cell_w, cell_h) * 0.34, min(cell_w, cell_h) * 0.62)
                layer, polygon = render_instance(
                    class_name,
                    meshes[class_name],
                    rng,
                    target_side,
                    elevation,
                    azimuth,
                )
                col, row = slot % cols, slot // cols
                center_x = (col + 0.5) * cell_w + rng.uniform(-cell_w * 0.06, cell_w * 0.06)
                center_y = (row + 0.5) * cell_h + rng.uniform(-cell_h * 0.06, cell_h * 0.06)
                x = int(round(center_x - layer.width / 2))
                y = int(round(center_y - layer.height / 2))
                x = max(0, min(args.image_size - layer.width, x))
                y = max(0, min(args.image_size - layer.height, y))

                alpha = layer.getchannel("A")
                shadow_alpha = alpha.filter(ImageFilter.GaussianBlur(radius=4)).point(lambda value: value // 4)
                shadow = Image.new("RGBA", layer.size, (18, 20, 20, 0))
                shadow.putalpha(shadow_alpha)
                scene.alpha_composite(shadow, (min(args.image_size - layer.width, x + 4), min(args.image_size - layer.height, y + 5)))
                scene.alpha_composite(layer, (x, y))

                points = []
                for px, py in polygon:
                    nx = max(0.0, min(1.0, (x + px) / args.image_size))
                    ny = max(0.0, min(1.0, (y + py) / args.image_size))
                    points.extend((nx, ny))
                labels.append((class_id, points))

            stem = f"scene{scene_id:06d}__frame000"
            image_path = image_dir / f"{stem}.jpg"
            label_path = label_dir / f"{stem}.txt"
            scene.convert("RGB").save(image_path, quality=95, subsampling=0)
            with label_path.open("w", encoding="utf-8", newline="\n") as stream:
                for class_id, points in labels:
                    stream.write(str(class_id) + " " + " ".join(f"{value:.6f}" for value in points) + "\n")
            rows.append({
                "image": image_path.name,
                "group_id": f"scene{scene_id:06d}",
                "target_class": target_class,
                "classes": ";".join(CLASSES[class_id] for class_id, _ in labels),
                "instances": len(labels),
                "camera_elevation_deg": f"{elevation:.2f}",
                "camera_azimuth_deg": f"{azimuth:.2f}",
                "seed": args.seed,
            })
            scene_id += 1

    with metadata_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Generated {len(rows)} images and {sum(row['instances'] for row in rows)} instances in {args.out}")
    print(f"Class-balanced target scenes: {args.per_class} per class; seed={args.seed}")


if __name__ == "__main__":
    main()
