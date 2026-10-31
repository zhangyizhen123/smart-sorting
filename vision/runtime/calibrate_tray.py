"""Create a tray-plane homography from four measured tray corners."""

from __future__ import annotations

import argparse
from pathlib import Path

from .calibration import calibrate_from_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="JSON with image_points and tray_points_mm")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    transform = calibrate_from_json(args.input, args.output)
    print(f"Wrote homography to {args.output}")
    print("tray corners:", transform.project_many(transform.source_points))


if __name__ == "__main__":
    main()
