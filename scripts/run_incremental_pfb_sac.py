#!/usr/bin/env python3
"""Run PFB+SAC generation and automatically save to output/test_XX.png with incremental numbering."""

import argparse
import subprocess
import sys
from pathlib import Path

def get_next_output_path(output_dir: Path, prefix: str = "test_", ext: str = ".png") -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    idx = 1
    while (output_dir / f"{prefix}{idx:02d}{ext}").exists():
        idx += 1
    return output_dir / f"{prefix}{idx:02d}{ext}"

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="a cat", help="Generation prompt")
    parser.add_argument(
        "--style-image",
        default="styles/style_04_kid_crayon_drawing.png",
        help="Style reference image path",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory to save generated images",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pn", default="0.25M")
    args, unknown = parser.parse_known_args()

    project_root = Path(__file__).resolve().parents[1]
    output_path = get_next_output_path(project_root / args.output_dir)

    cmd = [
        sys.executable,
        str(project_root / "scripts" / "run_pfb_sac.py"),
        "--prompt", args.prompt,
        "--style-image", str(project_root / args.style_image if not Path(args.style_image).is_absolute() else args.style_image),
        "--output", str(output_path),
        "--seed", str(args.seed),
        "--pn", str(args.pn),
        *unknown,
    ]
    print(f"Target output: {output_path}")
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
