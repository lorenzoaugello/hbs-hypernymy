#!/usr/bin/env python3
"""Run the notebook experiment blocks as standalone Python scripts."""
from __future__ import annotations
import argparse
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import os
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))
import hbs_core as core  # noqa: E402

BLOCKS = {
    "A": ["block_a_direction.py"],
    "B": ["block_b_generality.py"],
    "C": ["block_b_generality.py", "block_c_erasure.py"],
    "D": ["block_b_generality.py", "block_d_steering_helpers.py", "block_d_steering.py"],
    "E": ["block_b_generality.py", "block_d_steering_helpers.py", "block_e_ablation.py"],
}
ALL = ["block_a_direction.py", "block_b_generality.py", "block_c_erasure.py",
       "block_d_steering_helpers.py", "block_d_steering.py", "block_e_ablation.py",
       "finalize_ledger.py"]

def execute(filename: str) -> None:
    path = ROOT / "scripts" / "blocks" / filename
    code = compile(path.read_text(encoding="utf-8"), str(path), "exec")
    exec(code, core.__dict__)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("part", choices=["A", "B", "C", "D", "E", "all"])
    parser.add_argument("--figures", action="store_true", help="create the combined diagnostic figure after all blocks")
    args = parser.parse_args()
    (ROOT / "work" / "followup").mkdir(parents=True, exist_ok=True)
    sequence = ALL if args.part == "all" else BLOCKS[args.part]
    # Avoid rerunning B when the selected part is B itself.
    for filename in sequence:
        execute(filename)
    if args.figures:
        if args.part != "all":
            parser.error("--figures is available only with part=all")
        execute("make_figures.py")

if __name__ == "__main__":
    main()
