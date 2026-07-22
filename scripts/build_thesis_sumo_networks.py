#!/usr/bin/env python
"""Build generated thesis SUMO `.net.xml` files in a Colab-tolerant way."""
from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.generate_thesis_sumo import main as generate_thesis_sumo
from src.environment.sumo_env import _check_call_accepting_colab_sigsegv


def build_network(node_file: Path, edge_file: Path, output_file: Path) -> None:
    netconvert = shutil.which("netconvert")
    if not netconvert:
        raise RuntimeError("netconvert binary not found on PATH; install SUMO tools first")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        netconvert,
        "--node-files", str(node_file),
        "--edge-files", str(edge_file),
        "--tls.guess", "true",
        "--output-file", str(output_file),
    ]
    _check_call_accepting_colab_sigsegv(cmd, output_file)


def main() -> None:
    generate_thesis_sumo()
    base = Path("sumo/thesis_ch4_ch5")
    build_network(
        base / "two_intersection/two_intersection.nod.xml",
        base / "two_intersection/two_intersection.edg.xml",
        base / "two_intersection/two_intersection.net.xml",
    )
    build_network(
        base / "nine_intersection/nine_intersection.nod.xml",
        base / "nine_intersection/nine_intersection.edg.xml",
        base / "nine_intersection/nine_intersection.net.xml",
    )
    print("Built thesis SUMO .net.xml files under sumo/thesis_ch4_ch5")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc
