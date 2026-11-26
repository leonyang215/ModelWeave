#!/usr/bin/env python3
"""Render the repository's editable Graphviz diagrams as SVG and PNG."""

from pathlib import Path
import shutil
import subprocess
import sys


def main() -> int:
    dot = shutil.which("dot")
    if dot is None:
        print("Graphviz is required: install it so `dot` is on PATH.", file=sys.stderr)
        return 1

    directory = Path(__file__).resolve().parents[1] / "assets" / "diagrams"
    for source in sorted(directory.glob("*.dot")):
        for extension in ("svg", "png"):
            output = source.with_suffix(f".{extension}")
            # Fixed Graphviz coordinates keep control-plane links separate
            # from the main request path across the two documentation rows.
            command = [dot, "-Kneato", "-n2", f"-T{extension}"]
            if extension == "png":
                command.append("-Gdpi=160")
            subprocess.run([*command, str(source), "-o", str(output)], check=True)
            print(output.relative_to(directory.parents[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
