"""Render diagrams/*.mmd to SVG and PNG with Mermaid CLI.

    python scripts/render_diagrams.py                 # uses `npx -y @mermaid-js/mermaid-cli`
    MMDC=/path/to/mmdc python scripts/render_diagrams.py

PNG output is rendered at 2x scale on a white background (good for slides and video overlays);
SVG output is transparent and scales cleanly on GitHub.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "diagrams"
THEME = {
    "theme": "base",
    "themeVariables": {
        "primaryColor": "#E8F0F8",
        "primaryBorderColor": "#1F4E79",
        "primaryTextColor": "#1B2733",
        "lineColor": "#51667A",
        "secondaryColor": "#FFF3E6",
        "tertiaryColor": "#F6F8FA",
        "fontFamily": "Segoe UI, Helvetica, Arial, sans-serif",
        "fontSize": "15px",
    },
    "flowchart": {"curve": "basis", "htmlLabels": True},
}


def mmdc_command() -> list[str]:
    if os.environ.get("MMDC"):
        return [os.environ["MMDC"]]
    if shutil.which("mmdc"):
        return ["mmdc"]
    npx = shutil.which("npx") or shutil.which("npx.cmd")
    if not npx:
        sys.exit("Mermaid CLI not found: install Node.js or set MMDC=/path/to/mmdc")
    return [npx, "-y", "@mermaid-js/mermaid-cli"]


def main() -> None:
    cmd = mmdc_command()
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as cfg:
        json.dump(THEME, cfg)
    for src in sorted(SRC.glob("*.mmd")):
        for fmt, extra in (("svg", ["-b", "transparent"]), ("png", ["-b", "white", "-s", "2"])):
            out = src.with_suffix(f".{fmt}")
            subprocess.run(
                [*cmd, "-i", str(src), "-o", str(out), "-c", cfg.name, *extra],
                check=True,
                capture_output=True,
            )
        print(f"rendered {src.stem}.svg / .png")
    os.unlink(cfg.name)


if __name__ == "__main__":
    main()
