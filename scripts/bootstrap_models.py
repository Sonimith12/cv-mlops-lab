# scripts/bootstrap_models.py
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Optional
import sys

try:
    import requests
except ImportError:
    print("requests not installed. Run: pip install requests", file=sys.stderr)
    sys.exit(1)

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

WEIGHTS = {
    # detection
    "yolov8n.pt": {
        "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
        "sha256": None,
    },
    # segmentation
    "yolov8n-seg.pt": {
        "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt",
        "sha256": None,
    },
    # classification
    "yolov8n-cls.pt": {
        "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-cls.pt",
        "sha256": None,
    },
}

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def download_one(name: str, url: str, sha256: Optional[str]) -> None:
    out = MODELS_DIR / name
    if out.exists() and (sha256 is None or _sha256(out) == sha256):
        print(f"[skip] {name} already present.")
        return

    print(f"[downloading] {name} → {out}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with out.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    if sha256:
        got = _sha256(out)
        if got != sha256:
            out.unlink(missing_ok=True)
            raise RuntimeError(f"SHA256 mismatch for {name}: got {got}, want {sha256}")

def main() -> None:
    for fname, meta in WEIGHTS.items():
        download_one(fname, meta["url"], meta.get("sha256"))

if __name__ == "__main__":
    main()
