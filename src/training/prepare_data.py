from pathlib import Path
import random
from PIL import Image, ImageDraw
import yaml

out = Path("data/processed")
(out / "images").mkdir(parents=True, exist_ok=True)
(out / "labels").mkdir(parents=True, exist_ok=True)
with open(out / "data.yaml", "w") as f:
    yaml.safe_dump(
        {
            "path": str(out.resolve()),
            "train": "images",
            "val": "images",
            "names": {0: "person", 1: "helmet", 2: "vest"},
        },
        f,
    )
for i in range(10):
    w, h = 640, 480
    img = Image.new("RGB", (w, h), (20, 20, 20))
    d = ImageDraw.Draw(img)
    d.rectangle([200, 120, 380, 360], outline=(200, 200, 200), width=3)
    img.save(out / "images" / f"img_{i:04d}.jpg", quality=90)
print("data ready")
