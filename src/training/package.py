from pathlib import Path, shutil

src = Path("models/weights.pt")
dst = Path("models/serving")
dst.mkdir(parents=True, exist_ok=True)
shutil.copy2(src, dst / "model.pt") if src.exists() else print("no weights")
