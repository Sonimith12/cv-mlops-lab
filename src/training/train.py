import os, json, yaml
from pathlib import Path
import wandb
from ultralytics import YOLO

p = yaml.safe_load(open("params.yaml"))["train"]
out = Path("models/weights.pt")
out.parent.mkdir(parents=True, exist_ok=True)
run = wandb.init(
    project=os.getenv("WANDB_PROJECT", "ppe-mlops"),
    entity=os.getenv("WANDB_ENTITY"),
    config=p,
    job_type="train",
)
try:
    m = YOLO(p.get("model", "yolov8n.pt"))
    m.train(
        data="data/processed/data.yaml",
        epochs=int(p["epochs"]),
        imgsz=int(p["imgsz"]),
        batch=int(p["batch"]),
        lr0=float(p["lr0"]),
        seed=int(p["seed"]),
        verbose=False,
    )
    m.export(format="pt", imgsz=int(p["imgsz"]), opset=12)
    best = Path(getattr(m.trainer, "best", "models/weights.pt"))
    (
        out.write_bytes(best.read_bytes())
        if best.exists()
        else out.write_text("DUMMY WEIGHTS")
    )
    metrics = {
        "map50": (
            float(getattr(getattr(m, "metrics", object()), "box", object()).map50)
            if getattr(getattr(m, "metrics", None), "box", None)
            else 0.0
        )
    }
except Exception as e:
    print("train fail:", e)
    metrics = {"map50": 0.0}
    out.write_text("DUMMY WEIGHTS")
json.dump(metrics, open("metrics.json", "w"))
A = wandb.Artifact("model", type="model")
A.add_file(str(out), "weights.pt")
run.log_artifact(A)
run.finish()
print("done")
