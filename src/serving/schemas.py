from pydantic import BaseModel


class Health(BaseModel):
    model_config = {"protected_namespaces": ()}
    status: str = "ok"
    model_backend: str = "yolo"
    model_version: str = "unknown"
