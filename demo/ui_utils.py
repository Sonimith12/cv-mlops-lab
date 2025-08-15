from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List, Tuple

COLOR = {
    "person": (0, 180, 255),
    "helmet": (0, 200, 0),
    "vest": (255, 165, 0),
    "_default": (255, 0, 0),
}


def _font():
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def draw_boxes(
    pil_img: Image.Image, prediction: Dict, conf_thr: float = 0.35
) -> Image.Image:
    """Draw YOLO-style boxes. Expect prediction['bboxes'] as list of dicts {x1,y1,x2,y2,conf,cls}."""
    img = pil_img.copy()
    draw = ImageDraw.Draw(img)
    font = _font()
    for b in prediction.get("bboxes", []):
        conf = float(b.get("conf", 0.0))
        if conf < conf_thr:
            continue
        x1, y1, x2, y2 = map(int, (b["x1"], b["y1"], b["x2"], b["y2"]))
        cls = str(b.get("cls", ""))
        color = COLOR.get(cls, COLOR["_default"])
        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=3)
        label = f"{cls} {conf:.2f}"
        if font:
            left, top, right, bottom = draw.textbbox((x1, y1), label, font=font)
            pad = 2
            draw.rectangle(
                [(left - pad, top - pad), (right + pad, bottom + pad)], fill=color
            )
            draw.text((x1, y1), label, fill=(0, 0, 0), font=font)
        else:
            draw.text((x1, y1), label, fill=color)
    return img


def overlay_masks(
    pil_img: Image.Image, masks: List[Dict], alpha: float = 0.45
) -> Image.Image:
    """
    Draw semi-transparent polygons over the image.
    Expect each mask item like: {"points": [[x,y],...], "cls": "...", "conf": 0.9}
    """
    base = pil_img.copy().convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    for m in masks:
        pts = m.get("points", [])
        if len(pts) < 3:
            continue
        cls = str(m.get("cls", ""))
        color = COLOR.get(cls, COLOR["_default"])
        col = (*color, int(255 * alpha))
        # polygon fill + border
        draw.polygon(
            [tuple(map(int, p)) for p in pts], fill=col, outline=color, width=2
        )
    out = Image.alpha_composite(base, overlay).convert("RGB")
    return out
