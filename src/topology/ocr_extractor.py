"""
Extract text labels from device bounding-box crops using Tesseract OCR.
"""

import re
import numpy as np

try:
    import pytesseract
    from PIL import Image
    _OCR_AVAILABLE = True
except ImportError:
    _OCR_AVAILABLE = False


_TESSERACT_CONFIG = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_/."


def extract_label(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
    padding: int = 6,
) -> str:
    """Run Tesseract on a single device crop and return the cleaned label.

    Parameters
    ----------
    image:   Full BGR image (numpy array).
    bbox:    (x1, y1, x2, y2) bounding box in pixel coordinates.
    padding: Extra pixels to include around the bounding box.

    Returns
    -------
    Cleaned label string, or empty string if OCR is unavailable or fails.
    """
    if not _OCR_AVAILABLE:
        return ""

    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]
    crop = image[
        max(0, y1 - padding): min(h, y2 + padding),
        max(0, x1 - padding): min(w, x2 + padding),
    ]
    if crop.size == 0:
        return ""

    pil_crop = Image.fromarray(crop[..., ::-1])  # BGR → RGB
    try:
        raw = pytesseract.image_to_string(pil_crop, config=_TESSERACT_CONFIG)
    except Exception:
        return ""

    return _clean(raw)


def _clean(text: str) -> str:
    text = text.strip().upper()
    text = re.sub(r"[^A-Z0-9\-_/.]", "", text)
    return text[:32]


def extract_all_labels(
    image: np.ndarray,
    detections: list[dict],
    padding: int = 6,
) -> list[dict]:
    """Add an ``ocr_label`` field to each detection dict in-place and return the list."""
    for det in detections:
        det["ocr_label"] = extract_label(image, det["bbox"], padding)
    return detections
