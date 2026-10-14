import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np

from app.preprocessing import preprocess_image

logger = logging.getLogger("tetrax.ocr")

SUPPORTED_LANGUAGES_INFO: Dict[str, Dict[str, str]] = {
    "en": {"name": "English", "script": "Latin", "sample": "Invoice Number: INV-1045"},
    "hi": {"name": "Hindi", "script": "Devanagari", "sample": "कर बीजक संख्या"},
    "mr": {"name": "Marathi", "script": "Devanagari", "sample": "पावती क्रमांक"},
    "ta": {"name": "Tamil", "script": "Tamil", "sample": "விலைப்பட்டியல் எண்"},
    "te": {"name": "Telugu", "script": "Telugu", "sample": "ఇన్వాయిస్ సంఖ్య"},
    "ur": {"name": "Urdu", "script": "Perso-Arabic", "sample": "انوائس نمبر"},
    "bn": {"name": "Bengali", "script": "Bengali", "sample": "চালান নম্বর"},
    "gu": {"name": "Gujarati", "script": "Gujarati", "sample": "બીલ નંબર"},
    "kn": {"name": "Kannada", "script": "Kannada", "sample": "ಇನ್‌ವಾಯ್ಸ್ ಸಂಖ್ಯೆ"},
    "ml": {"name": "Malayalam", "script": "Malayalam", "sample": "ഇൻവോയ്സ് നമ്പർ"},
    "or": {"name": "Odia", "script": "Odia", "sample": "ଚାଲାଣ ସଂଖ୍ୟା"},
    "pa": {"name": "Punjabi", "script": "Gurmukhi", "sample": "ਇਨਵੌਇਸ ਨੰਬਰ"},
    "sa": {"name": "Sanskrit", "script": "Devanagari", "sample": "प्रमाणपत्रम्"},
}

SUPPORTED_LANGS = set(SUPPORTED_LANGUAGES_INFO.keys())

def is_gpu_requested() -> bool:
    val = os.getenv("TETRAX_USE_GPU", "false").lower()
    return val in ("1", "true", "yes", "on")

@lru_cache(maxsize=16)
def get_engine(lang: str, use_gpu: Optional[bool] = None):
    """
    Initializes and caches the PaddleOCR engine for the specified language.
    Downloads models on first call for the specific script.
    """
    if lang not in SUPPORTED_LANGS:
        raise ValueError(
            f"Unsupported language code {lang!r}. Allowed: {sorted(SUPPORTED_LANGS)}"
        )
    if use_gpu is None:
        use_gpu = is_gpu_requested()

    try:
        from paddleocr import PaddleOCR
    except ImportError as e:
        logger.error("PaddleOCR is not installed: %s", e)
        raise RuntimeError(
            "PaddleOCR is not installed in the current environment. "
            "Please run 'pip install paddleocr paddlepaddle'."
        ) from e

    # Initialize PaddleOCR engine with angle classifier for rotated Indian documents
    try:
        # Standard PaddleOCR 2.x / 3.x API
        return PaddleOCR(
            use_angle_cls=True,
            lang=lang,
            use_gpu=use_gpu,
            show_log=False
        )
    except TypeError:
        # Newer v3.x PaddleX configuration fallback
        return PaddleOCR(
            lang=lang,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=True,
        )

def _parse_standard_ocr_result(result: Any) -> List[dict]:
    """
    Parses standard PaddleOCR return format:
    result = [ [ [box, (text, score)], ... ] ]
    """
    lines = []
    if not result:
        return lines

    pages = result if isinstance(result, list) else [result]
    for page in pages:
        if not page or not isinstance(page, list):
            continue
        for item in page:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            box, rec = item[0], item[1]
            if isinstance(rec, (list, tuple)) and len(rec) >= 2:
                text = str(rec[0]).strip()
                score = float(rec[1]) if rec[1] is not None else 0.0
            elif isinstance(rec, str):
                text = rec.strip()
                score = 1.0
            else:
                continue

            # Ensure box is python list of floats
            clean_box = []
            if isinstance(box, (list, tuple, np.ndarray)):
                for pt in box:
                    if isinstance(pt, (list, tuple, np.ndarray)):
                        clean_box.append([float(pt[0]), float(pt[1])])
            lines.append({
                "text": text,
                "confidence": round(score, 4),
                "bounding_box": clean_box if clean_box else None,
            })
    return lines

def _parse_predict_result(result: Any) -> List[dict]:
    """
    Parses PaddleX / PaddleOCR 3.x .predict() output structure.
    """
    payload = getattr(result, "json", None)
    if callable(payload):
        payload = payload()
    if payload is None and isinstance(result, dict):
        payload = result
    if isinstance(payload, str):
        payload = json.loads(payload)
    if isinstance(payload, dict) and "res" in payload and isinstance(payload["res"], dict):
        payload = payload["res"]

    texts = payload.get("rec_texts", []) if isinstance(payload, dict) else []
    scores = payload.get("rec_scores", []) if isinstance(payload, dict) else []
    boxes = payload.get("rec_polys", payload.get("dt_polys", [])) if isinstance(payload, dict) else []

    lines = []
    for i, text in enumerate(texts or []):
        score = scores[i] if i < len(scores or []) else None
        box = boxes[i] if i < len(boxes or []) else None
        clean_box = []
        if isinstance(box, (list, tuple, np.ndarray)):
            for pt in box:
                if isinstance(pt, (list, tuple, np.ndarray)):
                    clean_box.append([float(pt[0]), float(pt[1])])
        lines.append({
            "text": str(text).strip(),
            "confidence": round(float(score), 4) if score is not None else None,
            "bounding_box": clean_box if clean_box else None,
        })
    return lines

def extract_image(
    image_input: Union[str, np.ndarray],
    lang: str = "en",
    apply_preprocessing: bool = True
) -> dict:
    """
    Extracts text from an image using PaddleOCR Option A.
    Accepts image file path or numpy array (BGR).
    """
    preprocessing_meta = {}
    if apply_preprocessing:
        processed_img, preprocessing_meta = preprocess_image(image_input)
    else:
        if isinstance(image_input, str):
            import cv2
            processed_img = cv2.imread(image_input)
            if processed_img is None:
                raise ValueError(f"Could not read image: {image_input}")
        else:
            processed_img = image_input

    engine = get_engine(lang)

    lines: List[dict] = []
    # 1. Try standard ocr(img, cls=True)
    if hasattr(engine, "ocr"):
        try:
            raw_result = engine.ocr(processed_img, cls=True)
            lines = _parse_standard_ocr_result(raw_result)
        except Exception as e:
            logger.warning("ocr() invocation encountered error, trying predict(): %s", e)

    # 2. Try predict(input=...) if lines empty or ocr() failed
    if not lines and hasattr(engine, "predict"):
        try:
            results = engine.predict(input=processed_img)
            for res in results or []:
                lines.extend(_parse_predict_result(res))
        except Exception as e:
            logger.error("predict() invocation failed: %s", e)

    full_text = "\n".join(item["text"] for item in lines)

    return {
        "text": full_text,
        "lines": lines,
        "line_count": len(lines),
        "language": lang,
        "preprocessing_applied": apply_preprocessing,
        "preprocessing_meta": preprocessing_meta,
    }
