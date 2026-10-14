"""
Benchmark any OCR model on Sarvam AI's Indic OCR Bench.
Evaluates Character Error Rate (CER) and Word Error Rate (WER) per language.
"""
import argparse
import collections
import unicodedata
from pathlib import Path
from jiwer import cer, wer
from tqdm import tqdm

from app.ocr_engine import extract_image

def normalize(text: str) -> str:
    return unicodedata.normalize("NFC", (text or "").strip())

# Mapping of dataset language names to PaddleOCR language codes
LANG_CODE_MAP = {
    "hindi": "hi",
    "marathi": "mr",
    "tamil": "ta",
    "telugu": "te",
    "english": "en",
    "bengali": "bn",
    "gujarati": "gu",
    "kannada": "kn",
    "malayalam": "ml",
    "odia": "or",
    "punjabi": "pa",
    "urdu": "ur",
    "sanskrit": "sa",
}

def main():
    parser = argparse.ArgumentParser(description="Benchmark OCR on sarvamai/indic-ocr-bench.")
    parser.add_argument("--split", default="small_representative", choices=["small_representative", "test"])
    parser.add_argument("--languages", default="hindi,english", help="Comma-separated languages to evaluate")
    parser.add_argument("--max-samples", type=int, default=50, help="Number of samples to evaluate")
    args = parser.parse_args()

    from datasets import load_dataset
    print(f"Loading '{args.split}' from 'sarvamai/indic-ocr-bench'...")
    ds = load_dataset("sarvamai/indic-ocr-bench", split=args.split)

    target_langs = None
    if args.languages.strip().lower() != "all":
        target_langs = {l.strip().lower() for l in args.languages.split(",")}
    results_by_lang = collections.defaultdict(lambda: {"refs": [], "preds": []})

    count = 0
    for item in tqdm(ds, desc="Evaluating"):
        lang = str(item.get("language", "")).lower()
        if target_langs is not None and lang not in target_langs:
            continue

        ref = normalize(str(item.get("gt", "")))
        if not ref:
            continue

        pil_img = item["image"]
        paddle_lang = LANG_CODE_MAP.get(lang, "en")

        try:
            import numpy as np
            import cv2
            img_bgr = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
            res = extract_image(img_bgr, lang=paddle_lang, apply_preprocessing=True)
            pred = normalize(res["text"].replace("\n", " "))
        except Exception as e:
            pred = ""

        results_by_lang[lang]["refs"].append(ref)
        results_by_lang[lang]["preds"].append(pred)

        count += 1
        if args.max_samples and count >= args.max_samples:
            break

    print("\n" + "=" * 65)
    print("    SARVAM INDIC OCR BENCHMARK RESULTS")
    print("=" * 65)
    for lang, data in results_by_lang.items():
        refs, preds = data["refs"], data["preds"]
        if not refs:
            continue
        c_val = cer(refs, preds)
        w_val = wer(refs, preds)
        exact = sum(1 for r, p in zip(refs, preds) if r == p)
        acc = (exact / len(refs)) * 100.0
        print(f"Language: {lang.upper():12s} | Samples: {len(refs):3d} | Exact: {acc:5.1f}% | CER: {c_val*100:5.2f}% | WER: {w_val*100:5.2f}%")
    print("=" * 65)

if __name__ == "__main__":
    main()
