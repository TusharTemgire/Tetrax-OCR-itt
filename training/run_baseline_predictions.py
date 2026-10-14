"""
Runs PaddleOCR baseline inference over a test dataset and outputs datasets/predictions.csv
for automatic CER/WER evaluation.
"""
import argparse
import csv
from pathlib import Path
from tqdm import tqdm

from app.ocr_engine import extract_image

def main():
    parser = argparse.ArgumentParser(description="Generate OCR predictions for a test dataset.")
    parser.add_argument("--test-csv", default="datasets/annotations/test.csv", help="CSV with image_path,text columns")
    parser.add_argument("--output", default="datasets/predictions.csv", help="Destination predictions CSV")
    parser.add_argument("--lang", default="en", help="Language code to evaluate (e.g. 'en', 'hi', 'ta')")
    parser.add_argument("--no-preprocess", action="store_true", help="Disable preprocessing")
    args = parser.parse_args()

    test_csv_path = Path(args.test_csv)
    if not test_csv_path.exists():
        raise SystemExit(f"Test CSV not found: {test_csv_path}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(test_csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    print(f"Running baseline PaddleOCR inference on {len(rows)} samples (lang='{args.lang}')...")

    results = []
    for r in tqdm(rows):
        img_path = r["image_path"]
        ref = r.get("text", "")
        # Resolve path
        p = Path(img_path)
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()

        if not p.exists():
            print(f"Warning: image {p} not found, skipping.")
            continue

        try:
            res = extract_image(
                str(p),
                lang=args.lang,
                apply_preprocessing=not args.no_preprocess
            )
            pred_text = res["text"].replace("\n", " ").strip()
        except Exception as e:
            print(f"Inference error on {p}: {e}")
            pred_text = ""

        results.append({
            "image_path": str(p),
            "reference": ref,
            "prediction": pred_text,
        })

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "reference", "prediction"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved {len(results)} predictions to {out_path}")
    print("Now run evaluation with:")
    print(f"python -m training.evaluate --predictions {out_path}")

if __name__ == "__main__":
    main()
