import argparse
import csv
import unicodedata
from jiwer import cer, wer

def normalize(text: str) -> str:
    if not text:
        return ""
    # NFC normalization for consistent Indian script representation
    return unicodedata.normalize("NFC", text.strip())

def evaluate_predictions(predictions_csv_path: str):
    refs, preds = [], []
    exact_matches = 0

    with open(predictions_csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ref = normalize(row.get("reference", ""))
            pred = normalize(row.get("prediction", ""))
            refs.append(ref)
            preds.append(pred)
            if ref == pred:
                exact_matches += 1

    total = len(refs)
    if total == 0:
        raise SystemExit("No evaluation samples found in file.")

    computed_cer = cer(refs, preds)
    computed_wer = wer(refs, preds)
    acc = (exact_matches / total) * 100.0

    print("=" * 45)
    print("      TETRAX OCR EVALUATION REPORT")
    print("=" * 45)
    print(f"Total Test Samples:   {total}")
    print(f"Exact Match Accuracy: {acc:.2f}% ({exact_matches}/{total})")
    print(f"Character Error Rate: {computed_cer:.4f} ({computed_cer * 100:.2f}%)")
    print(f"Word Error Rate:      {computed_wer:.4f} ({computed_wer * 100:.2f}%)")
    print("=" * 45)

    return {
        "samples": total,
        "exact_match_accuracy": acc,
        "cer": computed_cer,
        "wer": computed_wer,
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate OCR predictions using CER, WER, and Exact Match.")
    parser.add_argument("--predictions", required=True, help="CSV containing 'reference' and 'prediction' columns")
    args = parser.parse_args()
    evaluate_predictions(args.predictions)

if __name__ == "__main__":
    main()
