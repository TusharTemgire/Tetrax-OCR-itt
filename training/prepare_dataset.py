import argparse
import csv
import os
import random
import unicodedata
from pathlib import Path
from typing import List, Tuple, Set

def normalize_text(text: str) -> str:
    """NFC normalization is essential for Indian scripts (vowel signs / matras / conjuncts)."""
    return unicodedata.normalize("NFC", text.strip())

def extract_rows(csv_path: Path, root_dir: Path) -> List[Tuple[str, str]]:
    rows = []
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_rel = row["image_path"].strip()
            text = normalize_text(row.get("text", ""))
            if not text:
                continue

            img_p = Path(img_rel)
            if not img_p.is_absolute():
                img_p = (root_dir / img_p).resolve()

            # Record relative path to root_dir if possible for portable datasets
            try:
                portable_path = str(img_p.relative_to(root_dir)).replace("\\", "/")
            except ValueError:
                portable_path = str(img_p).replace("\\", "/")

            rows.append((portable_path, text))
    return rows

def generate_char_dictionary(texts: List[str], dict_out_path: Path):
    """
    Extracts all unique characters across the corpus and generates
    the character dictionary needed by PaddleOCR.
    """
    chars: Set[str] = set()
    for t in texts:
        for c in t:
            if c not in ("\n", "\r", "\t"):
                chars.add(c)

    sorted_chars = sorted(chars)
    dict_out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dict_out_path, "w", encoding="utf-8") as f:
        for ch in sorted_chars:
            f.write(f"{ch}\n")
    print(f"Generated character dictionary with {len(sorted_chars)} characters at: {dict_out_path}")

def main():
    parser = argparse.ArgumentParser(description="Prepare dataset labels and character dictionary for PaddleOCR.")
    parser.add_argument("--csv", help="Single CSV containing image_path,text columns")
    parser.add_argument("--annotations-dir", default="datasets/annotations", help="Directory with train.csv, validation.csv, test.csv")
    parser.add_argument("--output-dir", default="datasets/paddle_rec", help="Output directory for PaddleOCR label files")
    parser.add_argument("--dict-output", default="datasets/dicts/custom_dict.txt", help="Output path for character dictionary")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path.cwd()
    out = (root / args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    all_texts = []

    # Case 1: Pre-existing train/val/test CSV files
    ann_dir = (root / args.annotations_dir).resolve()
    train_csv = ann_dir / "train.csv"
    val_csv = ann_dir / "validation.csv"
    test_csv = ann_dir / "test.csv"

    if train_csv.exists() and not args.csv:
        print(f"Reading pre-split annotations from {ann_dir}...")
        splits = {
            "train.txt": extract_rows(train_csv, root),
            "val.txt": extract_rows(val_csv, root) if val_csv.exists() else [],
            "test.txt": extract_rows(test_csv, root) if test_csv.exists() else [],
        }
        for _, rows in splits.items():
            all_texts.extend(text for _, text in rows)

    # Case 2: Single CSV to split
    elif args.csv:
        csv_file = (root / args.csv).resolve()
        if not csv_file.exists():
            raise SystemExit(f"CSV file not found: {csv_file}")
        rows = extract_rows(csv_file, root)
        if len(rows) < 5:
            print("Warning: fewer than 5 examples provided.")
        random.Random(args.seed).shuffle(rows)
        n = len(rows)
        train_end = int(n * args.train_ratio)
        val_end = train_end + int(n * args.val_ratio)
        splits = {
            "train.txt": rows[:train_end],
            "val.txt": rows[train_end:val_end],
            "test.txt": rows[val_end:],
        }
        all_texts = [text for _, text in rows]
    else:
        raise SystemExit(f"No valid annotation CSVs found in {ann_dir} and no --csv argument specified.")

    for filename, items in splits.items():
        dest = out / filename
        with open(dest, "w", encoding="utf-8") as f:
            for portable_path, text in items:
                # PaddleOCR label format: <image_path>\t<text>
                f.write(f"{portable_path}\t{text}\n")
        print(f"Wrote {dest.name}: {len(items)} items")

    if all_texts and args.dict_output:
        dict_dest = (root / args.dict_output).resolve()
        generate_char_dictionary(all_texts, dict_dest)

    print("\nDataset preparation complete! Files ready for PaddleOCR training.")

if __name__ == "__main__":
    main()
