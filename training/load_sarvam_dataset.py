"""
Dataset loader and preprocessor for Sarvam AI's Indic OCR Benchmark
Hugging Face: sarvamai/indic-ocr-bench

Exports images, builds train/val/test splits, generates PaddleOCR-formatted
label files (train.txt, val.txt, test.txt), and creates the character dictionary
for training your custom OCR model.
"""
import argparse
import csv
import os
import random
import unicodedata
from pathlib import Path
from typing import List, Dict, Set
from tqdm import tqdm

def normalize(text: str) -> str:
    """NFC Unicode normalization is essential for Indian scripts (vowels, matras, conjuncts)."""
    return unicodedata.normalize("NFC", text.strip())

def main():
    parser = argparse.ArgumentParser(description="Download and prepare sarvamai/indic-ocr-bench for OCR training.")
    parser.add_argument("--split", default="small_representative", choices=["small_representative", "test"],
                        help="'small_representative' (1,173 images, ~95MB) or 'test' (6,909 images, ~638MB)")
    parser.add_argument("--languages", default="all",
                        help="Comma-separated languages (e.g. 'hindi,english,tamil,telugu,marathi') or 'all'")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum total samples to extract (useful for fast pipeline testing)")
    parser.add_argument("--output-dir", default="datasets/sarvam_indic",
                        help="Directory to store extracted images, CSVs, and PaddleOCR labels")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.output_dir).resolve()
    images_dir = out_dir / "images"
    ann_dir = out_dir / "annotations"
    paddle_dir = out_dir / "paddle_rec"
    dict_dir = out_dir / "dicts"

    for d in [images_dir, ann_dir, paddle_dir, dict_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"Loading '{args.split}' split from 'sarvamai/indic-ocr-bench'...")
    try:
        import io
        from PIL import Image as PILImage
        from datasets import load_dataset, Image as HFImage
        ds = load_dataset("sarvamai/indic-ocr-bench", split=args.split)
        # Disable eager decoding to prevent crashes on broken dataset rows (e.g. indic_ocr_bench_test_eng_5.png)
        ds = ds.cast_column("image", HFImage(decode=False))
    except Exception as e:
        raise SystemExit(f"Failed to load dataset from Hugging Face: {e}")

    print(f"Total rows in dataset: {len(ds)}")

    target_langs = None
    if args.languages.lower() != "all":
        target_langs = {l.strip().lower() for l in args.languages.split(",")}
        print(f"Filtering for languages: {target_langs}")

    extracted_records: List[Dict[str, str]] = []
    unique_chars: Set[str] = set()

    for idx, item in enumerate(tqdm(ds, desc="Processing images")):
        lang = str(item.get("language", "")).lower()
        if target_langs and lang not in target_langs:
            continue

        gt_text = normalize(str(item.get("gt", "")))
        if not gt_text:
            continue

        img_name = item.get("image_name") or f"indic_{idx:05d}.png"
        if not Path(img_name).suffix:
            img_name += ".png"

        image_path = images_dir / img_name
        
        # Save image to disk if not already present
        if not image_path.exists():
            img_entry = item.get("image")
            pil_img = None
            try:
                if isinstance(img_entry, dict):
                    raw_bytes = img_entry.get("bytes")
                    raw_path = img_entry.get("path")
                    if raw_bytes:
                        pil_img = PILImage.open(io.BytesIO(raw_bytes))
                    elif raw_path and os.path.exists(raw_path):
                        pil_img = PILImage.open(raw_path)
                elif hasattr(img_entry, "convert"):
                    pil_img = img_entry
                
                if pil_img is None:
                    continue
                if pil_img.mode != "RGB":
                    pil_img = pil_img.convert("RGB")
                pil_img.save(image_path)
            except Exception:
                continue

        for c in gt_text:
            if c not in ("\n", "\r", "\t"):
                unique_chars.add(c)

        # Save relative path from project root for portability
        try:
            portable_path = str(image_path.relative_to(Path.cwd())).replace("\\", "/")
        except ValueError:
            portable_path = str(image_path).replace("\\", "/")

        extracted_records.append({
            "image_path": portable_path,
            "text": gt_text,
            "language": lang,
        })

        if args.max_samples and len(extracted_records) >= args.max_samples:
            print(f"Reached requested limit of {args.max_samples} samples.")
            break

    total = len(extracted_records)
    print(f"\nExtracted {total} samples across targeted languages.")
    if total == 0:
        raise SystemExit("No samples matched the filter criteria.")

    # Shuffle and split
    random.Random(args.seed).shuffle(extracted_records)
    train_end = int(total * args.train_ratio)
    val_end = train_end + int(total * args.val_ratio)

    splits = {
        "train": extracted_records[:train_end],
        "validation": extracted_records[train_end:val_end],
        "test": extracted_records[val_end:],
    }

    # 1. Write standard CSV annotations
    for split_name, rows in splits.items():
        csv_file = ann_dir / f"{split_name}.csv"
        with open(csv_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["image_path", "text", "language"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"Created {csv_file.name}: {len(rows)} samples")

    # 2. Write PaddleOCR label format (image_path\ttext)
    paddle_files = {
        "train.txt": splits["train"],
        "val.txt": splits["validation"],
        "test.txt": splits["test"],
    }
    for filename, rows in paddle_files.items():
        txt_path = paddle_dir / filename
        with open(txt_path, "w", encoding="utf-8") as f:
            for r in rows:
                clean_text = r["text"].replace("\n", " ").replace("\r", " ")
                f.write(f"{r['image_path']}\t{clean_text}\n")
        print(f"Created PaddleOCR label: {txt_path.name}: {len(rows)} samples")

    # 3. Write character dictionary (required for Indian scripts)
    dict_file = dict_dir / "sarvam_dict.txt"
    sorted_chars = sorted(unique_chars)
    with open(dict_file, "w", encoding="utf-8") as f:
        for ch in sorted_chars:
            f.write(f"{ch}\n")
    print(f"Created character dictionary: {dict_file} ({len(sorted_chars)} unique characters)")

    print("\n" + "=" * 60)
    print("Sarvam Indic OCR Dataset preparation completed successfully!")
    print(f"Images location:     {images_dir}")
    print(f"PaddleOCR labels:    {paddle_dir}")
    print(f"Character dict:      {dict_file}")
    print("=" * 60)

if __name__ == "__main__":
    main()
