"""
Dataset download and acquisition tool for Tetrax OCR.
Focuses on English and Indian Language document datasets:
- AI4Bharat IndicOCR (Devanagari, Tamil, Telugu, Bengali, Gujarati, etc.)
- SROIE (English Receipts and Invoices)
- Hugging Face Document Understanding benchmarks
"""
import argparse
import os
import sys
from pathlib import Path

DATASETS_DIR = Path(__file__).resolve().parent

DATASET_REPOSITORIES = {
    "indic_ocr": {
        "title": "AI4Bharat IndicOCR",
        "languages": ["Hindi", "Marathi", "Tamil", "Telugu", "Bengali", "Gujarati", "Kannada", "Malayalam", "Odia", "Punjabi", "Urdu"],
        "description": "State-of-the-art benchmark dataset with millions of annotated text lines across printed Indian books, magazines, and government records.",
        "url": "https://indicnlp.ai4bharat.org/indic-ocr/",
        "hf_id": "ai4bharat/IndicOCR",
    },
    "sroie": {
        "title": "SROIE (Scanned Receipts OCR and Information Extraction)",
        "languages": ["English"],
        "description": "Standard benchmark of 1,000 scanned store receipts and invoices with word-level bounding boxes and text transcriptions.",
        "hf_id": "darentang/sroie",
        "url": "https://rrc.cvc.uab.es/?ch=13",
    },
    "funsd": {
        "title": "FUNSD (Form Understanding in Noisy Scanned Documents)",
        "languages": ["English"],
        "description": "199 scanned noisy forms, ideal for verifying OCR robustness on complex printed forms.",
        "hf_id": "nielsr/funsd",
        "url": "https://guillaumejaume.github.io/FUNSD/",
    },
    "iiit_ilst": {
        "title": "IIIT-ILST Indian Script Dataset (IIIT Hyderabad)",
        "languages": ["Hindi", "Telugu", "Tamil", "Bangla"],
        "description": "High-quality cropped word and line images for Indian printed documents across multiple fonts and sizes.",
        "url": "http://cvit.iiit.ac.in/research/projects/cvit-projects/indian-language-script-technology",
    },
    "midas_id": {
        "title": "MIDAS Indian ID & Document Benchmark",
        "languages": ["English", "Hindi"],
        "description": "PAN card, Aadhaar, and Driving License text fields and OCR challenge dataset.",
        "url": "https://github.com/midas-research",
    }
}

def list_datasets():
    print("\n" + "=" * 70)
    print("  CURATED DATASETS FOR ENGLISH & INDIAN DOCUMENT OCR")
    print("=" * 70)
    for key, info in DATASET_REPOSITORIES.items():
        print(f"\n[{key.upper()}] - {info['title']}")
        print(f"  Languages:   {', '.join(info['languages'])}")
        print(f"  Description: {info['description']}")
        print(f"  Source URL:  {info['url']}")
        if "hf_id" in info:
            print(f"  HuggingFace: {info['hf_id']}")
    print("=" * 70)

def download_hf_dataset(dataset_key: str, output_dir: Path):
    if dataset_key not in DATASET_REPOSITORIES:
        print(f"Error: Unknown dataset '{dataset_key}'. Use --list to view options.")
        return

    info = DATASET_REPOSITORIES[dataset_key]
    hf_id = info.get("hf_id")
    if not hf_id:
        print(f"Manual download required for {info['title']}. Visit: {info['url']}")
        return

    print(f"\nAttempting to download {info['title']} ({hf_id}) via Hugging Face `datasets`...")
    try:
        from datasets import load_dataset
        ds = load_dataset(hf_id)
        print(f"Successfully loaded dataset structure for {hf_id}:")
        print(ds)
        dest = output_dir / dataset_key
        dest.mkdir(parents=True, exist_ok=True)
        print(f"Target directory ready at {dest}")
        print("Tip: See datasets/DATASETS_GUIDE.md for conversion to PaddleOCR format.")
    except Exception as e:
        print(f"Could not automatically download {hf_id}: {e}")
        print(f"You can also download directly from: {info['url']}")

def main():
    parser = argparse.ArgumentParser(description="Explore and download English and Indian OCR datasets.")
    parser.add_argument("--list", action="store_true", help="List all curated English and Indian OCR datasets")
    parser.add_argument("--download", choices=list(DATASET_REPOSITORIES.keys()), help="Dataset key to download")
    parser.add_argument("--dest", default=str(DATASETS_DIR / "external"), help="Destination directory for downloads")
    args = parser.parse_args()

    if args.list or not args.download:
        list_datasets()
        if not args.download:
            print("\nTo download an accessible dataset, run:")
            print("  python -m datasets.download_datasets --download sroie")
            print("  python -m datasets.download_datasets --download funsd")
            print("\nFor detailed Indian datasets setup (AI4Bharat / IndicOCR), see datasets/DATASETS_GUIDE.md")
            return

    download_hf_dataset(args.download, Path(args.dest))

if __name__ == "__main__":
    main()
