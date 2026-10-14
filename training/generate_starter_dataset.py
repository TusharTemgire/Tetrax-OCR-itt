"""
Generates starter dataset images and annotations for English and Indian documents.
Produces clean text-line images with exact ground truth transcriptions.
"""
import csv
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

DATASETS_DIR = Path(__file__).resolve().parent.parent / "datasets"
IMAGES_DIR = DATASETS_DIR / "images"
ANNOTATIONS_DIR = DATASETS_DIR / "annotations"

IMAGES_DIR.mkdir(parents=True, exist_ok=True)
ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

# Dataset samples covering English, Hindi, Marathi, Tamil, Telugu business documents & IDs
SAMPLES = [
    # English Invoice & Business Fields
    {"id": "en_001", "text": "INVOICE NUMBER: INV-2026-8941", "lang": "en", "split": "train"},
    {"id": "en_002", "text": "TAX INVOICE / BILL OF SUPPLY", "lang": "en", "split": "train"},
    {"id": "en_003", "text": "Customer Name: Rajesh Verma", "lang": "en", "split": "train"},
    {"id": "en_004", "text": "GSTIN: 27AABCT3518Q1ZY", "lang": "en", "split": "train"},
    {"id": "en_005", "text": "Subtotal: Rs. 14,500.00", "lang": "en", "split": "train"},
    {"id": "en_006", "text": "CGST @ 9%: Rs. 1,305.00", "lang": "en", "split": "train"},
    {"id": "en_007", "text": "SGST @ 9%: Rs. 1,305.00", "lang": "en", "split": "train"},
    {"id": "en_008", "text": "Total Payable: Rs. 17,110.00", "lang": "en", "split": "val"},
    {"id": "en_009", "text": "Due Date: 28th October 2026", "lang": "en", "split": "val"},
    {"id": "en_010", "text": "Payment Mode: NEFT / RTGS Transfer", "lang": "en", "split": "test"},
    {"id": "en_011", "text": "PAN: ABCDE1234F", "lang": "en", "split": "test"},
    {"id": "en_012", "text": "Place of Supply: Maharashtra (27)", "lang": "en", "split": "test"},

    # Hindi (Devanagari) Documents & Receipts
    {"id": "hi_001", "text": "कर बीजक (Tax Invoice)", "lang": "hi", "split": "train"},
    {"id": "hi_002", "text": "ग्राहक का नाम: अमित कुमार शर्मा", "lang": "hi", "split": "train"},
    {"id": "hi_003", "text": "बीजक संख्या: बीजे-४५९०२", "lang": "hi", "split": "train"},
    {"id": "hi_004", "text": "कुल देय राशि: रु. २५,०००.००", "lang": "hi", "split": "train"},
    {"id": "hi_005", "text": "भारत सरकार - आयकर विभाग", "lang": "hi", "split": "train"},
    {"id": "hi_006", "text": "दिनांक: १५/१०/२०२६", "lang": "hi", "split": "val"},
    {"id": "hi_007", "text": "नकद भुगतान रसीद", "lang": "hi", "split": "val"},
    {"id": "hi_008", "text": "माल एवं सेवा कर पहचान संख्या", "lang": "hi", "split": "test"},
    {"id": "hi_009", "text": "प्राधिकृत हस्ताक्षरकर्ता", "lang": "hi", "split": "test"},

    # Marathi (Devanagari) Documents
    {"id": "mr_001", "text": "पावती क्रमांक: एमएच-७८९", "lang": "mr", "split": "train"},
    {"id": "mr_002", "text": "ग्राहकाचे नाव: सचिन तानाजी कदम", "lang": "mr", "split": "train"},
    {"id": "mr_003", "text": "एकूण रक्कम: रु. १२,४५०.००", "lang": "mr", "split": "test"},

    # Tamil Documents
    {"id": "ta_001", "text": "விலைப்பட்டியல் எண்: டிஎன்-1082", "lang": "ta", "split": "train"},
    {"id": "ta_002", "text": "மொத்த தொகை: ரூ. 18,750.00", "lang": "ta", "split": "train"},
    {"id": "ta_003", "text": "வாடிக்கையாளர் பெயர்: கார்த்திக்", "lang": "ta", "split": "val"},
    {"id": "ta_004", "text": "தேதி: 14 அக்டோபர் 2026", "lang": "ta", "split": "test"},

    # Telugu Documents
    {"id": "te_001", "text": "ఇన్వాయిస్ సంఖ్య: ఏపీ-3041", "lang": "te", "split": "train"},
    {"id": "te_002", "text": "మొత్తం చెల్లింపు: రూ. 9,800.00", "lang": "te", "split": "train"},
    {"id": "te_003", "text": "తేదీ: 20 అక్టోబర్ 2026", "lang": "te", "split": "test"},
]

def find_font():
    """Finds a compatible Unicode font supporting Latin and Devanagari/Indic scripts on Windows/Linux."""
    candidates = [
        "C:\\Windows\\Fonts\\mangal.ttf",      # Windows Devanagari font
        "C:\\Windows\\Fonts\\aparaj.ttf",     # Windows Devanagari font
        "C:\\Windows\\Fonts\\arial.ttf",      # Windows Arial
        "C:\\Windows\\Fonts\\segoeui.ttf",    # Windows Segoe UI
        "C:\\Windows\\Fonts\\Nirmala.ttf",    # Windows Nirmala UI (All Indic scripts: Hindi, Tamil, Telugu, etc.)
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    return None

def create_text_image(text: str, filename: str, font_path: str = None) -> Path:
    # Use Nirmala UI if available on Windows as it has universal Indic script coverage
    nirmala_path = "C:\\Windows\\Fonts\\Nirmala.ttf"
    chosen_font_path = nirmala_path if Path(nirmala_path).exists() else font_path

    font_size = 28
    font = None
    if chosen_font_path and Path(chosen_font_path).exists():
        try:
            font = ImageFont.truetype(chosen_font_path, font_size)
        except Exception:
            font = ImageFont.load_default()
    else:
        font = ImageFont.load_default()

    # Create dummy image to calculate bounding box
    dummy = Image.new("RGB", (10, 10), color="white")
    d_draw = ImageDraw.Draw(dummy)
    bbox = d_draw.textbbox((0, 0), text, font=font)
    text_w = max(bbox[2] - bbox[0], 120)
    text_h = max(bbox[3] - bbox[1], 30)

    # Padding around text line
    padding_x = 24
    padding_y = 14
    img_w = text_w + (padding_x * 2)
    img_h = text_h + (padding_y * 2)

    # Draw crisp document text line (off-white invoice paper background + dark text)
    img = Image.new("RGB", (img_w, img_h), color=(250, 250, 252))
    draw = ImageDraw.Draw(img)

    # Draw faint border or ruled line (typical in Indian invoices/forms)
    draw.rectangle([0, 0, img_w - 1, img_h - 1], outline=(225, 228, 232), width=1)
    
    # Draw text
    draw.text((padding_x - bbox[0], padding_y - bbox[1]), text, fill=(20, 24, 33), font=font)

    out_file = IMAGES_DIR / filename
    img.save(out_file)
    return out_file

def main():
    font_path = find_font()
    print(f"Using font for rendering starter samples: {font_path}")

    splits_data = {"train": [], "val": [], "test": []}

    for s in SAMPLES:
        fname = f"{s['id']}.png"
        img_path = create_text_image(s["text"], fname, font_path)
        row = {
            "image_path": f"datasets/images/{fname}",
            "text": s["text"],
            "language": s["lang"]
        }
        splits_data[s["split"]].append(row)

    # Write split CSV files
    for split_name, rows in splits_data.items():
        csv_filename = "validation.csv" if split_name == "val" else f"{split_name}.csv"
        csv_path = ANNOTATIONS_DIR / csv_filename
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["image_path", "text", "language"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"Generated {len(rows)} samples in {csv_path}")

    # Also generate a consolidated annotations.csv
    all_rows = splits_data["train"] + splits_data["val"] + splits_data["test"]
    consolidated_csv = DATASETS_DIR / "annotations.csv"
    with open(consolidated_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "text", "language"])
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Generated consolidated annotations at {consolidated_csv}")

if __name__ == "__main__":
    main()
