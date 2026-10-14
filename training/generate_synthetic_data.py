"""
Synthetic Document Line Generator for English and Indian Business Documents.
Generates realistic line crops with varied fonts, background textures, and noise
for training PaddleOCR PP-OCR recognition models.
"""
import argparse
import csv
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "datasets" / "synthetic"

# Vocabulary & templates for bilingual Indian documents
VENDORS = [
    ("TATA ENTERPRISES PVT LTD", "टाटा एंटरप्राइजेज प्रा. लि."),
    ("RELIANCE RETAIL LIMITED", "रिलायंस रिटेल लिमिटेड"),
    ("BHARTI AIRTEL SERVICES", "भारती एयरटेल सर्विसेज"),
    ("INFOSYS BPO LIMITED", "इन्फोसिस बीपीओ लिमिटेड"),
    ("HINDUSTAN UNILEVER LTD", "हिंदुस्तान यूनिलीवर लिमिटेड"),
    ("LARSEN & TOUBRO INFOTECH", "लार्सन एंड टुब्रो इन्फोटेक"),
    ("CHENNAI TEXTILES TRADERS", "சென்னை டெக்ஸ்டைல்ஸ்"),
    ("HYDERABAD PHARMA SUPPLIERS", "హైదరాబాద్ ఫార్మా సప్లయర్స్"),
]

DOCUMENT_HEADERS = [
    ("TAX INVOICE", "कर बीजक"),
    ("BILL OF SUPPLY", "आपूर्ति बिल"),
    ("CASH MEMO", "कैश मेमो"),
    ("DELIVERY CHALLAN", "वितरण चालान"),
    ("PURCHASE ORDER", "क्रय आदेश"),
    ("PAYMENT RECEIPT", "भुगतान रसीद"),
    ("INVOICE SUMMARY", "बीजक सारांश"),
]

INDIAN_NAMES = [
    ("Rajesh Kumar Sharma", "राजेश कुमार शर्मा"),
    ("Priya Chandrasekhar", "प्रिया चंद्रशेखर"),
    ("Amitabh Mukhopadhyay", "अमिताभ मुखोपाध्याय"),
    ("Sunita Ramesh Patel", "सुनीता रमेश पटेल"),
    ("Karthikeyan Natarajan", "கார்த்திகேயன் நடராஜன்"),
    ("Srinivasa Rao Anumolu", "శ్రీనివాస రావు"),
    ("Vikram Singh Chauhan", "विक्रम सिंह चौहान"),
    ("Ananya Roy Chowdhury", "অনন্যা রায় চৌধুরী"),
]

INDIAN_CITIES_STATES = [
    ("Mumbai, Maharashtra - 400001", "मुंबई, महाराष्ट्र - ४००००१"),
    ("New Delhi, Delhi - 110001", "नई दिल्ली, दिल्ली - ११०००१"),
    ("Bengaluru, Karnataka - 560001", "बेंगलुरु, कर्नाटक - ५६०००१"),
    ("Chennai, Tamil Nadu - 600001", "சென்னை, தமிழ்நாடு - 600001"),
    ("Hyderabad, Telangana - 500001", "హైదరాబాద్, తెలంగాణ - 500001"),
    ("Kolkata, West Bengal - 700001", "কলকাতা, পশ্চিমবঙ্গ - 700001"),
    ("Ahmedabad, Gujarat - 380001", "અમદાવાદ, ગુજરાત - 380001"),
    ("Pune, Maharashtra - 411001", "पुणे, महाराष्ट्र - ४११००१"),
]

def generate_random_gstin():
    state_code = f"{random.randint(1, 37):02d}"
    pan = "".join(random.choices("ABCDEFGHJKLMNPQRSTUVWXYZ", k=5)) + f"{random.randint(1000, 9999)}" + random.choice("ABCDEFGHJKLMNPQRSTUVWXYZ")
    return f"{state_code}{pan}1Z{random.choice('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')}"

def generate_random_amount():
    amt = random.uniform(150.0, 95000.0)
    return f"{amt:,.2f}"

def generate_line_text() -> str:
    category = random.choice(["header", "vendor", "customer", "gstin", "amount", "address", "invoice_no"])
    
    if category == "header":
        pair = random.choice(DOCUMENT_HEADERS)
        return random.choice([pair[0], pair[1], f"{pair[0]} / {pair[1]}"])
    elif category == "vendor":
        pair = random.choice(VENDORS)
        return random.choice([pair[0], pair[1]])
    elif category == "customer":
        pair = random.choice(INDIAN_NAMES)
        prefix_en = random.choice(["Customer Name: ", "Billed To: ", "M/s. "])
        prefix_hi = random.choice(["ग्राहक: ", "क्रेता: "])
        if random.random() < 0.6:
            return f"{prefix_en}{pair[0]}"
        return f"{prefix_hi}{pair[1]}"
    elif category == "gstin":
        gst = generate_random_gstin()
        prefix = random.choice(["GSTIN: ", "GST No: ", "जीएसटी संख्या: "])
        return f"{prefix}{gst}"
    elif category == "amount":
        amt = generate_random_amount()
        term = random.choice([
            f"Total Amount: Rs. {amt}",
            f"Net Payable: INR {amt}",
            f"Subtotal: {amt}",
            f"कुल देय राशि: रु. {amt}",
            f"CGST @ 9%: Rs. {generate_random_amount()}",
            f"SGST @ 9%: Rs. {generate_random_amount()}",
        ])
        return term
    elif category == "address":
        pair = random.choice(INDIAN_CITIES_STATES)
        return random.choice([pair[0], pair[1]])
    else:
        inv_no = f"INV-{random.randint(2024, 2026)}-{random.randint(10000, 99999)}"
        prefix = random.choice(["Invoice No: ", "Bill No: ", "बीजक सं: "])
        return f"{prefix}{inv_no}"

def get_font(size=26):
    font_candidates = [
        "C:\\Windows\\Fonts\\Nirmala.ttf",
        "C:\\Windows\\Fonts\\mangal.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for c in font_candidates:
        if Path(c).exists():
            try:
                return ImageFont.truetype(c, size)
            except Exception:
                pass
    return ImageFont.load_default()

def render_synthetic_image(text: str, out_path: Path, font):
    dummy = Image.new("RGB", (10, 10))
    draw_d = ImageDraw.Draw(dummy)
    bbox = draw_d.textbbox((0, 0), text, font=font)
    tw = max(bbox[2] - bbox[0], 100)
    th = max(bbox[3] - bbox[1], 24)

    pad_x = random.randint(16, 28)
    pad_y = random.randint(8, 16)
    w = tw + pad_x * 2
    h = th + pad_y * 2

    # Background color with slight paper tint variation
    bg_r = random.randint(245, 255)
    bg_g = random.randint(245, 255)
    bg_b = random.randint(240, 252)
    img = Image.new("RGB", (w, h), color=(bg_r, bg_g, bg_b))
    draw = ImageDraw.Draw(img)

    # Text color (black / dark gray / navy ink)
    ink_color = random.choice([(15, 20, 25), (30, 35, 45), (10, 15, 60), (25, 25, 25)])
    draw.text((pad_x - bbox[0], pad_y - bbox[1]), text, fill=ink_color, font=font)

    # Optional mild gaussian blur to simulate optical scan
    if random.random() < 0.25:
        img = img.filter(ImageFilter.GaussianBlur(radius=0.4))

    img.save(out_path)

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic English and Indian document text lines.")
    parser.add_argument("--count", type=int, default=100, help="Number of synthetic samples to generate")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    font = get_font(size=26)
    csv_path = out_dir / "synthetic_annotations.csv"

    rows = []
    print(f"Generating {args.count} synthetic English & Indian document lines...")
    for i in range(1, args.count + 1):
        text = generate_line_text()
        fname = f"synth_{i:05d}.png"
        fpath = images_dir / fname
        render_synthetic_image(text, fpath, font)
        rows.append({
            "image_path": f"datasets/synthetic/images/{fname}",
            "text": text,
        })

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "text"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Successfully generated {args.count} synthetic images in {images_dir}")
    print(f"Saved annotations to {csv_path}")

if __name__ == "__main__":
    main()
