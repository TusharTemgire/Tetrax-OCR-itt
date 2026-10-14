# English and Indian Document Datasets for Tetrax OCR

This guide curates the primary public datasets and benchmarks for training and evaluating **PaddleOCR (Option A)** on **English and Indian documents** (invoices, receipts, forms, identity cards, and legal records).

---

## 1. Indian Language Document Datasets

Indian documents present unique OCR challenges: conjunct consonants, vowel modifiers (*matras*), multi-script headers, and bilingual layouts (English + Hindi/Tamil/Telugu/etc.).

### 1.1 AI4Bharat IndicOCR
- **Overview**: The largest, most authoritative OCR dataset for Indian languages, developed by IIT Madras and AI4Bharat.
- **Languages Supported**: 
  - **Devanagari**: Hindi (`hi`), Marathi (`mr`), Sanskrit (`sa`)
  - **Dravidian**: Tamil (`ta`), Telugu (`te`), Kannada (`kn`), Malayalam (`ml`)
  - **Eastern Indo-Aryan**: Bengali (`bn`), Odia (`or`)
  - **Western Indo-Aryan**: Gujarati (`gu`), Punjabi (`pa`)
  - **Perso-Arabic**: Urdu (`ur`)
- **Scale**: Over 10 million annotated printed text lines extracted from government gazettes, classical literature, legal journals, and public records.
- **Access / Website**: [https://indicnlp.ai4bharat.org/indic-ocr/](https://indicnlp.ai4bharat.org/indic-ocr/)
- **Hugging Face**: `ai4bharat/IndicOCR`
- **Recommended Use**: Pre-training and fine-tuning PaddleOCR recognition models for high accuracy on Indian scripts.

### 1.2 IIIT-ILST (Indian Language Script Technology, CVIT IIIT-H)
- **Overview**: Curated by the Center for Visual Information Technology (CVIT) at IIIT Hyderabad.
- **Languages Supported**: Hindi, Telugu, Tamil, Bengali, Malayalam.
- **Format**: Word-level and text-line level bounding boxes and Unicode ground truth.
- **Website**: [http://cvit.iiit.ac.in/research/projects/cvit-projects/indian-language-script-technology](http://cvit.iiit.ac.in/research/projects/cvit-projects/indian-language-script-technology)
- **Recommended Use**: Evaluating font variation tolerance (Akshar, Shree, dynamic Indian fonts).

### 1.3 MIDAS Indian ID Card & Document Benchmark
- **Overview**: Benchmark dataset focusing on scanned and camera-captured Indian documents:
  - Permanent Account Number (PAN) cards
  - Aadhaar card snippets
  - Driving Licenses & Voter IDs
  - Indian GST Tax Invoices
- **Website / GitHub**: [https://github.com/midas-research](https://github.com/midas-research)
- **Recommended Use**: Fine-tuning for document-understanding pipelines (Phase 6 of `doc.md`).

---

## 2. English Business Document Datasets

### 2.1 SROIE (Scanned Receipts OCR and Information Extraction)
- **Overview**: ICDAR 2019 competition dataset with 1,000 scanned receipts.
- **Annotations**: Word-level bounding boxes and full text transcriptions.
- **Hugging Face**: `darentang/sroie`
- **Download Command**:
  ```bash
  python -m datasets.download_datasets --download sroie
  ```

### 2.2 DocILE Benchmark (Document Information Localization and Extraction)
- **Overview**: 6,700 real business documents (invoices, purchase orders, credit notes).
- **Scale**: 100,000+ line items and structured key-value annotations.
- **Website**: [https://docile.fit.vutbr.cz/](https://docile.fit.vutbr.cz/)

### 2.3 FUNSD (Form Understanding in Noisy Scanned Documents)
- **Overview**: 199 real scanned noisy forms with varied alignments, stamp markings, and low resolution.
- **Hugging Face**: `nielsr/funsd`
- **Download Command**:
  ```bash
  python -m datasets.download_datasets --download funsd
  ```

---

## 3. Bundled Starter Dataset

This repository includes an immediately runnable starter dataset in `datasets/`:

| Directory / File | Description | Count |
| :--- | :--- | :--- |
| `datasets/images/` | Rendered document lines (English invoices, Hindi GST lines, Tamil/Telugu/Marathi terms) | 31 images |
| `datasets/annotations/train.csv` | Training split (`image_path`, `text`, `language`) | 18 samples |
| `datasets/annotations/validation.csv` | Validation split | 5 samples |
| `datasets/annotations/test.csv` | Held-out test split | 8 samples |
| `datasets/synthetic/` | 50+ auto-generated synthetic bilingual document lines | 50+ images |

---

## 4. How to Convert External Data to PaddleOCR Format

PaddleOCR requires text recognition labels in the following tab-delimited format:

```text
<relative_image_path>\t<ground_truth_transcription>
```

Example (`datasets/paddle_rec/train.txt`):
```text
datasets/images/en_001.png	INVOICE NUMBER: INV-2026-8941
datasets/images/hi_001.png	कर बीजक (Tax Invoice)
datasets/images/hi_002.png	ग्राहक का नाम: अमित कुमार शर्मा
datasets/images/ta_001.png	விலைப்பட்டியல் எண்: டிஎன்-1082
```

Use the built-in dataset preparation utility to convert any CSV or split directory:

```bash
python -m training.prepare_dataset \
  --annotations-dir datasets/annotations \
  --output-dir datasets/paddle_rec \
  --dict-output datasets/dicts/custom_dict.txt
```

### Script & Character Dictionary Tip
When training on Indian languages:
1. Always normalize labels to **Unicode NFC** (`unicodedata.normalize("NFC", text)`).
2. Ensure the character dictionary (`datasets/dicts/custom_dict.txt`) includes all *matras*, *viramas*, and conjunct characters present in your training set.
3. Keep test documents separated by document source rather than random line splits to prevent data leakage.
