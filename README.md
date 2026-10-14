# Tetrax OCR — Option A (PaddleOCR) & Indian Document Training Starter

A high-performance OCR service and fine-tuning starter for document images and multi-page PDFs, specialized in **English and Indian documents** (invoices, receipts, tax forms, IDs, and books) based on [`doc.md`](file:///d:/tetrax-ocr-starter/tetrax-ocr-starter/doc.md).

> 🚀 **Cloud Fine-Tuning Guides & Colab Notebook:**
> - [Open Directly in Google Colab (Notebook)](file:///d:/tetrax-ocr-starter/tetrax-ocr-starter/Tetrax_OCR_Colab_Training.ipynb)
> - [Google Colab Step-by-Step Training Guide (Markdown)](file:///d:/tetrax-ocr-starter/tetrax-ocr-starter/README_COLAB.md)
> - [RunPod Cloud GPU Step-by-Step Training Guide](file:///d:/tetrax-ocr-starter/tetrax-ocr-starter/README_RUNPOD.md)

---

## 1. Architecture Overview

Tetrax OCR implements **Option A (PaddleOCR)** from `doc.md`:
1. **Document Ingestion & Preprocessing**: Deskewing, CLAHE contrast enhancement, and noise reduction (`app/preprocessing.py`).
2. **Multi-page PDF Processing**: PyMuPDF (`fitz`) and `pypdfium2` page renderer (`app/pdf_processor.py`).
3. **PaddleOCR Engine**: Language routing across 13 Indian languages and English with polygon bounding boxes (`app/ocr_engine.py`).
4. **FastAPI Endpoints**: Validated REST API with structured Pydantic schemas (`app/main.py`).
5. **Training Pipelines**:
   - **Option A (PaddleOCR Rec)**: Training pipeline using PP-OCR recognition configs and generated Unicode character dictionaries.
   - **Option B (TrOCR)**: VisionEncoderDecoder transformer fine-tuning (`training/train_trocr.py`).
6. **Dataset Integrations**: Direct support for **Sarvam AI Indic OCR Bench** (`sarvamai/indic-ocr-bench`), AI4Bharat IndicOCR, and SROIE.

```
tetrax-ocr/
├── app/
│   ├── main.py              # FastAPI endpoints (/health, /v1/languages, /v1/ocr)
│   ├── ocr_engine.py        # PaddleOCR inference engine with language routing
│   ├── preprocessing.py     # Deskew, CLAHE contrast enhancement, denoise
│   ├── pdf_processor.py     # Multi-page PDF renderer (PyMuPDF / pypdfium2)
│   └── schemas.py           # Pydantic request and response schemas
├── training/
│   ├── load_sarvam_dataset.py  # Download and prepare sarvamai/indic-ocr-bench
│   ├── train_trocr.py          # PyTorch + Transformers Seq2SeqTrainer fine-tuning
│   ├── evaluate.py             # CER, WER, and Exact Match evaluation with NFC
│   ├── benchmark_sarvam.py     # Benchmark models against sarvamai/indic-ocr-bench
│   ├── generate_synthetic_data.py # Generate realistic bilingual Indian invoice lines
│   ├── generate_starter_dataset.py# Bundled starter dataset generator
│   └── config.py               # Hyperparameters, paths, and character dictionaries
├── configs/
│   └── rec_sarvam_indic.yml # PaddleOCR recognition training config
├── datasets/
│   ├── images/              # Bundled starter document lines (English & Indic)
│   ├── annotations/         # train.csv, validation.csv, test.csv
│   ├── synthetic/           # Auto-generated synthetic document crops
│   ├── dicts/               # Character dictionaries for Indian scripts
│   └── DATASETS_GUIDE.md    # Detailed guide on Indian & English OCR datasets
└── tests/
    └── test_ocr.py          # Automated test suite
```

---

## 2. Quickstart: Run the OCR API

### 2.1 Install Dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2.2 Start the API Server
```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
Open interactive documentation at: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

### 2.3 Test OCR Extraction
Extract text from an English or Hindi document:
```bash
# English invoice
curl -X POST http://127.0.0.1:8000/v1/ocr \
  -F "file=@datasets/images/en_001.png" \
  -F "lang=en"

# Hindi / Devanagari document
curl -X POST http://127.0.0.1:8000/v1/ocr \
  -F "file=@datasets/images/hi_001.png" \
  -F "lang=hi"
```

Supported language codes:
- `en` (English)
- `hi` (Hindi)
- `mr` (Marathi)
- `ta` (Tamil)
- `te` (Telugu)
- `kn` (Kannada)
- `ml` (Malayalam)
- `bn` (Bengali)
- `gu` (Gujarati)
- `pa` (Punjabi)
- `ur` (Urdu)
- `or` (Odia)
- `sa` (Sanskrit)

---

## 3. Training on `sarvamai/indic-ocr-bench`

[Sarvam AI Indic OCR Bench](https://huggingface.co/datasets/sarvamai/indic-ocr-bench) contains 6,909 curated samples across 22 Indian languages and English.

### Step 3.1: Download and Prepare the Dataset
Use the built-in loader to download the images, split train/validation/test sets, and automatically extract the character dictionary:

```bash
# Prepare Hindi and English samples (quick subset)
python -m training.load_sarvam_dataset \
  --split small_representative \
  --languages hindi,english \
  --output-dir datasets/sarvam_indic

# Or extract all 22 Indian languages:
python -m training.load_sarvam_dataset \
  --split small_representative \
  --languages all \
  --output-dir datasets/sarvam_indic
```

This creates:
- `datasets/sarvam_indic/images/`: Extracted document images
- `datasets/sarvam_indic/annotations/`: `train.csv`, `validation.csv`, `test.csv`
- `datasets/sarvam_indic/paddle_rec/`: `train.txt`, `val.txt`, `test.txt` formatted for PaddleOCR
- `datasets/sarvam_indic/dicts/sarvam_dict.txt`: Complete Unicode character set

---

### Step 3.2: Fine-Tune using PaddleOCR (Option A)

PaddleOCR's recognition engine (PP-OCRv4 / PP-OCRv5) uses CTC + Attention multi-head architecture:

1. Clone the PaddleOCR training repository:
   ```bash
   git clone https://github.com/PaddlePaddle/PaddleOCR.git vendor/PaddleOCR
   cd vendor/PaddleOCR
   ```
2. Run training with the pre-configured YAML config [`configs/rec_sarvam_indic.yml`](file:///d:/tetrax-ocr-starter/tetrax-ocr-starter/configs/rec_sarvam_indic.yml):
   ```bash
   python tools/train.py \
     -c ../../configs/rec_sarvam_indic.yml \
     -o Train.dataset.data_dir=../../datasets/sarvam_indic \
        Train.dataset.label_file_list='["../../datasets/sarvam_indic/paddle_rec/train.txt"]' \
        Eval.dataset.data_dir=../../datasets/sarvam_indic \
        Eval.dataset.label_file_list='["../../datasets/sarvam_indic/paddle_rec/val.txt"]' \
        Global.character_dict_path=../../datasets/sarvam_indic/dicts/sarvam_dict.txt \
        Global.save_model_dir=./output/sarvam_indic_rec
   ```

---

### Step 3.3: Fine-Tune using TrOCR (Option B from `doc.md` Phase 3)

If training a Vision-Encoder-Decoder model with Hugging Face:
```bash
python -m training.train_trocr \
  --train-csv datasets/sarvam_indic/annotations/train.csv \
  --val-csv datasets/sarvam_indic/annotations/validation.csv \
  --output-dir models/checkpoints/trocr-tetrax \
  --epochs 5 \
  --batch-size 8
```

---

## 4. Evaluation and Benchmarking

### Evaluate Predictions (CER / WER / Exact Match)
```bash
python -m training.evaluate --predictions datasets/predictions.csv
```

### Benchmark Directly on Sarvam Indic OCR Bench
Measure Character Error Rate and Word Error Rate of the baseline or your fine-tuned model:
```bash
python -m training.benchmark_sarvam \
  --split small_representative \
  --languages hindi,english \
  --max-samples 50
```

---

## 5. Running Automated Tests
```bash
python -m unittest tests/test_ocr.py
```
