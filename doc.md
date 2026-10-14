# Build your own Image-to-Text AI model for Tetrax — step by step

You can build your own image-to-text extraction service by starting with a pretrained OCR model, fine-tuning it on your own document images, and exposing it through an API. The system can extract text from photographs, scanned documents, invoices, receipts, forms, IDs, and other document images.

My recommendation: Don't train a complete AI model from scratch initially. Start with an existing OCR model, build a reliable extraction pipeline, fine-tune it using your own dataset, and gradually develop your own model and API.

For Tetrax, you can build a service similar to a document OCR API, where customers upload an image or PDF and receive extracted text, structured JSON, or a searchable document.

## 1. Decide what your model should extract

[Review Opinion Letters: How They Differ from Full Valuation Reporting](https://images.openai.com/static-rsc-4/rWHKMGcpfUdbYJEBeUkYMzDgzS0xD5-rnb-wvKD50waj8qgFkzCS48gaaeHLtBC7Fw5Uw_nlIo0nMel3PxpCJxKzANNTaT5_pOEh1_Y_D5bGR3lmktQ_WQKeUF5OblPBG2PIq9RARbWhkBx2jq9tOUJcR2LGzreohylPhCQ5IgQ?purpose=inline)

Printed text extraction

Extract paragraphs, headings, tables, numbers, and other text from photographs and scanned pages.

Example: JPG → editable text

[10 Free Basic Invoice Templates](https://images.openai.com/static-rsc-4/Fcf9J8DVp7m-uAfoQMpJXpBAiLNJkZicPpQI0_BKA84-wuDId3kCPMYNkr1hyRNZoEEdI9Im9iKABGjURh_zpX3uhl4PdYa20LHRPvd_u03DiefmoccqiH2KcCLY4O7tEPs8Xdadz2Ckgka0P8Z1xsOkHOsL-tj-SsRX5AroWOU?purpose=inline)

Document understanding

Identify invoice numbers, dates, customer details, totals, tax amounts, and table rows.

Example: Invoice → structured JSON

[Custom Letter Writing Service, Personal Scribe, Letter to a Friend, Family Notes, Wedding Speech, Penpal, Journal, Handwriting - Etsy](https://images.openai.com/static-rsc-4/noicafWEk5rQIonYDuHq7Gn80tXrioakjmueBgyOIl3CREI5L9Dnjp3SE-NsBt46tNeLfOl_eQwp94rt73w_bzIeyM8ppWwVupgKMs2pYAK1dS-cQKzPNRzsjtLFGe5dlYLWhRulTPEiVPL0623KA41aJ665JbEs2Sw0XZamljQ?purpose=inline)

Handwriting recognition

Recognize handwritten notes and forms. This requires appropriate handwriting training data and may need a separate model.

[Mishaal Rahman (@MishaalRahman) on X](https://images.openai.com/static-rsc-4/1TE_VyuBlbqzl8YGxVjVnj5-9CNWvaq32JkcN034X5NFd4HJl-vsOfGCMjOK2k29SxYGekC8sSxB3C57r5le9eXdNwiR9GRalk2LdxWVoBluPrCuXBsb7PUY4QXr9MtCIYEYq8xqJwJzT3ETqNPPWdUMhEOMe-NSwqu0YOBEENQ?purpose=inline)

Real-world image extraction

Handle tilted photos, shadows, low resolution, rotated pages, and complex layouts using image preprocessing and OCR.

You don't need four independently trained models for the first version. Begin with one general-purpose OCR model and add specialized document processing as needed.

## 2. Choose the model you will train

There are three practical approaches. They solve related but different problems.

[ocr - How to сorrectly recognize 90 and 270 degrees rotated text? - Stack Overflow](https://images.openai.com/static-rsc-4/YUO-WSPLYyytEv_6qMUpAwHPkxkrDSv7eXoiyXnIy2RV1YtReZ3mJwWSWWZz5WsLFZdp1-qvbP78i714n_3_iKAAc63vojGQUzS9taWx6jEatSQ2as6aYv8kbvjSiF1_hQtlYA5as6GhvJcz09MO4lbfitBp0zMnFKz_bazoFHo?purpose=inline)

Option A — PaddleOCR

Recommended starting point

A practical OCR toolkit for detecting text regions and recognizing printed text. Use it to build your first working API, then fine-tune the recognition component if your chosen model and training pipeline support your data.

[A Comprehensive Evaluation of TrOCR with Varying Image Effects - NHSJS](https://images.openai.com/static-rsc-4/wM21te7Zzj-tmIX4LUHRs2_9Bv3HGTaZSuEwwOEPtyHoPGT4anOW8shFM7MVhwXp0ql8uGT-YI_moZkKP1Lj3wYjjsUgxbFlyEa3j_5yikGvVJOD_3qNU7Q7Y5a2i7oW6DoiXab1__5nM8MWN9H2y0LKnOqNfj7xkJ6LOw_Xwi0?purpose=inline)

Option B — TrOCR

A transformer-based OCR model that predicts text from image inputs. Suitable for a more direct image-to-text fine-tuning experiment, especially for cropped text lines.

[Why Data Chunking is Essential for LLM Processing | PyMuPDF](https://images.openai.com/static-rsc-4/_3SMdTaPf2Ab2zh3vDmF2V3LfE4byurKH8prMSp5Td6O8A0xBQsWyrPqxmj0mbYmV8LdlLVakvZlLahDk001EAa5fsLl5lSRPWd8-AUWoG5i9o6syRC0PpW6ZSqRm-D2D7ZWSUrmQzPFlMlMI-MesSSZPRP675_nW0GdKmMMb8Y?purpose=inline)

Option C — Document AI model

For documents with complex layouts, tables, and fields, combine OCR with layout analysis or fine-tune a document-understanding model. This is a later phase rather than your first task.

My choice for your project: PaddleOCR for the initial production pipeline, followed by a dedicated recognition-model fine-tuning experiment. If your main objective is specifically to fine-tune a pretrained image-to-text transformer, start with TrOCR instead.

A useful distinction: OCR reads text; document understanding identifies what the text means and where it belongs.

## 3. The complete architecture

Input document

JPG · PNG · TIFF · PDF

Image preprocessing

Orientation · deskew · contrast · page rendering

Your fine-tuned OCR model

Text detection + recognition

Post-processing

Reading order · paragraphs · tables · validation

Tetrax OCR API

Extracted text · searchable documents · structured JSON

Your platform will have two separate workflows:

- Training workflow: prepare labeled data → fine-tune → evaluate → version the model.
- Inference workflow: upload image → process image → run model → return results.

Training should happen on an on-demand GPU machine. Your API can run on a separate CPU or GPU server depending on throughput and latency requirements.

## 4. Phase-by-phase development roadmap

Phase 1 — Build a working OCR baseline

Install an existing OCR model, upload images, extract text, support multi-page PDFs, and measure accuracy before changing model weights.

Deliverable: working local OCR application.

Phase 2 — Create your training dataset

Collect representative document images and their exact transcriptions. Label text lines or regions as required by the selected model.

Deliverable: reproducible training, validation, and test datasets.

Phase 3 — Fine-tune your model

Start from pretrained weights, train on your labeled examples, save checkpoints, and monitor validation error.

Deliverable: your own fine-tuned checkpoint.

Phase 4 — Evaluate and improve

Compare the baseline and fine-tuned model on unseen documents. Measure character error rate, word error rate, and field extraction accuracy.

Deliverable: documented accuracy and failure cases.

Phase 5 — Build your own API

Serve the model using FastAPI, add file validation, authentication, asynchronous processing, and API usage tracking.

Deliverable: private OCR API with versioned models.

Phase 6 — Build document intelligence

Add tables, key-value fields, document classification, searchable PDFs, and structured invoice or form extraction.

Deliverable: a more complete document-processing service.

## 5. Set up your development environment

I recommend Python for training and inference, FastAPI for the OCR service, and Next.js/React for your Tetrax dashboard.

Recommended technology stack

| Model training     | PyTorch + Hugging Face Transformers |
| ------------------ | ----------------------------------- |
| Initial OCR engine | PaddleOCR                           |
| Image processing   | OpenCV + Pillow                     |
| PDF processing     | PyMuPDF                             |
| API                | FastAPI + Uvicorn                   |
| Training hardware  | On-demand NVIDIA GPU                |
| Dashboard          | Next.js + TypeScript                |

### Step 5.1 — Create the project

On Windows, open PowerShell:

```
mkdir tetrax-ocr
cd tetrax-ocr

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
```

Create these directories:

```
mkdir app, training, datasets, models, tests
mkdir datasets\images
mkdir datasets\annotations
mkdir models\checkpoints
```

For the initial OCR baseline, install:

```
pip install "paddleocr==2.7.3" "paddlepaddle>=2.6,<3"
pip install opencv-python pillow pymupdf
pip install fastapi uvicorn python-multipart
```

These commands target the PaddleOCR 2.x API. Check that the PaddlePaddle package supports your Python version and operating system. For GPU training, use a compatible PyTorch/CUDA environment on your cloud GPU rather than assuming the local Windows setup will support every GPU configuration.

### Step 5.2 — Your project structure

```

tetrax-ocr/
├── app/
│   ├── main.py
│   ├── ocr_engine.py
│   ├── preprocessing.py
│   ├── pdf_processor.py
│   └── schemas.py
├── training/
│   ├── prepare_dataset.py
│   ├── train_trocr.py
│   ├── evaluate.py
│   └── config.py
├── datasets/
│   ├── images/
│   └── annotations/
│       ├── train.csv
│       ├── validation.csv
│       └── test.csv
├── models/
│   └── checkpoints/
├── tests/
│   └── test_ocr.py
├── requirements.txt
└── README.md

```

## 6. Phase 1 — Make your first image-to-text model work

Before fine-tuning anything, verify that a pretrained OCR engine can read your images.

Create `app/ocr_engine.py`:

```
from paddleocr import PaddleOCR# Initialize once, not on every API request.ocr_engine = PaddleOCR(    use_angle_cls=True,    lang="en",    use_gpu=False,)def extract_text(image_path: str) -> dict:    result = ocr_engine.ocr(image_path, cls=True)    lines = []    if result and result[0]:        for item in result[0]:            box, recognition = item            text, confidence = recognition            lines.append({                "text": text,                "confidence": float(confidence),                "bounding_box": box,            })    return {        "text": "\n".join(line["text"] for line in lines),        "lines": lines,    }if __name__ == "__main__":    output = extract_text("sample.png")    print(output["text"])
```

Put a test image named `sample.png` in the project root, then run:

```
python -m app.ocr_engine
```

Expected output might look like this:

```
INVOICE
Invoice Number: INV-1045
Date: 12/10/2026
Subtotal: 2500.00
Tax: 450.00
Total: 2950.00
```

The output above is illustrative. Actual accuracy depends on the image, text quality, language, and model.

At this stage, you've built an OCR application, not yet a fine-tuned model. Save its results as your baseline so you can prove whether training improves accuracy later.

## 7. Phase 2 — Build your own training dataset

Fine-tuning requires paired examples: the image and its correct text transcription.

For example:

[python - create bounding box on meaningful word - Stack Overflow](https://images.openai.com/static-rsc-4/fImSuuxbjpPFabpZNvJh-MIU3PuiwjrBxONjhCmP-d2702eS1AnV5QwaKmMoRF0xKrCVDLyPrVt7sdkquXnyjvxmba8UejzQH8liPDjU4OdnyhD7J9E6K9vHSgLQggniyfXFaf55AqIThVmTIvNd5nxuHPFYLUnKUGrgmwNPnVE?purpose=inline)

Input image

A document or cropped text line

GROUND-TRUTH TRANSCRIPTION

`Invoice Number: INV-1045`

### How much data do you need?

These are practical starting targets, not guarantees of accuracy.

| Training goal                                   | Initial target                             |
| ----------------------------------------------- | ------------------------------------------ |
| Test the training pipeline                      | 500–2,000 labeled text-line images         |
| Fine-tune for a narrow document domain          | 5,000–20,000 labeled examples              |
| Broader variation in fonts and document quality | 20,000+ examples, often substantially more |

You can start with a smaller dataset to test the process. Include different fonts, image resolutions, lighting conditions, document layouts, and the languages you intend to support.

### Step 7.1 — Prepare the annotation CSV files

For TrOCR fine-tuning, each image should ideally contain one line or a short, clearly readable text region. Use the following CSV format:

`datasets/annotations/train.csv`

```
image_path,text
images/line_0001.png,"Invoice Number: INV-1045"
images/line_0002.png,"Customer Name: Rahul Sharma"
images/line_0003.png,"Total Amount: 2950.00"
```

Create equivalent `validation.csv` and `test.csv` files.

- Training set: teaches the model.
- Validation set: helps select the best checkpoint.
- Test set: evaluates the final model on examples it has never seen.

Keep all text lines from the same original document in the same split. Otherwise, near-duplicate examples can leak into validation and test data, making the model appear more accurate than it really is.

### Step 7.2 — Where to get training data

Useful public datasets to investigate include:

- FUNSD — noisy scanned forms and document understanding.
- CORD — receipt images with structured annotations.
- IAM Handwriting Database — handwritten English text.

Review each dataset's license, access terms, and annotation format before using it commercially. These datasets serve different purposes; a form-understanding dataset is not automatically a ready-to-use text-line OCR dataset.

You can also create synthetic documents with known ground-truth text to supplement real-world images. Synthetic data helps with font and layout variation but does not replace real images for testing.

## 8. Phase 3 — Fine-tune your own image-to-text model

For this experiment, use TrOCR's pretrained printed-text checkpoint. It predicts a text sequence from an image, so it is a good way to learn model fine-tuning directly.

Important limitation: TrOCR expects a cropped text line or text region, not an entire arbitrary page with multiple columns and tables. Use PaddleOCR for page-level text detection, or add a detection model to locate text lines before passing each crop to TrOCR.

### Step 8.1 — Prepare a separate training environment

On your cloud GPU instance, create a Python environment with a compatible PyTorch build and install:

```
pip install torch torchvision
pip install transformers datasets accelerate
pip install pandas pillow evaluate jiwer
```

Check that the GPU is available:

```
import torchprint("CUDA available:", torch.cuda.is_available())if torch.cuda.is_available():    print("GPU:", torch.cuda.get_device_name(0))
```

If CUDA is unavailable, resolve the PyTorch and GPU driver compatibility before beginning training.

### Step 8.2 — Create the training script

Create `training/train_trocr.py`.

This example loads your CSV annotations, fine-tunes a pretrained model, evaluates on the validation set, and saves the resulting checkpoint.

```
import osimport pandas as pdimport torchfrom PIL import Imagefrom torch.utils.data import Datasetfrom transformers import (    TrOCRProcessor,    VisionEncoderDecoderModel,    Seq2SeqTrainer,    Seq2SeqTrainingArguments,    default_data_collator,)BASE_MODEL = "microsoft/trocr-small-printed"OUTPUT_DIR = "models/checkpoints/trocr-tetrax"# CSV paths are relative to the project root.TRAIN_CSV = "datasets/annotations/train.csv"VAL_CSV = "datasets/annotations/validation.csv"processor = TrOCRProcessor.from_pretrained(BASE_MODEL)model = VisionEncoderDecoderModel.from_pretrained(BASE_MODEL)model.config.decoder_start_token_id = (    processor.tokenizer.cls_token_id)model.config.pad_token_id = processor.tokenizer.pad_token_idmodel.config.eos_token_id = processor.tokenizer.sep_token_idmodel.config.vocab_size = model.config.decoder.vocab_sizemodel.config.max_length = 128model.config.num_beams = 2class OCRDataset(Dataset):    def __init__(self, csv_path):
```

Before running: this script is a starter implementation. Verify the Transformers version supports `eval_strategy` and `processing_class`; older releases use different argument names. Also inspect your annotations, image paths, maximum text length, and model tokenizer settings before a full training run.

Start with a small dataset and a short run to catch configuration errors. Increase epochs and dataset size only after validating the pipeline.

### Step 8.3 — Test the fine-tuned checkpoint

```
from PIL import Imagefrom transformers import TrOCRProcessor, VisionEncoderDecoderModelcheckpoint = "models/checkpoints/trocr-tetrax"processor = TrOCRProcessor.from_pretrained(checkpoint)model = VisionEncoderDecoderModel.from_pretrained(checkpoint)model.eval()image = Image.open("datasets/images/line_0001.png").convert("RGB")inputs = processor(images=image, return_tensors="pt")import torchwith torch.no_grad():    generated_ids = model.generate(        inputs.pixel_values,        max_new_tokens=128,    )text = processor.batch_decode(    generated_ids, skip_special_tokens=True)[0]print(text)
```

On a GPU, move the model and inputs to the same CUDA device if you want GPU inference. The example above runs on CPU by default.

At this point, you have a separately saved checkpoint containing your fine-tuned weights. It is not a model trained entirely from scratch, but it is your own fine-tuned model.

## 9. Phase 4 — Measure whether training actually improved it

Never evaluate your model only on the images it trained on.

Use a fixed test set and compare the original model with the fine-tuned checkpoint.

Two key metrics are:

- Character Error Rate (CER): measures character-level substitutions, deletions, and insertions.
- Word Error Rate (WER): measures word-level substitutions, deletions, and insertions.

Install the evaluation library:

```
pip install jiwer
```

Create a simple evaluation script:

```
from jiwer import cer, werground_truth = [    "Invoice Number: INV-1045",    "Total Amount: 2950.00",]predictions = [    "Invoice Number: INV-1045",    "Total Amount: 2950.00",]print("CER:", cer(ground_truth, predictions))print("WER:", wer(ground_truth, predictions))
```

The example uses matching placeholder predictions to demonstrate the metric calculation. Replace them with actual model outputs from your test dataset.

A lower CER and WER indicate better transcription. Also inspect failures involving dates, decimal points, currency symbols, invoice IDs, and table boundaries. For business documents, a single incorrect digit may matter more than a small change in average CER.

## 10. Phase 5 — Build your own OCR API

Once the baseline works, expose it through an API so your Next.js dashboard or external customers can send documents to Tetrax.

Create `app/main.py`:

```
import osimport tempfilefrom pathlib import Pathfrom fastapi import FastAPI, UploadFile, File, HTTPExceptionfrom app.ocr_engine import extract_textapp = FastAPI(title="Tetrax OCR API", version="0.1.0")ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif"}MAX_BYTES = 10 * 1024 * 1024  # 10 MB@app.get("/health")def health():    return {"status": "ok", "service": "tetrax-ocr"}@app.post("/v1/ocr")async def ocr_document(file: UploadFile = File(...)):    suffix = Path(file.filename or "").suffix.lower()    if suffix not in ALLOWED_EXTENSIONS:        raise HTTPException(            status_code=415,            detail="Unsupported file type"        )    content = await file.read(MAX_BYTES + 1)    if not content or len(content) > MAX_BYTES:        raise HTTPException(            status_code=413,            detail="File is empty or exceeds the 10 MB limit"        )    temp_path = None
```

Run the server from the project root:

```
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` to test your endpoint interactively.

The endpoint above supports images only. Add PDF processing separately: render each PDF page to an image using PyMuPDF, run OCR on each page, and return page-numbered results. Enforce page-count limits and reject encrypted or malformed files where appropriate.

### Example API response

```

{
  "success": true,
  "filename": "invoice.jpg",
  "text": "Invoice Number: INV-1045\nTotal Amount: 2950.00",
  "lines": [
    {
      "text": "Invoice Number: INV-1045",
      "confidence": 0.98,
      "bounding_box": [[10, 20], [300, 20], [300, 45], [10, 45]]
    }
  ]
}

```

This response is illustrative. The actual confidence values and bounding boxes come from the OCR engine.

For production, add authentication, per-user rate limits, request IDs, structured logs, request-size limits at the reverse proxy, and an asynchronous job queue for large PDFs. Do not log sensitive document contents by default.

## 11. Phase 6 — Extract structured documents, not just text

After text extraction is reliable, build a second processing layer for invoices, forms, and other documents.

For example, convert extracted invoice text into:

```

{
  "document_type": "invoice",
  "invoice_number": "INV-1045",
  "invoice_date": "2026-10-12",
  "currency": "INR",
  "subtotal": 2500.00,
  "tax": 450.00,
  "total": 2950.00
}

```

This is an illustrative result, not a guaranteed output from OCR. Your document parser must validate the extracted fields against the source image. For example, check that subtotal plus tax matches the total where the document's rules permit that calculation.

A sensible processing pipeline is:

1. Detect and recognize text.
2. Preserve text positions and reading order.
3. Identify tables and key-value pairs.
4. Classify the document type.
5. Convert extracted content to a validated JSON schema.
6. Flag low-confidence fields for review.

OCR confidence is not the same as the probability that a complete field is correct. Validate important fields independently.

## 12. Phase 7 — Deploy the service

Next.js frontend

Upload documents, preview extracted text, review results, download output

Nginx + HTTPS

TLS, request-size limits, reverse proxy, rate limiting

FastAPI inference service

Model loading, OCR jobs, validation and API authentication

Storage and job tracking

PostgreSQL for metadata; private object storage for documents when needed

For your initial version, use a CPU server for low-volume OCR if performance is adequate. Move inference to a GPU when your latency or throughput measurements justify it. Keep training on a separate on-demand GPU machine so model experiments do not interrupt production.

Before accepting customer documents, implement authentication, private storage, retention and deletion policies, encryption, and tenant isolation.

## 13. What I recommend you build first

Don't attempt all document types at once. Follow this sequence:

1. Build the PaddleOCR baseline and test it against 20–50 representative images.
2. Save the baseline's extracted text and manually correct the ground-truth transcriptions.
3. Prepare a small text-line dataset and run a TrOCR fine-tuning experiment on a cloud GPU.
4. Compare the original and fine-tuned models on the same held-out test set.
5. Build the FastAPI service and connect it to a simple Next.js upload page.
6. Add PDF support, table extraction, and structured JSON after the OCR results are reliable.

One final decision will help determine the best training pipeline for your needs.

## Customize your Tetrax OCR build plan

Choose your priorities so the next step can include the exact dataset format, training script, and deployment configuration you need.

1\. What documents do you want to support first?

All images and documents — general OCR

Invoices, receipts, and business documents

Handwritten text and notes

Forms, tables, and structured fields

2\. What is your model-building goal?

Fine-tune a pretrained model first (recommended)

Train a model from scratch for research

Build a baseline, fine-tune it, and later train from scratch

3\. Where will you train?

On-demand cloud GPU

My own computer with an NVIDIA GPU

CPU only

4\. Which languages?

English first

English + Indian languages

Multilingual documents from the start

Build my detailed implementation plan

You can also use the OpenAI Developers integration if you want help with the API architecture and production AI workflow directly in ChatGPT; it requires connecting the integration first.