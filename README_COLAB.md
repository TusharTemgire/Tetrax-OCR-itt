# Step-by-Step Guide: Fine-Tuning OCR on Google Colab

This guide walks you through fine-tuning your custom OCR model (both **Option A: PaddleOCR** and **Option B: TrOCR**) on **Google Colab** using the [**`sarvamai/indic-ocr-bench`**](https://huggingface.co/datasets/sarvamai/indic-ocr-bench) English & Indian document dataset.

---

## 1. Set Up Google Colab Environment

### Step 1.1: Open Colab and Enable GPU
1. Go to [Google Colab](https://colab.research.google.com/) and create a **New Notebook**.
2. Click **Runtime** > **Change runtime type**.
3. Under **Hardware accelerator**, select **T4 GPU** (free tier) or **A100 / V100** (Colab Pro).
4. Click **Save**.

### Step 1.2: Verify GPU Allocation
In the first Colab cell, run:
```python
!nvidia-smi
```
*Expected output: You should see an NVIDIA GPU (e.g., Tesla T4) with Driver Version and CUDA Version (typically 12.x).*

---

## 2. Mount Google Drive & Clone Repository

Mount Google Drive so your trained model checkpoints are saved permanently and won't be lost when the Colab session disconnects.

```python
from google.colab import drive
drive.mount('/content/drive')
```

Clone your Tetrax OCR repository into Colab (or open [`Tetrax_OCR_Colab_Training.ipynb`](file:///d:/tetrax-ocr-starter/tetrax-ocr-starter/Tetrax_OCR_Colab_Training.ipynb)):
```bash
!git clone https://github.com/TusharTemgire/tetrax-OCR-ITT.git /content/tetrax-ocr
%cd /content/tetrax-ocr
```

*(Alternative without Git: Zip your local `tetrax-ocr-starter` folder, upload it to Google Drive, and unzip with `!unzip /content/drive/MyDrive/tetrax-ocr-starter.zip -d /content/tetrax-ocr`).*

---

## 3. Install Dependencies in Colab

```bash
# Update pip
!python -m pip install --upgrade pip

# Install PyTorch, Hugging Face, OpenCV, PyMuPDF, tokenizers, and evaluation tools
!pip install datasets transformers accelerate jiwer pillow opencv-python-headless pypdfium2 tqdm sentencepiece tiktoken
```

---

## 4. Download and Prepare `sarvamai/indic-ocr-bench`

Extract images, split train/val/test partitions, and generate character dictionaries for Indian languages and English:

```bash
# Option A: Fast test run on Hindi and English (small_representative split)
!python -m training.load_sarvam_dataset \
  --split small_representative \
  --languages hindi,english \
  --output-dir datasets/sarvam_indic

# Option B: Or load all 22 Indian languages:
# !python -m training.load_sarvam_dataset --split small_representative --languages all --output-dir datasets/sarvam_indic
```

Verify the dataset layout in Colab:
```bash
!ls -lh datasets/sarvam_indic/paddle_rec/
!head -n 5 datasets/sarvam_indic/paddle_rec/train.txt
!head -n 20 datasets/sarvam_indic/dicts/sarvam_dict.txt
```

---

## 5. Training Path 1: Option A — PaddleOCR (Recommended)

### Step 5.1: Install PaddlePaddle GPU for Colab CUDA
In Colab (CUDA 12.x), install the GPU build of PaddlePaddle and PaddleOCR:
```bash
!pip install paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/stable/cu120/
!pip install "paddleocr>=2.7"
```

Verify PaddlePaddle detects the Colab GPU:
```python
import paddle
print("Paddle version:", paddle.__version__)
print("CUDA available:", paddle.is_compiled_with_cuda())
print("Device count:", paddle.device.cuda.device_count())
```

### Step 5.2: Clone PaddleOCR Training Repo & Pretrained Weights
```bash
!git clone https://github.com/PaddlePaddle/PaddleOCR.git /content/PaddleOCR
%cd /content/PaddleOCR
!pip install -r requirements.txt
```

Download the official pretrained PP-OCRv4 / PP-OCRv5 multilingual recognition checkpoint:
```bash
!mkdir -p pretrain_models
%cd pretrain_models
# Download PP-OCR mobile/server multilingual recognition base weights
!wget -nc https://paddleocr.bj.bcebos.com/PP-OCRv4/multilingual/multilingual_PP-OCRv4_rec_train.tar
!tar -xf multilingual_PP-OCRv4_rec_train.tar
%cd /content/PaddleOCR
```

### Step 5.3: Launch PaddleOCR Fine-Tuning
Run training using the configuration prepared in [`configs/rec_sarvam_indic.yml`](file:///d:/tetrax-ocr-starter/tetrax-ocr-starter/configs/rec_sarvam_indic.yml):

```bash
!python tools/train.py \
  -c /content/tetrax-ocr/configs/rec_sarvam_indic.yml \
  -o Global.use_gpu=True \
     Global.epoch_num=30 \
     Global.pretrained_model=./pretrain_models/multilingual_PP-OCRv4_rec_train/best_accuracy \
     Global.character_dict_path=/content/tetrax-ocr/datasets/sarvam_indic/dicts/sarvam_dict.txt \
     Global.save_model_dir=/content/drive/MyDrive/tetrax_ocr_checkpoints \
     Train.dataset.data_dir=/content/tetrax-ocr \
     Train.dataset.label_file_list='["/content/tetrax-ocr/datasets/sarvam_indic/paddle_rec/train.txt"]' \
     Eval.dataset.data_dir=/content/tetrax-ocr \
     Eval.dataset.label_file_list='["/content/tetrax-ocr/datasets/sarvam_indic/paddle_rec/val.txt"]'
```

*Note: Models are automatically saved to your mounted Google Drive (`/content/drive/MyDrive/tetrax_ocr_checkpoints`), keeping them safe.*

### Step 5.4: Export Inference Model
Once training completes, export the dynamic model weights into static inference files (`inference.pdmodel`, `inference.pdiparams`):
```bash
!python tools/export_model.py \
  -c /content/tetrax-ocr/configs/rec_sarvam_indic.yml \
  -o Global.checkpoints=/content/drive/MyDrive/tetrax_ocr_checkpoints/best_accuracy \
     Global.save_inference_dir=/content/drive/MyDrive/tetrax_ocr_inference
```

---

## 6. Training Path 2: Option B — TrOCR Transformer (Hugging Face)

Fine-tune directly using **`sarvamai/indic-ocr-bench`** in memory without needing any local images or CSVs on disk:

```bash
%cd /content/tetrax-ocr

!python -m training.train_trocr \
  --dataset "sarvamai/indic-ocr-bench" \
  --split small_representative \
  --languages hindi,english \
  --output-dir /content/drive/MyDrive/trocr_tetrax_checkpoint \
  --epochs 3 \
  --batch-size 8 \
  --lr 5e-5
```

---

## 7. Evaluate Model Performance (CER & WER)

After training, evaluate the model on unseen test samples:

```bash
%cd /content/tetrax-ocr

# Run benchmark across languages
!python -m training.benchmark_sarvam \
  --split small_representative \
  --languages hindi,english \
  --max-samples 100
```

You will receive an accuracy breakdown:
```
=================================================================
    SARVAM INDIC OCR BENCHMARK RESULTS
=================================================================
Language: HINDI        | Samples:  50 | Exact: 92.0% | CER:  1.42% | WER:  3.10%
Language: ENGLISH      | Samples:  50 | Exact: 96.0% | CER:  0.85% | WER:  1.90%
=================================================================
```

---

## 8. Summary Checklist for Colab
- [x] Hardware accelerator set to **GPU (T4 or A100)**.
- [x] Google Drive mounted to `/content/drive`.
- [x] Dataset converted via `python -m training.load_sarvam_dataset`.
- [x] Training outputs saved directly to Drive.
- [x] Model exported for deployment.
