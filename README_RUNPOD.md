# Step-by-Step Guide: Fine-Tuning OCR on RunPod (Cloud GPU)

This guide walks you through setting up a dedicated GPU instance on **RunPod**, preparing the **`sarvamai/indic-ocr-bench`** dataset, and fine-tuning your custom OCR model using **Option A (PaddleOCR)** or **Option B (TrOCR)**.

---

## 1. Deploy a GPU Pod on RunPod

### Step 1.1: Select a GPU
1. Log in to [RunPod.io](https://www.runpod.io/) and navigate to **Pods** > **Deploy Pod**.
2. Select a Community or Secure Cloud GPU:
   - **Cost-effective**: RTX 3090 (24GB VRAM) or RTX 4090 (24GB VRAM) — *Ideal for fast OCR training.*
   - **Enterprise**: A5000 or A6000.
3. Click **Deploy**.

### Step 1.2: Select Pod Template & Volume Storage
- **Template**: Choose **`RunPod PyTorch 2.2+`** or **`Ubuntu 22.04 with CUDA 12.1`**.
- **Container Disk**: 20 GB.
- **Volume Disk (Persistent)**: At least **50 GB** mounted at `/workspace` (persists files across pod restarts).
- Click **Continue** and deploy the pod.

### Step 1.3: Connect to the Pod
Once the pod status changes to **Running**:
- **Option A (Browser)**: Click **Connect** > **Connect to Web Terminal** (recommended for quick access).
- **Option B (SSH)**: Add your public SSH key in RunPod Settings, then run:
  ```bash
  ssh root@<POD_IP> -p <PORT> -i ~/.ssh/id_rsa
  ```

---

## 2. Set Up the Environment in RunPod

All operations should be run in `/workspace` so code, datasets, and checkpoints persist.

```bash
cd /workspace

# Update system packages and install required graphics libraries
apt-get update && apt-get install -y libgl1 libglib2.0-0 tmux htop git git-lfs

# Verify GPU availability
nvidia-smi
```

Clone your Tetrax OCR repository:
```bash
git clone https://github.com/YOUR_USERNAME/tetrax-ocr.git /workspace/tetrax-ocr
cd /workspace/tetrax-ocr
```

Create and activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

Install core dependencies:
```bash
pip install -r requirements.txt
pip install datasets accelerate jiwer sentencepiece tiktoken
```

---

## 3. Prepare `sarvamai/indic-ocr-bench` Dataset

Download and format the dataset for OCR training:

```bash
cd /workspace/tetrax-ocr

# Download and preprocess English and Hindi data
python -m training.load_sarvam_dataset \
  --split small_representative \
  --languages hindi,english \
  --output-dir datasets/sarvam_indic

# Or prepare all 22 Indian languages from full benchmark:
# python -m training.load_sarvam_dataset --split test --languages all --output-dir datasets/sarvam_indic
```

Verify output files:
```bash
ls -lh datasets/sarvam_indic/paddle_rec/
cat datasets/sarvam_indic/dicts/sarvam_dict.txt | head -n 30
```

---

## 4. Option A: Fine-Tune PaddleOCR on RunPod (Recommended)

### Step 4.1: Install PaddlePaddle GPU for RunPod CUDA
On RunPod (typically CUDA 12.x):
```bash
pip install paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/stable/cu120/
pip install "paddleocr>=2.7"
```

Verify Paddle GPU installation:
```python
python3 -c "import paddle; print('CUDA Available:', paddle.is_compiled_with_cuda(), '| GPU:', paddle.device.cuda.get_device_name(0))"
```

### Step 4.2: Clone PaddleOCR Training Framework
```bash
cd /workspace
git clone https://github.com/PaddlePaddle/PaddleOCR.git vendor/PaddleOCR
cd vendor/PaddleOCR
pip install -r requirements.txt
```

Download pretrained multilingual recognition weights:
```bash
mkdir -p pretrain_models
cd pretrain_models
wget -nc https://paddleocr.bj.bcebos.com/PP-OCRv4/multilingual/multilingual_PP-OCRv4_rec_train.tar
tar -xf multilingual_PP-OCRv4_rec_train.tar
cd /workspace/vendor/PaddleOCR
```

### Step 4.3: Launch Training Inside a `tmux` Session
Using `tmux` ensures that if your browser or SSH connection disconnects, training will continue uninterrupted in the background:

```bash
# Start a new tmux session
tmux new -s tetrax_train
```

Launch the fine-tuning run:
```bash
python tools/train.py \
  -c /workspace/tetrax-ocr/configs/rec_sarvam_indic.yml \
  -o Global.use_gpu=True \
     Global.epoch_num=50 \
     Global.pretrained_model=./pretrain_models/multilingual_PP-OCRv4_rec_train/best_accuracy \
     Global.character_dict_path=/workspace/tetrax-ocr/datasets/sarvam_indic/dicts/sarvam_dict.txt \
     Global.save_model_dir=/workspace/output/tetrax_paddle_rec \
     Train.dataset.data_dir=/workspace/tetrax-ocr \
     Train.dataset.label_file_list='["/workspace/tetrax-ocr/datasets/sarvam_indic/paddle_rec/train.txt"]' \
     Train.loader.batch_size_per_card=64 \
     Eval.dataset.data_dir=/workspace/tetrax-ocr \
     Eval.dataset.label_file_list='["/workspace/tetrax-ocr/datasets/sarvam_indic/paddle_rec/val.txt"]'
```

*Tips for `tmux`:*
- To detach from session: Press `Ctrl + B`, then press `D`.
- To reattach later: `tmux attach -t tetrax_train`.
- To monitor GPU utilization in another terminal: `watch -n 1 nvidia-smi`.

### Step 4.4: Export Trained Model for Production Inference
Convert the training checkpoint to lightweight static inference models:
```bash
python tools/export_model.py \
  -c /workspace/tetrax-ocr/configs/rec_sarvam_indic.yml \
  -o Global.checkpoints=/workspace/output/tetrax_paddle_rec/best_accuracy \
     Global.save_inference_dir=/workspace/output/tetrax_inference_model
```

---

## 5. Option B: Fine-Tune TrOCR on RunPod

Fine-tune directly using **`sarvamai/indic-ocr-bench`** in memory without needing local images or CSVs on disk:

```bash
cd /workspace/tetrax-ocr

tmux new -s trocr_train

python -m training.train_trocr \
  --dataset "sarvamai/indic-ocr-bench" \
  --split small_representative \
  --languages hindi,english \
  --output-dir /workspace/output/trocr-tetrax \
  --epochs 5 \
  --batch-size 8 \
  --lr 5e-5
```

---

## 6. Evaluation and Accuracy Reporting

Run evaluation across held-out test data:
```bash
cd /workspace/tetrax-ocr

python -m training.benchmark_sarvam \
  --split small_representative \
  --languages hindi,english \
  --max-samples 100
```

---

## 7. Download Model Checkpoints from RunPod

When training finishes, download your exported model files:

### Method 1: RunPod Web File Manager
1. In the RunPod dashboard, click **Connect** > **HTTP Service** or open the built-in Jupyter / File Browser.
2. Navigate to `/workspace/output/tetrax_inference_model`.
3. Right-click and choose **Download**.

### Method 2: SCP / RSYNC (From your local machine)
```bash
# Replace with your RunPod IP and SSH port
rsync -avz -P -e "ssh -p <PORT>" root@<POD_IP>:/workspace/output/tetrax_inference_model ./models/
```

### Method 3: Cloud Sync / Hugging Face Hub
You can upload the checkpoint directly to the Hugging Face Hub:
```bash
pip install huggingface_hub
huggingface-cli login
huggingface-cli upload your-username/tetrax-indic-ocr /workspace/output/tetrax_inference_model
```

---

## 8. RunPod Best Practices
1. **Always work in `/workspace`**: Files outside `/workspace` will be erased if the pod terminates.
2. **Use `tmux`**: Protects long-running training jobs from connection loss.
3. **Stop Pod when idle**: Stop the pod when training finishes to avoid incurring unnecessary cloud costs (persistent volume storage remains saved).
