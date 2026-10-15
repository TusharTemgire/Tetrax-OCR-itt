"""
Hugging Face TrOCR fine-tuning script for English and Indian documents.
Directly loads sarvamai/indic-ocr-bench from Hugging Face:
    from datasets import load_dataset
    ds = load_dataset("sarvamai/indic-ocr-bench")
No local image or CSV files required on disk.
"""
import argparse
import os
import unicodedata
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
)

def normalize(text: str) -> str:
    return unicodedata.normalize("NFC", (text or "").strip())

class HuggingFaceOCRDataset(Dataset):
    """
    In-memory dataset wrapper for Hugging Face Dataset objects
    (e.g., sarvamai/indic-ocr-bench) with on-the-fly tokenization.
    """
    def __init__(self, hf_dataset, processor: TrOCRProcessor, max_target_length: int = 64):
        self.dataset = hf_dataset
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        pil_img = item["image"]
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")

        text = normalize(item.get("gt", ""))

        pixel_values = self.processor(pil_img, return_tensors="pt").pixel_values.squeeze(0)

        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
            return_tensors="pt"
        ).input_ids.squeeze(0)

        # Ignore pad tokens in loss
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {"pixel_values": pixel_values, "labels": labels}

def main():
    parser = argparse.ArgumentParser(description="Fine-tune TrOCR directly using sarvamai/indic-ocr-bench.")
    parser.add_argument("--dataset", default="sarvamai/indic-ocr-bench", help="Hugging Face dataset identifier")
    parser.add_argument("--split", default="small_representative", choices=["small_representative", "test"],
                        help="'small_representative' (1,173 samples) or 'test' (6,909 samples)")
    parser.add_argument("--languages", default="all",
                        help="Target languages (e.g., 'hindi,english', 'tamil,telugu', or 'all')")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional max sample limit")
    parser.add_argument("--val-split-ratio", type=float, default=0.1, help="Validation split percentage")
    parser.add_argument("--output-dir", default="models/checkpoints/trocr-indic", help="Output directory for fine-tuned weights")
    parser.add_argument("--base-model", default="microsoft/trocr-small-printed", help="Pretrained Hugging Face base model")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--freeze-encoder", action="store_true", default=True, help="Freeze vision encoder for fast low-memory training")
    args = parser.parse_args()

    print(f"Loading '{args.dataset}' (split: {args.split}) directly from Hugging Face...")
    from datasets import Image as HFImage
    ds = load_dataset(args.dataset, split=args.split)
    ds = ds.cast_column("image", HFImage(decode=False))

    def _is_valid_image(item):
        img = item.get("image")
        if isinstance(img, dict):
            if img.get("bytes"):
                return True
            p = img.get("path")
            return bool(p and os.path.exists(p))
        return img is not None

    ds = ds.filter(_is_valid_image)
    ds = ds.cast_column("image", HFImage(decode=True))
    print(f"Total raw samples loaded: {len(ds)}")

    # Filter by target languages if specified
    if args.languages.lower() != "all":
        target_langs = {l.strip().lower() for l in args.languages.split(",")}
        print(f"Filtering for languages: {target_langs}...")
        ds = ds.filter(lambda x: str(x.get("language", "")).lower() in target_langs)
        print(f"Filtered samples: {len(ds)}")

    if args.max_samples and len(ds) > args.max_samples:
        ds = ds.select(range(args.max_samples))
        print(f"Trimmed to {len(ds)} samples.")

    # Split into train and validation
    split_ds = ds.train_test_split(test_size=args.val_split_ratio, seed=42)
    train_hf = split_ds["train"]
    val_hf = split_ds["test"]

    print(f"Train samples: {len(train_hf)} | Validation samples: {len(val_hf)}")

    print(f"\nLoading TrOCR Processor & Base Model: {args.base_model}...")
    processor = TrOCRProcessor.from_pretrained(args.base_model)
    model = VisionEncoderDecoderModel.from_pretrained(args.base_model)

    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.max_length = 64
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 2

    if args.freeze_encoder:
        print("Freezing vision encoder for efficient fine-tuning...")
        for param in model.encoder.parameters():
            param.requires_grad = False

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.1f}%)")

    train_dataset = HuggingFaceOCRDataset(train_hf, processor, max_target_length=64)
    val_dataset = HuggingFaceOCRDataset(val_hf, processor, max_target_length=64)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Compute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    training_kwargs = {
        "predict_with_generate": True,
        "save_strategy": "epoch",
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "fp16": torch.cuda.is_available(),
        "learning_rate": args.lr,
        "output_dir": args.output_dir,
        "num_train_epochs": args.epochs,
        "logging_steps": 10,
        "save_total_limit": 1,
        "optim": "adafactor" if not torch.cuda.is_available() else "adamw_torch",
        "report_to": "none",
        "remove_unused_columns": False,
        "dataloader_pin_memory": False,
    }

    try:
        training_args = Seq2SeqTrainingArguments(eval_strategy="epoch", **training_kwargs)
    except TypeError:
        training_args = Seq2SeqTrainingArguments(evaluation_strategy="epoch", **training_kwargs)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
    )

    print("\nStarting fine-tuning...")
    trainer.train()

    print(f"\nFine-tuning complete! Saving model checkpoint to: {args.output_dir}")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print("Model and processor saved successfully.")

if __name__ == "__main__":
    main()
