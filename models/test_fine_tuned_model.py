"""
Loads a fine-tuned VisionEncoderDecoder / TrOCR checkpoint
and runs inference on an input image.
"""
import argparse
from pathlib import Path
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

def extract_text_with_finetuned_model(image_path: str, checkpoint_path: str) -> str:
    print(f"Loading checkpoint from: {checkpoint_path}...")
    processor = TrOCRProcessor.from_pretrained(checkpoint_path)
    model = VisionEncoderDecoderModel.from_pretrained(checkpoint_path)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    pixel_values = pixel_values.to(device)

    with torch.no_grad():
        generated_ids = model.generate(pixel_values, max_new_tokens=128)

    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text.strip()

def main():
    parser = argparse.ArgumentParser(description="Test fine-tuned model checkpoint on an image.")
    parser.add_argument("--image", default="datasets/sarvam_indic/images", help="Path to test image or folder")
    parser.add_argument("--checkpoint", default="models/checkpoints/trocr-tetrax", help="Path to checkpoint directory")
    args = parser.parse_args()

    img_p = Path(args.image)
    if img_p.is_dir():
        # Pick first available image
        images = list(img_p.glob("*.png")) + list(img_p.glob("*.jpg"))
        if not images:
            raise SystemExit(f"No images found in {img_p}")
        img_p = images[0]

    print(f"Testing image: {img_p}")
    result_text = extract_text_with_finetuned_model(str(img_p), args.checkpoint)
    print("\n" + "=" * 50)
    print("      FINE-TUNED MODEL PREDICTION")
    print("=" * 50)
    print(result_text)
    print("=" * 50)

if __name__ == "__main__":
    main()
