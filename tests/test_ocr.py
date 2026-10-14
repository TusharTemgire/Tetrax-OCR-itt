import os
import unittest
import numpy as np
import cv2
from pathlib import Path
from fastapi.testclient import TestClient

from app.main import app
from app.preprocessing import deskew, enhance_contrast, denoise, preprocess_image
from app.schemas import (
    OCRLine,
    OCRImageResponse,
    HealthResponse,
    SupportedLanguagesResponse,
)
from training.evaluate import evaluate_predictions, normalize
from training.prepare_dataset import normalize_text, generate_char_dictionary

class TestTetraxOCR(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["service"], "tetrax-ocr")
        self.assertIn("PaddleOCR", data["engine"])

    def test_languages_endpoint(self):
        response = self.client.get("/v1/languages")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        codes = [item["code"] for item in data["languages"]]
        self.assertIn("en", codes)
        self.assertIn("hi", codes)
        self.assertIn("ta", codes)
        self.assertIn("te", codes)
        self.assertIn("mr", codes)
        self.assertGreaterEqual(data["total_supported"], 10)

    def test_schemas_validation(self):
        line = OCRLine(
            text="INVOICE NUMBER: INV-1045",
            confidence=0.985,
            bounding_box=[[10.0, 10.0], [200.0, 10.0], [200.0, 30.0], [10.0, 30.0]]
        )
        self.assertEqual(line.confidence, 0.985)
        self.assertEqual(len(line.bounding_box), 4)

        resp = OCRImageResponse(
            success=True,
            filename="sample.png",
            language="en",
            text="INVOICE NUMBER: INV-1045",
            lines=[line],
            line_count=1,
            preprocessing_applied=True
        )
        self.assertEqual(resp.line_count, 1)

    def test_preprocessing_pipeline(self):
        # Create a synthetic test image with text-like dark region
        test_img = np.ones((100, 300, 3), dtype=np.uint8) * 255
        cv2.rectangle(test_img, (20, 40), (280, 60), (0, 0, 0), -1)

        # Test contrast enhancement
        enhanced = enhance_contrast(test_img)
        self.assertEqual(enhanced.shape, test_img.shape)

        # Test denoising
        denoised = denoise(test_img)
        self.assertEqual(denoised.shape, test_img.shape)

        # Test full preprocessing pipeline
        processed, meta = preprocess_image(test_img, apply_deskew=True, apply_contrast=True)
        self.assertEqual(processed.shape, test_img.shape)
        self.assertTrue(meta["contrast_enhanced"])

    def test_unicode_normalization(self):
        # Decomposed Devanagari vs composed NFC
        decomposed = "क" + "\u093e"  # ka + aa matra
        composed = "का"
        self.assertEqual(normalize(decomposed), normalize(composed))
        self.assertEqual(normalize_text(decomposed), normalize_text(composed))

    def test_evaluate_metrics(self):
        # Test evaluation with exact matches and small differences
        test_csv_path = Path("datasets/test_eval_scratch.csv")
        try:
            with open(test_csv_path, "w", encoding="utf-8") as f:
                f.write("image_path,reference,prediction\n")
                f.write("img1.png,Invoice Number: INV-1045,Invoice Number: INV-1045\n")
                f.write("img2.png,Total Amount: 2950.00,Total Amount: 2950.00\n")

            metrics = evaluate_predictions(str(test_csv_path))
            self.assertEqual(metrics["exact_match_accuracy"], 100.0)
            self.assertEqual(metrics["cer"], 0.0)
            self.assertEqual(metrics["wer"], 0.0)
        finally:
            test_csv_path.unlink(missing_ok=True)

    def test_unsupported_file_extension(self):
        files = {"file": ("test.exe", b"fake binary executable content", "application/octet-stream")}
        response = self.client.post("/v1/ocr", files=files, data={"lang": "en"})
        self.assertEqual(response.status_code, 415)

    def test_unsupported_language_code(self):
        files = {"file": ("test.png", b"\x89PNG\r\n\x1a\nfakeimage", "image/png")}
        response = self.client.post("/v1/ocr", files=files, data={"lang": "xyz_invalid"})
        self.assertEqual(response.status_code, 400)

if __name__ == "__main__":
    unittest.main()
