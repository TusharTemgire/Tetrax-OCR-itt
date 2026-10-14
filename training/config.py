"""
Configuration module for Tetrax OCR (PaddleOCR Option A) fine-tuning and evaluation.
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset paths
DATASETS_DIR = BASE_DIR / "datasets"
IMAGES_DIR = DATASETS_DIR / "images"
ANNOTATIONS_DIR = DATASETS_DIR / "annotations"
PADDLE_REC_DIR = DATASETS_DIR / "paddle_rec"
OUTPUT_DIR = BASE_DIR / "models" / "checkpoints"

# Standard character sets for Indian languages & English
CHAR_DICTS = {
    "en": "ppocr_keys_v1.txt",
    "hi": "devanagari_dict.txt",
    "mr": "devanagari_dict.txt",
    "ta": "tamil_dict.txt",
    "te": "telugu_dict.txt",
    "bn": "bengali_dict.txt",
}

# Default PaddleOCR recognition training parameters (PP-OCRv4 / PP-OCRv5 rec)
TRAINING_DEFAULTS = {
    "algorithm": "SVTR_LCNet",
    "image_shape": [3, 48, 320],
    "batch_size_per_card": 64,
    "epochs": 100,
    "learning_rate": 0.001,
    "optimizer": "Adam",
    "use_gpu": True,
}

def get_rec_config(language: str = "en") -> dict:
    return {
        "Global": {
            "use_gpu": TRAINING_DEFAULTS["use_gpu"],
            "epoch_num": TRAINING_DEFAULTS["epochs"],
            "log_smooth_window": 20,
            "print_batch_step": 10,
            "save_model_dir": str(OUTPUT_DIR / f"paddle_{language}_rec"),
            "save_epoch_step": 5,
            "eval_batch_step": [0, 100],
            "cal_metric_during_train": True,
            "character_dict_path": str(DATASETS_DIR / "dicts" / CHAR_DICTS.get(language, "ppocr_keys_v1.txt")),
            "use_space_char": True,
        },
        "Optimizer": {
            "name": TRAINING_DEFAULTS["optimizer"],
            "lr": {
                "name": "Cosine",
                "learning_rate": TRAINING_DEFAULTS["learning_rate"],
            },
        },
        "Architecture": {
            "model_type": "rec",
            "algorithm": TRAINING_DEFAULTS["algorithm"],
            "Transform": None,
            "Backbone": {
                "name": "PPLCNetV3",
                "scale": 0.95,
            },
            "Head": {
                "name": "MultiHead",
                "head_list": [
                    {"CTCHead": {"Neck": {"name": "svtr", "dims": 120, "depth": 2, "hidden_dims": 120, "use_guide": True}}},
                    {"SARHead": {"enc_dim": 512, "max_text_len": 40}}
                ]
            }
        },
        "Train": {
            "dataset": {
                "name": "SimpleDataSet",
                "data_dir": str(DATASETS_DIR),
                "label_file_list": [str(PADDLE_REC_DIR / "train.txt")],
            },
            "loader": {
                "shuffle": True,
                "batch_size_per_card": TRAINING_DEFAULTS["batch_size_per_card"],
                "drop_last": True,
                "num_workers": 4,
            }
        },
        "Eval": {
            "dataset": {
                "name": "SimpleDataSet",
                "data_dir": str(DATASETS_DIR),
                "label_file_list": [str(PADDLE_REC_DIR / "val.txt")],
            },
            "loader": {
                "shuffle": False,
                "drop_last": False,
                "batch_size_per_card": 64,
                "num_workers": 4,
            }
        }
    }
