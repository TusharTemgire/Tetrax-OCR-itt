import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional

def deskew(image: np.ndarray, max_angle: float = 30.0) -> Tuple[np.ndarray, float]:
    """
    Detects skew angle in document images using text line orientation
    and rotates the image to straighten text lines.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Invert and threshold to get text pixels
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 8
    )

    # Find coordinates of all foreground pixels
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 50:
        return image, 0.0

    # Fit minimum bounding box to foreground points
    angle = cv2.minAreaRect(coords)[-1]

    # Convert OpenCV angle convention to skew angle
    if angle < -45:
        angle = -(90 + angle)
    elif angle > 45:
        angle = 90 - angle
    else:
        angle = -angle

    # Avoid rotating if estimated angle is too extreme or negligible
    if abs(angle) < 0.3 or abs(angle) > max_angle:
        return image, 0.0

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated, angle

def enhance_contrast(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization)
    to improve OCR accuracy on low-contrast or faded Indian documents.
    """
    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(image)

def denoise(image: np.ndarray) -> np.ndarray:
    """
    Applies bilateral filtering to smooth paper background noise
    while preserving sharp character edges and Indian script matras.
    """
    if len(image.shape) == 3:
        return cv2.bilateralFilter(image, d=7, sigmaColor=50, sigmaSpace=50)
    return cv2.bilateralFilter(image, d=7, sigmaColor=50, sigmaSpace=50)

def resize_if_needed(image: np.ndarray, max_dim: int = 2400) -> np.ndarray:
    """
    Downsamples very large scans or camera shots to avoid memory limits
    while keeping enough resolution for OCR.
    """
    h, w = image.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image

def preprocess_image(
    image_input,
    apply_deskew: bool = True,
    apply_contrast: bool = True,
    apply_denoise: bool = False,
    max_dimension: int = 2400
) -> Tuple[np.ndarray, dict]:
    """
    Complete preprocessing pipeline for document OCR.
    Accepts file path (str), PIL.Image, or numpy ndarray.
    Returns: (preprocessed_bgr_image, metadata_dict)
    """
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
        if img is None:
            raise ValueError(f"Unable to read image at {image_input}")
    elif isinstance(image_input, Image.Image):
        img = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
    elif isinstance(image_input, np.ndarray):
        img = image_input.copy()
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    meta = {"skew_angle": 0.0, "contrast_enhanced": False, "denoised": False, "resized": False}

    img = resize_if_needed(img, max_dim=max_dimension)
    if img.shape[:2] != (img.shape[0], img.shape[1]):
        meta["resized"] = True

    if apply_deskew:
        img, angle = deskew(img)
        meta["skew_angle"] = round(angle, 2)

    if apply_contrast:
        img = enhance_contrast(img)
        meta["contrast_enhanced"] = True

    if apply_denoise:
        img = denoise(img)
        meta["denoised"] = True

    return img, meta
