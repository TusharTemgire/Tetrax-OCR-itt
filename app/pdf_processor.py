import logging
from pathlib import Path
from typing import List, Dict, Any
import cv2
import numpy as np
from PIL import Image

from app.ocr_engine import extract_image

logger = logging.getLogger("tetrax.pdf")

def _render_pages_pymupdf(pdf_path: str, dpi: int) -> List[np.ndarray]:
    import fitz  # PyMuPDF
    doc = fitz.open(pdf_path)
    pages = []
    try:
        for page_index in range(doc.page_count):
            page = doc.load_page(page_index)
            pix = page.get_pixmap(dpi=dpi, alpha=False)
            img_bytes = pix.tobytes("png")
            nparr = np.frombuffer(img_bytes, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            pages.append(img_bgr)
    finally:
        doc.close()
    return pages

def _render_pages_pypdfium2(pdf_path: str, dpi: int) -> List[np.ndarray]:
    import pypdfium2 as pdfium
    doc = pdfium.PdfDocument(pdf_path)
    pages = []
    scale = dpi / 72.0
    for page in doc:
        rendered = page.render(scale=scale)
        pil_img = rendered.to_pil()
        img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        pages.append(img_bgr)
    return pages

def render_pdf_to_images(pdf_path: str, dpi: int = 200) -> List[np.ndarray]:
    """
    Renders PDF pages to BGR numpy arrays.
    Supports PyMuPDF (fitz) with automatic fallback to pypdfium2.
    """
    try:
        return _render_pages_pymupdf(pdf_path, dpi)
    except ImportError:
        pass

    try:
        return _render_pages_pypdfium2(pdf_path, dpi)
    except ImportError:
        pass

    raise RuntimeError(
        "Neither 'pymupdf' (fitz) nor 'pypdfium2' is installed. "
        "Please install either package to process PDF files."
    )

def extract_pdf(
    pdf_path: str,
    lang: str = "en",
    dpi: int = 200,
    max_pages: int = 50,
    apply_preprocessing: bool = True
) -> dict:
    """
    Renders multi-page PDF documents and extracts text per page using PaddleOCR.
    DPI defaults to 200 (optimal balance between detail for Indian script diacritics and OCR speed).
    """
    page_images = render_pdf_to_images(pdf_path, dpi=dpi)
    total_pages = len(page_images)

    if total_pages == 0:
        raise ValueError("PDF contains no pages")
    if total_pages > max_pages:
        raise ValueError(f"PDF exceeds maximum page limit ({total_pages} > {max_pages})")

    pages = []
    for page_index, img_bgr in enumerate(page_images):
        page_result = extract_image(
            img_bgr,
            lang=lang,
            apply_preprocessing=apply_preprocessing
        )
        pages.append({
            "page": page_index + 1,
            "text": page_result["text"],
            "lines": page_result["lines"],
            "line_count": len(page_result["lines"]),
        })

    full_document_text = "\n\n".join(p["text"] for p in pages if p["text"])
    return {
        "text": full_document_text,
        "pages": pages,
        "page_count": len(pages),
        "language": lang,
        "preprocessing_applied": apply_preprocessing,
    }
