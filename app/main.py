import os
import tempfile
import torch
from pathlib import Path
from typing import Union

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from app.ocr_engine import extract_image, SUPPORTED_LANGS, SUPPORTED_LANGUAGES_INFO
from app.pdf_processor import extract_pdf
from app.schemas import (
    HealthResponse,
    SupportedLanguagesResponse,
    LanguageInfo,
    OCRImageResponse,
    OCRPDFResponse,
)

app = FastAPI(
    title="Tetrax OCR API",
    description="High-performance document OCR service powered by PaddleOCR with English and Indian language support.",
    version="1.0.0",
)

# Enable CORS for dashboard integration (e.g. Next.js)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_BYTES = int(os.getenv("TETRAX_MAX_UPLOAD_BYTES", str(15 * 1024 * 1024)))
MAX_PDF_PAGES = int(os.getenv("TETRAX_MAX_PDF_PAGES", "50"))
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".pdf"}

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    gpu_available = False
    try:
        gpu_available = torch.cuda.is_available()
    except Exception:
        pass

    return HealthResponse(
        status="ok",
        service="tetrax-ocr",
        version="1.0.0",
        engine="PaddleOCR (Option A)",
        gpu_available=gpu_available,
    )

@app.get("/v1/languages", response_model=SupportedLanguagesResponse, tags=["OCR"])
def languages():
    lang_list = [
        LanguageInfo(
            code=code,
            name=info["name"],
            script=info["script"],
            sample_characters=info["sample"],
        )
        for code, info in sorted(SUPPORTED_LANGUAGES_INFO.items())
    ]
    return SupportedLanguagesResponse(
        languages=lang_list,
        default="en",
        total_supported=len(lang_list),
        note="Pretrained models for each language script are automatically fetched by PaddleOCR on initial request.",
    )

@app.post(
    "/v1/ocr",
    response_model=Union[OCRImageResponse, OCRPDFResponse],
    tags=["OCR"],
    summary="Extract text from document image or PDF",
)
async def ocr_document(
    file: UploadFile = File(..., description="Document image or PDF file"),
    lang: str = Form("en", description="Target language code (e.g., 'en', 'hi', 'mr', 'ta', 'te')"),
    preprocess: bool = Form(True, description="Apply deskewing and CLAHE contrast enhancement"),
):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{suffix}'. Supported: {sorted(ALLOWED_EXTENSIONS)}",
        )
    if lang not in SUPPORTED_LANGS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported language code '{lang}'. Choose from: {sorted(SUPPORTED_LANGS)}",
        )

    content = await file.read(MAX_BYTES + 1)
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )
    if len(content) > MAX_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum allowed size of {MAX_BYTES // (1024 * 1024)} MB.",
        )

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
            temp.write(content)
            temp_path = temp.name

        if suffix == ".pdf":
            result = extract_pdf(
                temp_path,
                lang=lang,
                max_pages=MAX_PDF_PAGES,
                apply_preprocessing=preprocess,
            )
            return OCRPDFResponse(
                success=True,
                filename=file.filename or "document.pdf",
                **result,
            )

        # Single page image
        result = extract_image(
            temp_path,
            lang=lang,
            apply_preprocessing=preprocess,
        )
        return OCRImageResponse(
            success=True,
            filename=file.filename or "document.png",
            language=result["language"],
            text=result["text"],
            lines=result["lines"],
            line_count=result["line_count"],
            preprocessing_applied=result["preprocessing_applied"],
        )

    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OCR execution failed: {str(exc)}",
        )
    finally:
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)
