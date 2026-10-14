from typing import List, Optional, Any
from pydantic import BaseModel, Field

class BoundingBox(BaseModel):
    """Four coordinate points representing the bounding polygon [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]."""
    points: List[List[float]] = Field(
        ...,
        description="4 coordinate points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]"
    )

class OCRLine(BaseModel):
    text: str = Field(..., description="Recognized text string")
    confidence: Optional[float] = Field(None, description="Confidence score between 0.0 and 1.0")
    bounding_box: Optional[List[List[float]]] = Field(
        None,
        description="Polygon coordinates [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]"
    )

class OCRImageResponse(BaseModel):
    success: bool = True
    filename: str
    language: str
    text: str = Field(..., description="Full concatenated extracted text")
    lines: List[OCRLine] = Field(default_factory=list, description="List of recognized text lines")
    line_count: int
    preprocessing_applied: bool = False

class OCRPage(BaseModel):
    page: int = Field(..., description="1-indexed page number")
    text: str
    lines: List[OCRLine]
    line_count: int

class OCRPDFResponse(BaseModel):
    success: bool = True
    filename: str
    language: str
    page_count: int
    text: str = Field(..., description="Full concatenated text across all pages")
    pages: List[OCRPage]
    preprocessing_applied: bool = False

class LanguageInfo(BaseModel):
    code: str
    name: str
    script: str
    sample_characters: str

class SupportedLanguagesResponse(BaseModel):
    languages: List[LanguageInfo]
    default: str = "en"
    total_supported: int
    note: str

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    engine: str
    gpu_available: bool
