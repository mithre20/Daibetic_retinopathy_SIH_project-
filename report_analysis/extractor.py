"""
Medical Report Ingestion and Text Extraction Module
AI DR Screen — Patient Health Report Analysis

Extracts text from medical laboratory reports in PDF, JPG, JPEG, and PNG formats.
Supports direct text extraction from digital PDFs via pypdf and OCR extraction
for scanned documents and image files via EasyOCR (with graceful fallbacks).
"""

import os
import io
import re
import tempfile
from typing import Dict, Any, Optional, Union, List
from PIL import Image
import numpy as np

SUPPORTED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png"}

# Global lazy OCR reader
_easyocr_reader = None


def get_ocr_reader():
    """
    Lazily initialize the EasyOCR reader with Windows UTF-8 console safety.
    """
    global _easyocr_reader
    if _easyocr_reader is None:
        try:
            import os
            os.environ["PYTHONIOENCODING"] = "utf-8"
            import easyocr.utils
            easyocr.utils.print = lambda *args, **kwargs: None
            import easyocr
            _easyocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        except Exception as e:
            print(f"[report_analysis] Warning: EasyOCR failed to initialize: {e}")
            _easyocr_reader = False
    return _easyocr_reader if _easyocr_reader is not False else None


class ReportExtractor:
    """
    Multi-format extractor for laboratory reports.
    """

    def __init__(self, max_file_size_mb: float = 25.0):
        self.max_file_size = int(max_file_size_mb * 1024 * 1024)

    def extract_from_file(
        self,
        file_input: Union[str, bytes, io.BytesIO],
        filename: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract text lines and metadata from a file path, raw bytes, or BytesIO object.
        """
        raw_bytes = None
        orig_filename = filename or "report_document"

        if isinstance(file_input, str):
            if not os.path.isfile(file_input):
                return {
                    "success": False,
                    "error": f"Report file '{file_input}' not found on disk.",
                    "extracted_text": "",
                    "lines": [],
                    "extraction_method": "none",
                    "confidence": 0.0
                }
            orig_filename = filename or os.path.basename(file_input)
            with open(file_input, "rb") as f:
                raw_bytes = f.read()
        elif isinstance(file_input, io.BytesIO):
            raw_bytes = file_input.getvalue()
        elif isinstance(file_input, bytes):
            raw_bytes = file_input
        else:
            return {
                "success": False,
                "error": "Unsupported input type. Expected file path, raw bytes, or BytesIO.",
                "extracted_text": "",
                "lines": [],
                "extraction_method": "none",
                "confidence": 0.0
            }

        # Validate file size
        if len(raw_bytes) > self.max_file_size:
            return {
                "success": False,
                "error": f"File size ({len(raw_bytes)/(1024*1024):.1f}MB) exceeds limit of {self.max_file_size/(1024*1024):.0f}MB.",
                "extracted_text": "",
                "lines": [],
                "extraction_method": "none",
                "confidence": 0.0
            }

        # Detect format
        ext = os.path.splitext(orig_filename.lower())[1]
        if not ext:
            # Try to sniff format from magic bytes
            if raw_bytes.startswith(b"%PDF"):
                ext = ".pdf"
            elif raw_bytes.startswith(b"\xff\xd8\xff"):
                ext = ".jpg"
            elif raw_bytes.startswith(b"\x89PNG"):
                ext = ".png"
            else:
                ext = ".unknown"

        if ext not in SUPPORTED_EXTENSIONS:
            return {
                "success": False,
                "error": f"Unsupported file type '{ext}'. Supported formats: PDF, JPG, JPEG, PNG.",
                "extracted_text": "",
                "lines": [],
                "extraction_method": "none",
                "confidence": 0.0
            }

        if ext == ".pdf":
            return self._extract_pdf(raw_bytes, orig_filename)
        else:
            return self._extract_image(raw_bytes, orig_filename)

    def _extract_pdf(self, pdf_bytes: bytes, filename: str) -> Dict[str, Any]:
        """
        Extract text from PDF using pypdf, falling back to OCR if digital text is empty.
        """
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
            num_pages = len(reader.pages)
            extracted_pages = []

            for p_idx, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    extracted_pages.append(text.strip())

            combined_text = "\n".join(extracted_pages)

            if len(combined_text.strip()) > 10:
                # Digital text was successfully extracted
                lines = [l.strip() for l in combined_text.splitlines() if l.strip()]
                return {
                    "success": True,
                    "filename": filename,
                    "file_type": "PDF",
                    "num_pages": num_pages,
                    "extracted_text": combined_text,
                    "lines": lines,
                    "extraction_method": "pypdf_digital_text",
                    "confidence": 0.98,
                    "page_count": num_pages
                }

            # If digital text was empty, this is likely a scanned PDF
            # Attempt to extract embedded images for OCR
            for page in reader.pages:
                for img_obj in page.images:
                    return self._extract_image(img_obj.data, f"{filename}_scanned_page.png")

            return {
                "success": False,
                "filename": filename,
                "file_type": "PDF",
                "error": "The uploaded PDF document contains no selectable digital text or extractable page scans.",
                "extracted_text": "",
                "lines": [],
                "extraction_method": "pypdf_empty",
                "confidence": 0.0
            }
        except Exception as e:
            return {
                "success": False,
                "filename": filename,
                "file_type": "PDF",
                "error": f"Failed to parse PDF document: {str(e)}",
                "extracted_text": "",
                "lines": [],
                "extraction_method": "pypdf_error",
                "confidence": 0.0
            }

    def _extract_image(self, image_bytes: bytes, filename: str) -> Dict[str, Any]:
        """
        Extract text from an image (JPG/PNG) using EasyOCR with graceful fallbacks.
        """
        try:
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            np_img = np.array(pil_img)
        except Exception as e:
            return {
                "success": False,
                "filename": filename,
                "file_type": "IMAGE",
                "error": f"Invalid image file: {str(e)}",
                "extracted_text": "",
                "lines": [],
                "extraction_method": "pil_error",
                "confidence": 0.0
            }

        reader = get_ocr_reader()
        if reader is not None:
            try:
                ocr_results = reader.readtext(np_img)
                lines = []
                confidences = []
                for item in ocr_results:
                    # item is: (bbox, text, confidence)
                    if len(item) >= 3:
                        text = str(item[1]).strip()
                        conf = float(item[2])
                        if text:
                            lines.append(text)
                            confidences.append(conf)

                avg_conf = float(np.mean(confidences)) if confidences else 0.5
                combined_text = "\n".join(lines)

                return {
                    "success": True,
                    "filename": filename,
                    "file_type": "IMAGE",
                    "extracted_text": combined_text,
                    "lines": lines,
                    "extraction_method": "EasyOCR",
                    "confidence": round(avg_conf, 3),
                    "dimensions": list(pil_img.size)
                }
            except Exception as e:
                print(f"[report_analysis] EasyOCR error: {e}")

        # Fallback: if EasyOCR reader is unavailable or encounters error, return structured empty extraction
        return {
            "success": False,
            "filename": filename,
            "file_type": "IMAGE",
            "error": "OCR engine could not extract readable text from the image. Please verify image clarity or enter values manually.",
            "extracted_text": "",
            "lines": [],
            "extraction_method": "ocr_unavailable",
            "confidence": 0.0
        }
