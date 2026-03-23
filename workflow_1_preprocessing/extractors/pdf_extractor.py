"""
PDF document extractor.
Handles PDF text extraction and conversion to images for OCR.
"""

from pathlib import Path
from typing import Tuple, Dict, Any, List
import io

from loguru import logger

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

try:
    from PyPDF2 import PdfReader
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

from PIL import Image

from extractors.base import BaseExtractor


class PDFExtractor(BaseExtractor):
    """Extractor for PDF documents."""
    
    SUPPORTED_FORMATS = [".pdf"]
    
    def supports_format(self, file_extension: str) -> bool:
        """Check if PDF format is supported."""
        return file_extension.lower() in self.SUPPORTED_FORMATS
    
    def extract(self, file_path: Path) -> Tuple[str, str, Dict[str, Any]]:
        """
        Extract text from PDF file.
        
        Tries multiple methods in order of preference:
        1. pdfplumber (best for structured PDFs)
        2. PyPDF2 (fallback for simple PDFs)
        3. Returns empty if both fail (will use OCR later)
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Tuple of (raw_text, markdown_content, metadata)
        """
        metadata = self.get_file_info(file_path)
        metadata["extraction_method"] = None
        metadata["page_count"] = 0
        
        raw_text = ""
        
        # Try pdfplumber first (better for complex layouts)
        if PDFPLUMBER_AVAILABLE:
            try:
                raw_text, page_count = self._extract_with_pdfplumber(file_path)
                if raw_text.strip():
                    metadata["extraction_method"] = "pdfplumber"
                    metadata["page_count"] = page_count
                    logger.info(f"Extracted {len(raw_text)} chars using pdfplumber")
            except Exception as e:
                logger.warning(f"pdfplumber extraction failed: {e}")
        
        # Fallback to PyPDF2
        if not raw_text.strip() and PYPDF2_AVAILABLE:
            try:
                raw_text, page_count = self._extract_with_pypdf2(file_path)
                if raw_text.strip():
                    metadata["extraction_method"] = "pypdf2"
                    metadata["page_count"] = page_count
                    logger.info(f"Extracted {len(raw_text)} chars using PyPDF2")
            except Exception as e:
                logger.warning(f"PyPDF2 extraction failed: {e}")
        
        # If text extraction failed, mark for OCR
        if not raw_text.strip():
            metadata["requires_ocr"] = True
            logger.info("Text extraction failed, OCR will be required")
        else:
            metadata["requires_ocr"] = False
        
        # Basic markdown conversion (will be enhanced by NuMarkdown)
        markdown_content = self._text_to_basic_markdown(raw_text)
        
        return raw_text, markdown_content, metadata
    
    def _extract_with_pdfplumber(self, file_path: Path) -> Tuple[str, int]:
        """Extract text using pdfplumber."""
        text_parts = []
        page_count = 0
        
        with pdfplumber.open(file_path) as pdf:
            page_count = len(pdf.pages)
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text_parts.append(page_text)
        
        return "\n\n".join(text_parts), page_count
    
    def _extract_with_pypdf2(self, file_path: Path) -> Tuple[str, int]:
        """Extract text using PyPDF2."""
        text_parts = []
        
        reader = PdfReader(file_path)
        page_count = len(reader.pages)
        
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text)
        
        return "\n\n".join(text_parts), page_count
    
    def convert_to_images(self, file_path: Path, dpi: int = 200) -> List[Image.Image]:
        """
        Convert PDF pages to images for OCR processing.
        
        Args:
            file_path: Path to PDF file
            dpi: Resolution for conversion
            
        Returns:
            List of PIL Image objects
        """
        if not PDF2IMAGE_AVAILABLE:
            raise ImportError("pdf2image is required for PDF to image conversion")
        
        images = convert_from_path(file_path, dpi=dpi)
        logger.info(f"Converted PDF to {len(images)} images")
        return images
    
    def _text_to_basic_markdown(self, text: str) -> str:
        """
        Convert plain text to basic markdown format.
        This is a simple conversion - NuMarkdown will provide better structure.
        """
        if not text.strip():
            return ""
        
        lines = text.split("\n")
        markdown_lines = []
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                markdown_lines.append("")
                continue
            
            # Simple heuristics for headings (all caps, short lines)
            if stripped.isupper() and len(stripped) < 50:
                markdown_lines.append(f"## {stripped.title()}")
            else:
                markdown_lines.append(stripped)
        
        return "\n".join(markdown_lines)
