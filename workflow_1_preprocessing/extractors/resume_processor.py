"""
Resume Processor - Main extraction orchestrator.
Coordinates different extractors based on file type and content.
"""

from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List
import re

from loguru import logger
from PIL import Image

from extractors.base import BaseExtractor
from extractors.pdf_extractor import PDFExtractor
from extractors.docx_extractor import DocxExtractor
from extractors.ocr_extractor import NuMarkdownExtractor, FallbackOCRExtractor


class ResumeProcessor:
    """
    Main resume processing class.
    Orchestrates extraction from various document formats.
    """
    
    def __init__(self, use_ocr: bool = True, use_numarkdown: bool = True):
        """
        Initialize the resume processor.
        
        Args:
            use_ocr: Whether to use OCR for scanned documents
            use_numarkdown: Whether to use NuMarkdown model (vs Tesseract fallback)
        """
        self.pdf_extractor = PDFExtractor()
        self.docx_extractor = DocxExtractor()
        
        self.use_ocr = use_ocr
        self.use_numarkdown = use_numarkdown
        
        # Lazy load OCR extractors
        self._ocr_extractor: Optional[BaseExtractor] = None
        self._fallback_ocr: Optional[FallbackOCRExtractor] = None
    
    @property
    def ocr_extractor(self) -> BaseExtractor:
        """Get or initialize OCR extractor."""
        if self._ocr_extractor is None:
            if self.use_numarkdown:
                try:
                    self._ocr_extractor = NuMarkdownExtractor(lazy_load=True)
                    logger.info("Using NuMarkdown for OCR")
                except Exception as e:
                    logger.warning(f"NuMarkdown not available: {e}")
                    self._ocr_extractor = self.fallback_ocr
            else:
                self._ocr_extractor = self.fallback_ocr
        return self._ocr_extractor
    
    @property
    def fallback_ocr(self) -> FallbackOCRExtractor:
        """Get or initialize fallback OCR extractor."""
        if self._fallback_ocr is None:
            self._fallback_ocr = FallbackOCRExtractor()
        return self._fallback_ocr
    
    def process_file(self, file_path: str | Path) -> Dict[str, Any]:
        """
        Process a resume file and extract all content.
        
        Args:
            file_path: Path to the resume file
            
        Returns:
            Dictionary containing extracted data and metadata
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_ext = file_path.suffix.lower()
        logger.info(f"Processing file: {file_path.name} (type: {file_ext})")
        
        result = {
            "filename": file_path.name,
            "file_type": file_ext.lstrip("."),
            "file_size_bytes": file_path.stat().st_size,
            "raw_text": "",
            "markdown_content": "",
            "extracted_data": {},
            "metadata": {}
        }
        
        try:
            # Route to appropriate extractor
            if file_ext == ".pdf":
                result = self._process_pdf(file_path, result)
            elif file_ext in [".docx", ".doc"]:
                result = self._process_docx(file_path, result)
            elif file_ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"]:
                result = self._process_image(file_path, result)
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
            
            # Parse structured data from extracted content
            result["extracted_data"] = self._parse_resume_data(
                result["raw_text"],
                result["markdown_content"]
            )
            
            result["metadata"]["processing_status"] = "completed"
            logger.success(f"Successfully processed: {file_path.name}")
            
        except Exception as e:
            logger.error(f"Error processing {file_path.name}: {e}")
            result["metadata"]["processing_status"] = "failed"
            result["metadata"]["error"] = str(e)
            raise
        
        return result
    
    def _process_pdf(self, file_path: Path, result: Dict[str, Any]) -> Dict[str, Any]:
        """Process PDF file."""
        # First try text extraction
        raw_text, markdown_content, metadata = self.pdf_extractor.extract(file_path)
        
        result["raw_text"] = raw_text
        result["markdown_content"] = markdown_content
        result["metadata"].update(metadata)
        
        # If text extraction failed or returned little content, use OCR
        if metadata.get("requires_ocr") or len(raw_text.strip()) < 100:
            if self.use_ocr:
                logger.info("Using OCR for PDF (text extraction insufficient)")
                result = self._process_pdf_with_ocr(file_path, result)
        
        return result
    
    def _process_pdf_with_ocr(self, file_path: Path, result: Dict[str, Any]) -> Dict[str, Any]:
        """Process PDF using OCR by converting pages to images."""
        try:
            # Convert PDF pages to images
            images = self.pdf_extractor.convert_to_images(file_path)
            
            # Process with NuMarkdown
            if isinstance(self.ocr_extractor, NuMarkdownExtractor):
                markdown_content = self.ocr_extractor.process_images(images)
                raw_text = self.ocr_extractor._markdown_to_plain_text(markdown_content)
            else:
                # Fallback to Tesseract
                text_parts = []
                for img in images:
                    text_parts.append(self.fallback_ocr.process_image(img))
                raw_text = "\n\n".join(text_parts)
                markdown_content = raw_text  # Basic text for fallback
            
            result["raw_text"] = raw_text
            result["markdown_content"] = markdown_content
            result["metadata"]["ocr_used"] = True
            result["metadata"]["ocr_method"] = "numarkdown" if self.use_numarkdown else "tesseract"
            
        except Exception as e:
            logger.error(f"OCR processing failed: {e}")
            result["metadata"]["ocr_error"] = str(e)
        
        return result
    
    def _process_docx(self, file_path: Path, result: Dict[str, Any]) -> Dict[str, Any]:
        """Process DOCX/DOC file."""
        raw_text, markdown_content, metadata = self.docx_extractor.extract(file_path)
        
        result["raw_text"] = raw_text
        result["markdown_content"] = markdown_content
        result["metadata"].update(metadata)
        
        return result
    
    def _process_image(self, file_path: Path, result: Dict[str, Any]) -> Dict[str, Any]:
        """Process image file with OCR."""
        if not self.use_ocr:
            raise ValueError("OCR is disabled but required for image files")
        
        raw_text, markdown_content, metadata = self.ocr_extractor.extract(file_path)
        
        result["raw_text"] = raw_text
        result["markdown_content"] = markdown_content
        result["metadata"].update(metadata)
        
        return result
    
    def _parse_resume_data(self, raw_text: str, markdown_content: str) -> Dict[str, Any]:
        """
        Parse structured data from resume text.
        
        Args:
            raw_text: Plain text content
            markdown_content: Markdown formatted content
            
        Returns:
            Dictionary with parsed resume sections
        """
        data = {
            "contact_info": self._extract_contact_info(raw_text),
            "sections": self._identify_sections(raw_text, markdown_content),
            "skills": self._extract_skills(raw_text),
            "education": self._extract_education(raw_text),
            "experience": self._extract_experience(raw_text),
            "word_count": len(raw_text.split()),
            "char_count": len(raw_text)
        }
        
        return data
    
    def _extract_contact_info(self, text: str) -> Dict[str, Optional[str]]:
        """Extract contact information from resume text."""
        contact = {
            "email": None,
            "phone": None,
            "linkedin": None,
            "github": None,
            "website": None
        }
        
        # Email pattern
        email_match = re.search(
            r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            text
        )
        if email_match:
            contact["email"] = email_match.group()
        
        # Phone pattern (various formats)
        phone_match = re.search(
            r'(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
            text
        )
        if phone_match:
            contact["phone"] = phone_match.group()
        
        # LinkedIn URL
        linkedin_match = re.search(
            r'(?:linkedin\.com/in/|linkedin:?\s*)([a-zA-Z0-9-]+)',
            text,
            re.IGNORECASE
        )
        if linkedin_match:
            contact["linkedin"] = f"linkedin.com/in/{linkedin_match.group(1)}"
        
        # GitHub URL
        github_match = re.search(
            r'(?:github\.com/|github:?\s*)([a-zA-Z0-9-]+)',
            text,
            re.IGNORECASE
        )
        if github_match:
            contact["github"] = f"github.com/{github_match.group(1)}"
        
        return contact
    
    def _identify_sections(self, raw_text: str, markdown_content: str) -> List[str]:
        """Identify major sections in the resume."""
        common_sections = [
            "summary", "objective", "experience", "work experience",
            "education", "skills", "technical skills", "projects",
            "certifications", "achievements", "awards", "publications",
            "languages", "interests", "hobbies", "references"
        ]
        
        found_sections = []
        text_lower = raw_text.lower()
        
        for section in common_sections:
            if section in text_lower:
                found_sections.append(section.title())
        
        return found_sections
    
    def _extract_skills(self, text: str) -> List[str]:
        """Extract skills from resume text."""
        # Common programming languages and technologies
        tech_keywords = [
            "python", "java", "javascript", "typescript", "c++", "c#",
            "ruby", "go", "rust", "php", "swift", "kotlin", "scala",
            "react", "angular", "vue", "node.js", "django", "flask",
            "spring", "docker", "kubernetes", "aws", "azure", "gcp",
            "sql", "mongodb", "postgresql", "mysql", "redis",
            "git", "linux", "agile", "scrum", "machine learning",
            "deep learning", "tensorflow", "pytorch", "pandas", "numpy"
        ]
        
        found_skills = []
        text_lower = text.lower()
        
        for skill in tech_keywords:
            if skill.lower() in text_lower:
                found_skills.append(skill)
        
        return found_skills
    
    def _extract_education(self, text: str) -> List[Dict[str, str]]:
        """Extract education information."""
        education = []
        
        # Common degree patterns
        degree_patterns = [
            r"(bachelor'?s?|b\.?s\.?|b\.?a\.?|b\.?e\.?|b\.?tech)[\s\w]*",
            r"(master'?s?|m\.?s\.?|m\.?a\.?|m\.?e\.?|m\.?tech|mba)[\s\w]*",
            r"(ph\.?d\.?|doctorate)[\s\w]*",
            r"(diploma|certificate)[\s\w]*"
        ]
        
        for pattern in degree_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if match and match not in [e.get("degree") for e in education]:
                    education.append({"degree": match.strip()})
        
        return education
    
    def _extract_experience(self, text: str) -> List[Dict[str, str]]:
        """Extract work experience information."""
        experience = []
        
        # Look for date ranges (common in experience sections)
        date_pattern = r'(\d{4})\s*[-–]\s*(\d{4}|present|current)'
        matches = re.findall(date_pattern, text, re.IGNORECASE)
        
        for match in matches:
            experience.append({
                "start_year": match[0],
                "end_year": match[1].lower()
            })
        
        return experience


def create_processor(use_numarkdown: bool = True) -> ResumeProcessor:
    """
    Factory function to create a resume processor.
    
    Args:
        use_numarkdown: Whether to use NuMarkdown model
        
    Returns:
        Configured ResumeProcessor instance
    """
    return ResumeProcessor(use_ocr=True, use_numarkdown=use_numarkdown)
