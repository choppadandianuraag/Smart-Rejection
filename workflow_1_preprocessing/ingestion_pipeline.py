"""
Section-Aware Resume Ingestion Pipeline (V2)
Refactored pipeline that segments → embeds → stores per-section.
"""

import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from uuid import UUID
from decimal import Decimal

from loguru import logger

# Import existing extractors (reuse PDF/DOCX/OCR logic)
from extractors.resume_processor import ResumeProcessor, create_processor

# Import new V2 components
from segmentation import segment_resume, Section
from embeddings.bert_embedder_v2 import get_section_embedder
from database.supabase_client_v2 import get_db_client
from database.models_v2 import (
    ApplicantProfileCreate,
    ApplicantEmbeddingCreate
)


class SectionAwareResumeProcessor:
    """
    Enhanced resume processor that segments, embeds, and stores per-section.
    
    Pipeline:
        1. Extract raw text (reuse existing extractors)
        2. Segment into sections
        3. Generate BERT embeddings per section
        4. Store in dual-table structure (profiles + embeddings)
    """
    
    def __init__(self, use_ocr: bool = True, use_numarkdown: bool = False):
        """
        Initialize processor.
        
        Args:
            use_ocr: Enable OCR for images/scanned PDFs
            use_numarkdown: Use NuMarkdown model (expensive) or Tesseract
        """
        # Reuse existing extraction logic
        self.text_extractor = create_processor(use_numarkdown=use_numarkdown)
        
        # New V2 components
        self.embedder = get_section_embedder()
        self.db = get_db_client()
        
        logger.info("Section-aware resume processor initialized")
    
    def process_resume(
        self,
        file_path: Path,
        name: Optional[str] = None,
        email: Optional[str] = None,
        contact_number: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a resume file end-to-end.
        
        Args:
            file_path: Path to resume file
            name: Applicant name (extracted if not provided)
            email: Applicant email (extracted if not provided)
            contact_number: Phone number (extracted if not provided)
            
        Returns:
            Dict with applicant_id and processing status
        """
        file_path = Path(file_path)
        logger.info(f"Processing resume: {file_path.name}")
        
        try:
            # STEP 1: Extract raw text (reuse existing extractors)
            extraction_result = self.text_extractor.process_file(file_path)
            raw_text = extraction_result["raw_text"]
            
            logger.info(f"Extracted {len(raw_text)} characters")
            
            # STEP 2: Segment into sections
            sections, metadata = segment_resume(raw_text)
            
            if not sections:
                raise ValueError("No sections detected in resume")
            
            logger.info(
                f"Segmented into {len(sections)} sections "
                f"(confidence: {metadata['avg_confidence']:.2f})"
            )
            
            # Extract contact info if not provided
            contact_section = next(
                (s for s in sections if s.section_type == 'contact_info'),
                None
            )
            
            if not email and contact_section:
                email = self._extract_email(contact_section.text)
            if not name and contact_section:
                name = self._extract_name(contact_section.text, file_path.stem)
            if not contact_number and contact_section:
                contact_number = self._extract_phone(contact_section.text)
            
            if not email:
                raise ValueError("Email address is required but could not be extracted")
            if not name:
                raise ValueError("Name is required but could not be extracted")
            
            # STEP 3: Organize sections and generate ONE combined embedding
            logger.info("Generating combined BERT embedding...")

            # Organize sections by type with fallback mapping
            section_texts = {}
            combined_text_parts = []

            # Map of canonical names to possible variations
            section_mapping = {
                'skills': 'skills',
                'education': 'education',
                'work_experience': 'work_experience',
                'projects': 'projects',
                'certifications': 'certifications',
                'summary': 'summary',
                'other': 'other'
            }

            for section in sections:
                section_type = section.section_type
                section_texts[section_type] = section.text

                # Exclude contact_info from the embedding
                if section_type != 'contact_info':
                    combined_text_parts.append(f"{section_type.upper()}:\n{section.text}")

            # Debug: Log what sections were detected
            logger.debug(f"Detected sections: {list(section_texts.keys())}")

            # Combine all sections (except contact_info) into one text
            combined_text = "\n\n".join(combined_text_parts)

            # Generate ONE embedding for the combined text
            combined_embedding = self.embedder.embed_text(combined_text)

            logger.success(f"Generated combined embedding (768 dims)")

            # STEP 4: Store in database (flat structure)
            applicant_id = self._store_in_database(
                name=name,
                email=email,
                contact_number=contact_number,
                file_path=file_path,
                raw_text=raw_text,
                section_texts=section_texts,
                combined_embedding=combined_embedding,
                metadata=metadata
            )
            
            return {
                "status": "success",
                "applicant_id": str(applicant_id),
                "name": name,
                "email": email,
                "section_count": len(sections),
                "avg_confidence": metadata["avg_confidence"],
                "needs_review": metadata["needs_manual_review"],
                "review_reason": metadata.get("review_reason")
            }
            
        except Exception as e:
            logger.error(f"Resume processing failed: {e}")
            raise
    
    def _store_in_database(
        self,
        name: str,
        email: str,
        contact_number: Optional[str],
        file_path: Path,
        raw_text: str,
        section_texts: Dict[str, str],
        combined_embedding: Any,
        metadata: Dict
    ) -> UUID:
        """
        Store applicant profile and embedding in database (flat structure).
        One row in applicant_profiles + one row in applicant_embeddings.
        """
        logger.info("Storing in database...")

        # Check if applicant already exists (resume update scenario)
        existing_applicant = self.db.get_applicant_by_email(email)

        if existing_applicant:
            logger.warning(f"Applicant {email} already exists, updating...")
            applicant_id = existing_applicant.applicant_id

            # Delete old embedding (will be replaced)
            self.db.delete_applicant_embedding(applicant_id)
        else:
            # Create new profile
            profile_data = ApplicantProfileCreate(
                name=name,
                email=email,
                contact_number=contact_number,
                original_filename=file_path.name,
                file_type=file_path.suffix.lstrip('.'),
                file_size_bytes=file_path.stat().st_size,
                raw_text=raw_text,
                segmentation_confidence=Decimal(str(metadata["avg_confidence"])),
                needs_manual_review=metadata["needs_manual_review"],
                review_reason=metadata.get("review_reason")
            )

            profile = self.db.create_applicant_profile(profile_data)

            if not profile:
                raise Exception("Failed to create applicant profile")

            applicant_id = profile.applicant_id

        # Create embedding (flat structure - one row per applicant)
        embedding_data = ApplicantEmbeddingCreate(
            applicant_id=applicant_id,
            skills_text=section_texts.get('skills'),
            education_text=section_texts.get('education'),
            work_experience_text=section_texts.get('work_experience'),
            projects_text=section_texts.get('projects'),
            certifications_text=section_texts.get('certifications'),
            summary_text=section_texts.get('summary'),
            resume_embedding=combined_embedding.tolist() if hasattr(combined_embedding, 'tolist') else combined_embedding
        )

        self.db.create_applicant_embedding(embedding_data)

        logger.success(
            f"Stored profile {applicant_id} with combined embedding"
        )

        return applicant_id
    
    def _extract_email(self, text: str) -> Optional[str]:
        """Extract email from text."""
        match = re.search(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            text
        )
        return match.group() if match else None
    
    def _extract_name(self, text: str, filename_fallback: str) -> Optional[str]:
        """Extract name from contact section (first line often)."""
        lines = text.split('\n')
        for line in lines[:5]:  # Check first 5 lines
            line = line.strip()
            # Name heuristic: 2-4 words, starts with capital, no @ or numbers
            if (
                line and
                2 <= len(line.split()) <= 4 and
                line[0].isupper() and
                '@' not in line and
                not any(char.isdigit() for char in line[:10])
            ):
                return line
        
        # Fallback to filename
        return filename_fallback.replace('_', ' ').replace('-', ' ').title()
    
    def _extract_phone(self, text: str) -> Optional[str]:
        """
        Extract phone number from text with improved patterns.
        Looks for 10-digit numbers or international format.
        """
        # Try Indian format first: +91-XXXXXXXXXX or +91 XXXXXXXXXX
        match = re.search(
            r'[\+]?91[-\s]?\d{10}',
            text
        )
        if match:
            return match.group().strip()

        # Try US/International format: +1-XXX-XXX-XXXX or (XXX) XXX-XXXX
        match = re.search(
            r'[\+]?1?[-\s]?[(]?\d{3}[)]?[-\s\.]?\d{3}[-\s\.]?\d{4}',
            text
        )
        if match:
            return match.group().strip()

        # Generic 10-digit number (avoid short numbers like years, counts)
        match = re.search(
            r'\b\d{3}[-\s\.]?\d{3}[-\s\.]?\d{4}\b',
            text
        )
        if match:
            return match.group().strip()

        return None


# Factory function
def create_section_aware_processor(
    use_ocr: bool = True,
    use_numarkdown: bool = False
) -> SectionAwareResumeProcessor:
    """
    Create a section-aware resume processor.
    
    Args:
        use_ocr: Enable OCR for images
        use_numarkdown: Use NuMarkdown (expensive) vs Tesseract
        
    Returns:
        Configured SectionAwareResumeProcessor
    """
    return SectionAwareResumeProcessor(
        use_ocr=use_ocr,
        use_numarkdown=use_numarkdown
    )
