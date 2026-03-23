"""
Workflow 1 - Resume Preprocessing

Handles resume ingestion, OCR/extraction, segmentation, and embedding generation.

Components:
- extractors: PDF, DOCX, and OCR-based text extraction
- segmentation: Resume section segmentation
- embeddings: S-BERT embedding generation for sections
- ingestion_pipeline: Main orchestration pipeline

Trigger: Form submission with resume upload (webhook)
Output: Stores data in applicant_profiles + applicant_embeddings tables

Usage:
    from workflow_1_preprocessing import SectionAwareResumeProcessor, create_section_aware_processor

    # Create processor
    processor = create_section_aware_processor(use_ocr=True)

    # Process a resume
    result = processor.process_resume("/path/to/resume.pdf")
"""

from .ingestion_pipeline import (
    SectionAwareResumeProcessor,
    create_section_aware_processor
)

__all__ = [
    'SectionAwareResumeProcessor',
    'create_section_aware_processor'
]
