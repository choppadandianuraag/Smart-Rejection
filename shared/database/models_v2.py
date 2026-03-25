"""
Database Models for Section-Aware Resume Screening (V2)
Pydantic models for the new schema with section-aware storage.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, EmailStr
from uuid import UUID, uuid4
from decimal import Decimal


# ============================================================================
# Applicant Models
# ============================================================================

class ApplicantProfileCreate(BaseModel):
    """Model for creating a new applicant profile."""
    name: str = Field(..., description="Applicant full name")
    email: EmailStr = Field(..., description="Applicant email address")
    contact_number: Optional[str] = Field(None, description="Phone number")
    original_filename: str = Field(..., description="Original resume filename")
    file_type: str = Field(..., description="File extension")
    file_size_bytes: int = Field(..., description="File size in bytes")
    raw_text: str = Field(..., description="Full extracted text")
    segmentation_confidence: Optional[Decimal] = Field(None, description="Overall segmentation confidence")
    needs_manual_review: bool = Field(False, description="Flag for manual review")
    review_reason: Optional[str] = Field(None, description="Reason for manual review")


class ApplicantProfile(BaseModel):
    """Complete applicant profile model."""
    applicant_id: UUID = Field(default_factory=uuid4)
    name: str
    email: EmailStr
    contact_number: Optional[str] = None
    match_score: Optional[Decimal] = None
    last_scored_job_id: Optional[UUID] = None
    original_filename: Optional[str] = None
    file_type: Optional[str] = None
    file_size_bytes: Optional[int] = None
    raw_text: Optional[str] = None
    segmentation_confidence: Optional[Decimal] = None
    needs_manual_review: bool = False
    review_reason: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_scored_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


# ============================================================================
# Embedding Models (Flat structure - one row per applicant)
# ============================================================================

class ApplicantEmbeddingCreate(BaseModel):
    """Model for creating applicant embedding (flat structure)."""
    applicant_id: UUID

    # Section texts (for display/reference)
    skills_text: Optional[str] = None
    education_text: Optional[str] = None
    work_experience_text: Optional[str] = None
    projects_text: Optional[str] = None
    certifications_text: Optional[str] = None
    summary_text: Optional[str] = None

    # ONE combined embedding (all sections except contact_info)
    resume_embedding: List[float] = Field(..., description="768-dim BERT vector of combined resume")


class ApplicantEmbedding(BaseModel):
    """Complete applicant embedding model (flat structure)."""
    id: Optional[int] = None  # Made optional for V1 compatibility
    applicant_id: UUID

    # Section texts
    skills_text: Optional[str] = None
    education_text: Optional[str] = None
    work_experience_text: Optional[str] = None
    projects_text: Optional[str] = None
    certifications_text: Optional[str] = None
    summary_text: Optional[str] = None

    # Combined embedding
    resume_embedding: Optional[List[float]] = None

    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        from_attributes = True


# ============================================================================
# Job Description Models
# ============================================================================

class JobDescriptionCreate(BaseModel):
    """Model for creating a job description."""
    title: str
    company: Optional[str] = None
    location: Optional[str] = None
    job_type: Optional[str] = None
    description: str
    raw_text: str


class JobDescription(BaseModel):
    """Complete job description model."""
    job_id: UUID = Field(default_factory=uuid4)
    title: str
    company: Optional[str] = None
    location: Optional[str] = None
    job_type: Optional[str] = None
    description: str
    raw_text: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = True
    
    class Config:
        from_attributes = True


class JobEmbeddingCreate(BaseModel):
    """Model for creating job section embedding."""
    job_id: UUID
    section_type: str
    section_text: str
    embedding_vector: List[float]
    char_offset_start: int
    char_offset_end: int
    section_order: int
    confidence_score: Optional[Decimal] = None


class JobEmbedding(BaseModel):
    """Complete job embedding model."""
    id: int
    job_id: UUID
    section_type: str
    section_text: str
    embedding_vector: List[float]
    char_offset_start: int
    char_offset_end: int
    section_order: int
    confidence_score: Optional[Decimal] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        from_attributes = True


# ============================================================================
# Scoring Models
# ============================================================================

class ScoringConfig(BaseModel):
    """Scoring configuration model."""
    id: Optional[int] = None
    config_name: str
    weights: Dict[str, float] = Field(..., description="Section-to-section weights")
    description: Optional[str] = None
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None
    
    class Config:
        from_attributes = True


class MatchHistoryCreate(BaseModel):
    """Model for creating a match history record."""
    applicant_id: UUID
    job_id: UUID
    overall_score: Decimal
    section_scores: Dict[str, float] = Field(..., description="Individual section scores")
    config_name: str
    weights_used: Dict[str, Any]  # Changed to Any to support nested dicts (ats_components)


class MatchHistory(BaseModel):
    """Complete match history model."""
    id: int
    applicant_id: UUID
    job_id: UUID
    overall_score: Decimal
    section_scores: Dict[str, float]
    config_name: str
    weights_used: Dict[str, Any]  # Changed to Any to support nested dicts
    scored_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        from_attributes = True


# ============================================================================
# Response Models
# ============================================================================

class ApplicantProfileResponse(BaseModel):
    """Response model for applicant profile API."""
    applicant_id: UUID
    name: str
    email: str
    match_score: Optional[float] = None
    needs_manual_review: bool
    section_count: Optional[int] = None
    created_at: datetime
    
    class Config:
        from_attributes = True


class MatchResultResponse(BaseModel):
    """Response model for matching results."""
    applicant_id: UUID
    name: str
    email: str
    overall_score: float
    section_scores: Dict[str, float]
    sections_matched: int
    rank: Optional[int] = None


class SectionSummary(BaseModel):
    """Summary of a section."""
    section_type: str
    text_preview: str = Field(..., description="First 100 chars of section text")
    confidence_score: Optional[float] = None
    has_embedding: bool = True


class ApplicantDetailResponse(BaseModel):
    """Detailed applicant response with sections."""
    applicant_id: UUID
    name: str
    email: str
    contact_number: Optional[str]
    match_score: Optional[float]
    sections: List[SectionSummary]
    needs_manual_review: bool
    review_reason: Optional[str]
    created_at: datetime
