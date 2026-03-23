"""
Database models for the Smart Rejection system.
Defines Pydantic models for data validation and serialization.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from uuid import uuid4


class ResumeBase(BaseModel):
    """Base resume model with common fields."""
    
    filename: str = Field(..., description="Original filename of the resume")
    file_type: str = Field(..., description="File extension (pdf, docx, etc.)")
    file_size_bytes: int = Field(..., description="File size in bytes")


class ResumeCreate(ResumeBase):
    """Model for creating a new resume entry."""
    
    raw_text: str = Field(..., description="Raw extracted text from resume")
    markdown_content: str = Field(..., description="Structured markdown content")
    extracted_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parsed structured data (name, email, skills, etc.)"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about extraction"
    )


class Resume(ResumeBase):
    """Complete resume model with all fields."""
    
    id: str = Field(default_factory=lambda: str(uuid4()))
    raw_text: str = Field(..., description="Raw extracted text from resume")
    markdown_content: str = Field(..., description="Structured markdown content")
    extracted_data: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Embedding fields (for future use)
    embedding_vector: Optional[List[float]] = Field(
        None, 
        description="TF-IDF or other embedding vector"
    )
    embedding_model: Optional[str] = Field(
        None,
        description="Model used for generating embeddings"
    )
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Processing status
    processing_status: str = Field(
        default="pending",
        description="Status: pending, processing, completed, failed"
    )
    error_message: Optional[str] = None
    
    class Config:
        from_attributes = True


class ResumeResponse(BaseModel):
    """Response model for API responses."""
    
    id: str
    filename: str
    file_type: str
    processing_status: str
    created_at: datetime
    extracted_data: Dict[str, Any]
    
    class Config:
        from_attributes = True


class ExtractedResumeData(BaseModel):
    """Structured data extracted from resume."""
    
    # Personal Information
    full_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    linkedin_url: Optional[str] = None
    github_url: Optional[str] = None
    portfolio_url: Optional[str] = None
    
    # Professional Summary
    summary: Optional[str] = None
    objective: Optional[str] = None
    
    # Experience
    work_experience: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Education
    education: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Skills
    technical_skills: List[str] = Field(default_factory=list)
    soft_skills: List[str] = Field(default_factory=list)
    languages: List[str] = Field(default_factory=list)
    
    # Other Sections
    certifications: List[Dict[str, Any]] = Field(default_factory=list)
    projects: List[Dict[str, Any]] = Field(default_factory=list)
    achievements: List[str] = Field(default_factory=list)
    publications: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Raw sections for backup
    raw_sections: Dict[str, str] = Field(default_factory=dict)
