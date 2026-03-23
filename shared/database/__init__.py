"""Shared database package."""
from .models import Resume, ResumeCreate, ResumeResponse, ExtractedResumeData
from .supabase_client import (
    SupabaseClient,
    ResumeRepository,
    get_resume_repository
)
from .supabase_client_v2 import (
    SupabaseClientV2,
    get_db_client
)
from .models_v2 import (
    ApplicantProfile,
    ApplicantEmbedding,
    JobDescription,
    JobEmbedding,
    MatchHistory,
    ScoringConfig
)

__all__ = [
    "Resume",
    "ResumeCreate",
    "ResumeResponse",
    "ExtractedResumeData",
    "SupabaseClient",
    "ResumeRepository",
    "get_resume_repository",
    "SupabaseClientV2",
    "get_db_client",
    "ApplicantProfile",
    "ApplicantEmbedding",
    "JobDescription",
    "JobEmbedding",
    "MatchHistory",
    "ScoringConfig"
]
