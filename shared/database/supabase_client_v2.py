"""
Supabase Client for Section-Aware Resume Screening (V2)
Handles all database operations for the new schema.
"""

from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from uuid import UUID, uuid4
import json

from supabase import create_client, Client
from loguru import logger

from config.settings import settings
from database.models_v2 import (
    ApplicantProfile, ApplicantProfileCreate,
    ApplicantEmbedding, ApplicantEmbeddingCreate,
    JobDescription, JobDescriptionCreate,
    JobEmbedding, JobEmbeddingCreate,
    MatchHistory, MatchHistoryCreate,
    ScoringConfig
)


class SupabaseClientV2:
    """
    Supabase client for section-aware storage.
    Handles dual-table writes and atomic transactions.
    """

    _instance: Optional['SupabaseClientV2'] = None
    _client: Optional[Client] = None

    # Core columns that must exist in applicant_profiles
    PROFILE_COLUMNS = "applicant_id, name, email, original_filename, file_type, file_size_bytes, raw_text, segmentation_confidence, needs_manual_review, review_reason, created_at, updated_at"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._client is None:
            self._client = create_client(
                settings.supabase_url,
                settings.supabase_key
            )
            logger.info("Supabase V2 client initialized successfully")
    
    @property
    def client(self) -> Client:
        """Get the Supabase client instance."""
        return self._client
    
    # ========================================================================
    # Applicant Profile Operations
    # ========================================================================
    
    def create_applicant_profile(
        self,
        profile_data: ApplicantProfileCreate
    ) -> Optional[ApplicantProfile]:
        """
        Create a new applicant profile.
        
        Args:
            profile_data: Applicant profile creation data
            
        Returns:
            Created ApplicantProfile or None on failure
        """
        try:
            applicant_id = str(uuid4())
            now = datetime.utcnow().isoformat()
            
            data = {
                "applicant_id": applicant_id,
                "name": profile_data.name,
                "email": profile_data.email,
                "original_filename": profile_data.original_filename,
                "file_type": profile_data.file_type,
                "file_size_bytes": profile_data.file_size_bytes,
                "raw_text": profile_data.raw_text,
                "segmentation_confidence": float(profile_data.segmentation_confidence) if profile_data.segmentation_confidence else None,
                "needs_manual_review": profile_data.needs_manual_review,
                "review_reason": profile_data.review_reason,
                "created_at": now,
                "updated_at": now
            }

            # Only add contact_number if it exists
            if profile_data.contact_number:
                data["contact_number"] = profile_data.contact_number
            
            result = self._client.table("applicant_profiles").insert(data).execute()
            
            if result.data:
                logger.success(f"Applicant profile created: {applicant_id}")
                return ApplicantProfile(**result.data[0])
            return None
            
        except Exception as e:
            logger.error(f"Error creating applicant profile: {e}")
            return None
    
    def get_applicant_profile(self, applicant_id: UUID) -> Optional[ApplicantProfile]:
        """Get applicant profile by ID."""
        try:
            result = self._client.table("applicant_profiles").select(self.PROFILE_COLUMNS).eq(
                "applicant_id", str(applicant_id)
            ).execute()
            
            if result.data:
                return ApplicantProfile(**result.data[0])
            return None
            
        except Exception as e:
            logger.error(f"Error fetching applicant profile: {e}")
            return None
    
    def get_applicant_by_email(self, email: str) -> Optional[ApplicantProfile]:
        """Get applicant profile by email."""
        try:
            result = self._client.table("applicant_profiles").select(self.PROFILE_COLUMNS).eq(
                "email", email
            ).execute()
            
            if result.data:
                return ApplicantProfile(**result.data[0])
            return None
            
        except Exception as e:
            logger.error(f"Error fetching applicant by email: {e}")
            return None
    
    def update_match_score(
        self,
        applicant_id: UUID,
        match_score: float,
        job_id: Optional[UUID] = None
    ) -> bool:
        """
        Update applicant's match score (called by scoring pipeline).
        
        Args:
            applicant_id: Applicant UUID
            match_score: Computed match score (0-1)
            job_id: Optional job ID that was matched against
            
        Returns:
            True if successful
        """
        try:
            now = datetime.utcnow().isoformat()
            
            update_data = {
                "match_score": match_score,
                "last_scored_at": now,
                "updated_at": now
            }
            
            if job_id:
                update_data["last_scored_job_id"] = str(job_id)
            
            result = self._client.table("applicant_profiles").update(update_data).eq(
                "applicant_id", str(applicant_id)
            ).execute()
            
            if result.data:
                logger.info(f"Updated match score for {applicant_id}: {match_score:.4f}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error updating match score: {e}")
            return False
    
    # ========================================================================
    # Applicant Embeddings Operations (Flat structure - one row per applicant)
    # ========================================================================

    def create_applicant_embedding(
        self,
        embedding_data: ApplicantEmbeddingCreate
    ) -> Optional[ApplicantEmbedding]:
        """
        Create applicant embedding (flat structure - one row per applicant).

        Args:
            embedding_data: Embedding creation data with section texts + combined embedding

        Returns:
            Created ApplicantEmbedding or None
        """
        try:
            # Convert embedding list to pgvector format string
            embedding_list = embedding_data.resume_embedding
            if isinstance(embedding_list, list):
                embedding_str = "[" + ",".join(str(x) for x in embedding_list) + "]"
            else:
                embedding_str = str(embedding_list)

            data = {
                "applicant_id": str(embedding_data.applicant_id),
                "skills_text": embedding_data.skills_text,
                "education_text": embedding_data.education_text,
                "work_experience_text": embedding_data.work_experience_text,
                "projects_text": embedding_data.projects_text,
                "certifications_text": embedding_data.certifications_text,
                "summary_text": embedding_data.summary_text,
                "resume_embedding": embedding_str,
                "created_at": datetime.utcnow().isoformat()
            }

            result = self._client.table("applicant_embeddings").insert(data).execute()

            if result.data:
                logger.success(f"Created embedding for applicant {embedding_data.applicant_id}")
                # Don't try to parse the result back - it's already stored
                return None  # Return None to avoid parsing issues
            return None

        except Exception as e:
            logger.error(f"Error creating applicant embedding: {e}")
            return None

    def get_applicant_embedding(
        self,
        applicant_id: UUID
    ) -> Optional[ApplicantEmbedding]:
        """
        Get embedding for an applicant (flat structure - returns single row).

        Args:
            applicant_id: Applicant UUID

        Returns:
            ApplicantEmbedding or None
        """
        try:
            result = self._client.table("applicant_embeddings").select("*").eq(
                "applicant_id", str(applicant_id)
            ).execute()

            if result.data:
                return ApplicantEmbedding(**result.data[0])
            return None

        except Exception as e:
            logger.error(f"Error fetching applicant embedding: {e}")
            return None

    def update_applicant_embedding(
        self,
        applicant_id: UUID,
        embedding_data: ApplicantEmbeddingCreate
    ) -> Optional[ApplicantEmbedding]:
        """
        Update existing embedding for an applicant.

        Args:
            applicant_id: Applicant UUID
            embedding_data: New embedding data

        Returns:
            Updated ApplicantEmbedding or None
        """
        try:
            data = {
                "skills_text": embedding_data.skills_text,
                "education_text": embedding_data.education_text,
                "work_experience_text": embedding_data.work_experience_text,
                "projects_text": embedding_data.projects_text,
                "certifications_text": embedding_data.certifications_text,
                "summary_text": embedding_data.summary_text,
                "resume_embedding": embedding_data.resume_embedding
            }

            result = self._client.table("applicant_embeddings").update(data).eq(
                "applicant_id", str(applicant_id)
            ).execute()

            if result.data:
                logger.success(f"Updated embedding for applicant {applicant_id}")
                return ApplicantEmbedding(**result.data[0])
            return None

        except Exception as e:
            logger.error(f"Error updating applicant embedding: {e}")
            return None

    def delete_applicant_embedding(self, applicant_id: UUID) -> bool:
        """
        Delete embedding for an applicant (used when re-processing resume).

        Args:
            applicant_id: Applicant UUID

        Returns:
            True if successful
        """
        try:
            self._client.table("applicant_embeddings").delete().eq(
                "applicant_id", str(applicant_id)
            ).execute()

            logger.info(f"Deleted embedding for {applicant_id}")
            return True

        except Exception as e:
            logger.error(f"Error deleting embedding: {e}")
            return False
    
    # ========================================================================
    # Job Description Operations
    # ========================================================================
    
    def create_job_description(
        self,
        job_data: JobDescriptionCreate
    ) -> Optional[JobDescription]:
        """Create a new job description."""
        try:
            job_id = str(uuid4())
            now = datetime.utcnow().isoformat()
            
            data = {
                "job_id": job_id,
                "title": job_data.title,
                "company": job_data.company,
                "location": job_data.location,
                "job_type": job_data.job_type,
                "description": job_data.description,
                "raw_text": job_data.raw_text,
                "created_at": now,
                "updated_at": now,
                "is_active": True
            }
            
            result = self._client.table("job_descriptions").insert(data).execute()
            
            if result.data:
                logger.success(f"Job description created: {job_id}")
                return JobDescription(**result.data[0])
            return None
            
        except Exception as e:
            logger.error(f"Error creating job description: {e}")
            return None
    
    def get_job_description(self, job_id: UUID) -> Optional[JobDescription]:
        """Get job description by ID."""
        try:
            result = self._client.table("job_descriptions").select("*").eq(
                "job_id", str(job_id)
            ).execute()
            
            if result.data:
                return JobDescription(**result.data[0])
            return None
            
        except Exception as e:
            logger.error(f"Error fetching job description: {e}")
            return None
    
    def create_job_embeddings_batch(
        self,
        embeddings: List[JobEmbeddingCreate]
    ) -> int:
        """Create multiple job embeddings at once."""
        try:
            data_list = []
            for emb in embeddings:
                data_list.append({
                    "job_id": str(emb.job_id),
                    "section_type": emb.section_type,
                    "section_text": emb.section_text,
                    "embedding_vector": emb.embedding_vector,
                    "char_offset_start": emb.char_offset_start,
                    "char_offset_end": emb.char_offset_end,
                    "section_order": emb.section_order,
                    "confidence_score": float(emb.confidence_score) if emb.confidence_score else None,
                    "created_at": datetime.utcnow().isoformat()
                })
            
            result = self._client.table("job_embeddings").insert(data_list).execute()
            
            if result.data:
                count = len(result.data)
                logger.success(f"Created {count} job embeddings")
                return count
            return 0
            
        except Exception as e:
            logger.error(f"Error batch creating job embeddings: {e}")
            return 0
    
    def get_job_embeddings(self, job_id: UUID) -> List[JobEmbedding]:
        """Get all embeddings for a job."""
        try:
            result = self._client.table("job_embeddings").select("*").eq(
                "job_id", str(job_id)
            ).order("section_order").execute()
            
            if result.data:
                return [JobEmbedding(**row) for row in result.data]
            return []
            
        except Exception as e:
            logger.error(f"Error fetching job embeddings: {e}")
            return []
    
    # ========================================================================
    # Scoring Operations
    # ========================================================================
    
    def get_scoring_config(self, config_name: str = "default") -> Optional[ScoringConfig]:
        """Get scoring configuration by name."""
        try:
            result = self._client.table("scoring_config").select("*").eq(
                "config_name", config_name
            ).eq("is_active", True).execute()
            
            if result.data:
                return ScoringConfig(**result.data[0])
            return None
            
        except Exception as e:
            logger.error(f"Error fetching scoring config: {e}")
            return None
    
    def create_match_history(
        self,
        match_data: MatchHistoryCreate
    ) -> Optional[MatchHistory]:
        """Create a match history record."""
        try:
            data = {
                "applicant_id": str(match_data.applicant_id),
                "job_id": str(match_data.job_id),
                "overall_score": float(match_data.overall_score),
                "section_scores": match_data.section_scores,
                "config_name": match_data.config_name,
                "weights_used": match_data.weights_used,
                "scored_at": datetime.utcnow().isoformat()
            }
            
            result = self._client.table("match_history").insert(data).execute()
            
            if result.data:
                return MatchHistory(**result.data[0])
            return None
            
        except Exception as e:
            logger.error(f"Error creating match history: {e}")
            return None
    
    # ========================================================================
    # Query Operations
    # ========================================================================
    
    def get_top_candidates_for_job(
        self,
        job_id: UUID,
        limit: int = 50
    ) -> List[Tuple[ApplicantProfile, float]]:
        """
        Get top N candidates ranked by match score for a job.
        
        Returns:
            List of (ApplicantProfile, score) tuples
        """
        try:
            # Get all applicants with match scores for this job
            result = self._client.table("match_history").select(
                "applicant_id, overall_score"
            ).eq("job_id", str(job_id)).order(
                "overall_score", desc=True
            ).limit(limit).execute()
            
            if not result.data:
                return []
            
            # Fetch full profiles
            candidates = []
            for row in result.data:
                profile = self.get_applicant_profile(UUID(row["applicant_id"]))
                if profile:
                    candidates.append((profile, float(row["overall_score"])))
            
            return candidates
            
        except Exception as e:
            logger.error(f"Error fetching top candidates: {e}")
            return []
    
    def get_all_applicants(
        self,
        limit: Optional[int] = None,
        needs_review_only: bool = False
    ) -> List[ApplicantProfile]:
        """Get all applicant profiles."""
        try:
            query = self._client.table("applicant_profiles").select(self.PROFILE_COLUMNS)
            
            if needs_review_only:
                query = query.eq("needs_manual_review", True)
            
            query = query.order("created_at", desc=True)
            
            if limit:
                query = query.limit(limit)
            
            result = query.execute()
            
            if result.data:
                return [ApplicantProfile(**row) for row in result.data]
            return []
            
        except Exception as e:
            logger.error(f"Error fetching applicants: {e}")
            return []


# Singleton accessor
def get_db_client() -> SupabaseClientV2:
    """Get or create singleton Supabase V2 client."""
    return SupabaseClientV2()
