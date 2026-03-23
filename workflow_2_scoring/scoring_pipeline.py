"""
Section-Aware Scoring Pipeline (V2)
Matches applicants against jobs using section-wise similarity and weighted scoring.
"""

from typing import List, Dict, Any, Optional, Tuple
from uuid import UUID
from decimal import Decimal

from loguru import logger
import numpy as np

from segmentation import segment_job_description, Section
from embeddings.bert_embedder_v2 import get_section_embedder
from database.supabase_client_v2 import get_db_client
from database.models_v2 import (
    JobDescription,
    JobDescriptionCreate,
    JobEmbeddingCreate,
    MatchHistoryCreate,
    MatchResultResponse
)


class SectionAwareScoringPipeline:
    """
    Scores applicants against job descriptions using section-wise matching.
    
    Workflow:
        1. Segment and embed job description (cached)
        2. For each applicant:
            a. Fetch their section embeddings
            b. Compute section-to-section similarities
            c. Apply configurable weights
            d. Calculate composite score
        3. Write scores back to profiles table
        4. Log match history
    """
    
    # Default section-to-section mappings
    SECTION_MAPPINGS = [
        ('skills', 'requirements', 'skills_to_requirements'),
        ('work_experience', 'responsibilities', 'experience_to_responsibilities'),
        ('education', 'qualifications', 'education_to_qualifications'),
        ('summary', 'overview', 'summary_to_overview'),
    ]
    
    def __init__(self, config_name: str = "default"):
        """
        Initialize scoring pipeline.
        
        Args:
            config_name: Name of scoring configuration to use
        """
        self.embedder = get_section_embedder()
        self.db = get_db_client()
        self.config_name = config_name
        
        # Load scoring configuration
        self.config = self.db.get_scoring_config(config_name)
        if not self.config:
            logger.warning(f"Config '{config_name}' not found, using default weights")
            self.weights = {
                "skills_to_requirements": 0.40,
                "experience_to_responsibilities": 0.35,
                "education_to_qualifications": 0.15,
                "summary_to_overview": 0.10
            }
        else:
            self.weights = self.config.weights
        
        logger.info(f"Scoring pipeline initialized with config: {config_name}")
        logger.info(f"Weights: {self.weights}")
    
    def process_job_description(
        self,
        job_text: str,
        title: str,
        company: Optional[str] = None,
        location: Optional[str] = None,
        job_type: Optional[str] = None
    ) -> UUID:
        """
        Process and store a job description with embeddings.
        
        Args:
            job_text: Full job description text
            title: Job title
            company: Company name
            location: Job location
            job_type: Employment type (full-time, etc.)
            
        Returns:
            job_id UUID
        """
        logger.info(f"Processing job description: {title}")
        
        try:
            # Create job description record
            job_data = JobDescriptionCreate(
                title=title,
                company=company,
                location=location,
                job_type=job_type,
                description=job_text,
                raw_text=job_text
            )
            
            job = self.db.create_job_description(job_data)
            if not job:
                raise Exception("Failed to create job description")
            
            job_id = job.job_id
            
            # Segment job description
            sections, metadata = segment_job_description(job_text)
            
            logger.info(f"Segmented JD into {len(sections)} sections")
            
            # Generate embeddings
            section_dicts = []
            for section in sections:
                section_dicts.append({
                    'section_type': section.section_type,
                    'text': section.text,
                    'char_offset_start': section.char_offset_start,
                    'char_offset_end': section.char_offset_end,
                    'section_order': section.section_order,
                    'confidence_score': section.confidence_score
                })
            
            embedded_sections = self.embedder.embed_sections(
                section_dicts,
                show_progress=False
            )
            
            # Store embeddings
            embedding_data_list = []
            for section in embedded_sections:
                embedding_data = JobEmbeddingCreate(
                    job_id=job_id,
                    section_type=section['section_type'],
                    section_text=section['text'],
                    embedding_vector=section['embedding'].tolist(),
                    char_offset_start=section['char_offset_start'],
                    char_offset_end=section['char_offset_end'],
                    section_order=section['section_order'],
                    confidence_score=Decimal(str(section['confidence_score']))
                )
                embedding_data_list.append(embedding_data)
            
            count = self.db.create_job_embeddings_batch(embedding_data_list)
            
            logger.success(
                f"Job {job_id} processed with {count} section embeddings"
            )
            
            return job_id
            
        except Exception as e:
            logger.error(f"Job processing failed: {e}")
            raise
    
    def score_applicants_for_job(
        self,
        job_id: UUID,
        applicant_ids: Optional[List[UUID]] = None,
        limit: Optional[int] = None
    ) -> List[MatchResultResponse]:
        """
        Score applicants against a job.
        
        Args:
            job_id: Job UUID
            applicant_ids: Optional list of specific applicants to score (None = all)
            limit: Limit number of results
            
        Returns:
            List of MatchResultResponse sorted by score (descending)
        """
        logger.info(f"Scoring applicants for job {job_id}")
        
        try:
            # Fetch job embeddings
            job_embeddings = self.db.get_job_embeddings(job_id)
            if not job_embeddings:
                raise ValueError(f"No embeddings found for job {job_id}")
            
            # Convert to dict for easy lookup
            job_emb_map = {
                emb.section_type: np.array(emb.embedding_vector)
                for emb in job_embeddings
            }
            
            logger.info(f"Loaded {len(job_emb_map)} job section embeddings")
            
            # Get applicants to score
            if applicant_ids:
                applicants = [
                    self.db.get_applicant_profile(aid)
                    for aid in applicant_ids
                ]
                applicants = [a for a in applicants if a is not None]
            else:
                applicants = self.db.get_all_applicants(limit=limit)
            
            logger.info(f"Scoring {len(applicants)} applicants...")
            
            results = []
            
            for applicant in applicants:
                try:
                    # Fetch applicant embeddings
                    applicant_embeddings = self.db.get_applicant_embeddings(
                        applicant.applicant_id,
                        active_only=True
                    )
                    
                    if not applicant_embeddings:
                        logger.warning(
                            f"No embeddings for applicant {applicant.applicant_id}, skipping"
                        )
                        continue
                    
                    # Convert to dict
                    applicant_emb_map = {
                        emb.section_type: np.array(emb.embedding_vector)
                        for emb in applicant_embeddings
                    }
                    
                    # Compute weighted score
                    score_result = self._compute_weighted_score(
                        applicant_emb_map,
                        job_emb_map
                    )
                    
                    # Update applicant profile with score
                    self.db.update_match_score(
                        applicant.applicant_id,
                        score_result['overall_score'],
                        job_id
                    )
                    
                    # Log match history
                    match_data = MatchHistoryCreate(
                        applicant_id=applicant.applicant_id,
                        job_id=job_id,
                        overall_score=Decimal(str(score_result['overall_score'])),
                        section_scores=score_result['section_scores'],
                        config_name=self.config_name,
                        weights_used=self.weights
                    )
                    self.db.create_match_history(match_data)
                    
                    # Add to results
                    results.append(MatchResultResponse(
                        applicant_id=applicant.applicant_id,
                        name=applicant.name,
                        email=applicant.email,
                        overall_score=score_result['overall_score'],
                        section_scores=score_result['section_scores'],
                        sections_matched=score_result['sections_matched']
                    ))
                    
                except Exception as e:
                    logger.error(
                        f"Error scoring applicant {applicant.applicant_id}: {e}"
                    )
                    continue
            
            # Sort by score (descending)
            results.sort(key=lambda x: x.overall_score, reverse=True)
            
            # Add rank
            for rank, result in enumerate(results, start=1):
                result.rank = rank
            
            logger.success(f"Scored {len(results)} applicants successfully")
            
            return results
            
        except Exception as e:
            logger.error(f"Scoring failed: {e}")
            raise
    
    def _compute_weighted_score(
        self,
        applicant_emb_map: Dict[str, np.ndarray],
        job_emb_map: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Compute weighted match score between applicant and job.
        Handles missing sections by redistributing weights.
        
        Args:
            applicant_emb_map: Dict mapping section_type to embedding vector
            job_emb_map: Dict mapping section_type to embedding vector
            
        Returns:
            Dict with overall_score, section_scores, sections_matched
        """
        section_scores = {}
        applicable_weights = []
        weighted_scores = []
        
        for applicant_section, job_section, weight_key in self.SECTION_MAPPINGS:
            if applicant_section in applicant_emb_map and job_section in job_emb_map:
                # Compute cosine similarity
                similarity = self._cosine_similarity(
                    applicant_emb_map[applicant_section],
                    job_emb_map[job_section]
                )
                
                section_scores[weight_key] = round(similarity, 4)
                
                # Apply weight
                if weight_key in self.weights:
                    applicable_weights.append(self.weights[weight_key])
                    weighted_scores.append(similarity * self.weights[weight_key])
        
        # Handle missing sections: redistribute weights proportionally
        if not applicable_weights:
            overall_score = 0.0
        else:
            # Normalize weights to sum to 1.0
            total_weight = sum(applicable_weights)
            overall_score = sum(weighted_scores) / total_weight if total_weight > 0 else 0.0
        
        return {
            "overall_score": round(overall_score, 4),
            "section_scores": section_scores,
            "sections_matched": len(section_scores)
        }
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return float(np.clip(similarity, 0.0, 1.0))
    
    def get_top_candidates(
        self,
        job_id: UUID,
        limit: int = 50
    ) -> List[MatchResultResponse]:
        """
        Get top N candidates for a job (must run score_applicants_for_job first).
        
        Args:
            job_id: Job UUID
            limit: Number of top candidates to return
            
        Returns:
            List of top candidates ranked by score
        """
        try:
            candidates = self.db.get_top_candidates_for_job(job_id, limit=limit)
            
            results = []
            for rank, (profile, score) in enumerate(candidates, start=1):
                # Get section scores from latest match history
                # (simplified - could fetch from match_history table)
                results.append(MatchResultResponse(
                    applicant_id=profile.applicant_id,
                    name=profile.name,
                    email=profile.email,
                    overall_score=float(score),
                    section_scores={},  # Fetch from match_history if needed
                    sections_matched=0,  # Fetch from match_history if needed
                    rank=rank
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"Error fetching top candidates: {e}")
            return []


# Factory function
def create_scoring_pipeline(config_name: str = "default") -> SectionAwareScoringPipeline:
    """
    Create a section-aware scoring pipeline.
    
    Args:
        config_name: Name of scoring config to use
        
    Returns:
        Configured SectionAwareScoringPipeline
    """
    return SectionAwareScoringPipeline(config_name=config_name)
