"""
Workflow 2 - Scoring & Ranking

Calculates match scores between applicants and job descriptions.

Components:
- scoring_pipeline: Section-wise similarity scoring with configurable weights
- ats_ranking: ATS-style ranking algorithms
- rank_resumes: Resume ranking utilities

Trigger: Scheduled (daily) or manual
Processing: Section-wise cosine similarity (50/30/20 weighted)
Output: Updates match_score, tags rejected candidates for feedback
"""

from .scoring_pipeline import (
    SectionAwareScoringPipeline,
    create_scoring_pipeline
)

__all__ = [
    'SectionAwareScoringPipeline',
    'create_scoring_pipeline'
]
