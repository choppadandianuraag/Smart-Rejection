"""
Workflow 3: Skills Gap Feedback for Rejected Resumes

This workflow generates personalized feedback emails for candidates who were
rejected (match score below threshold). It uses a RAG pipeline with:
- ChromaDB for resume/job storage (indexed by applicant_id/job_id)
- Hybrid retrieval (Dense + BM25) with cross-encoder reranking
- Qwen LLM for feedback email generation
"""

from .feedback_engine import FeedbackRAGEngine, create_feedback_engine
from .vector_store import ResumeVectorStore
from .prompts import FEEDBACK_SYSTEM_PROMPT, FEEDBACK_EMAIL_TEMPLATE

__all__ = [
    "FeedbackRAGEngine",
    "create_feedback_engine",
    "ResumeVectorStore",
    "FEEDBACK_SYSTEM_PROMPT",
    "FEEDBACK_EMAIL_TEMPLATE",
]
