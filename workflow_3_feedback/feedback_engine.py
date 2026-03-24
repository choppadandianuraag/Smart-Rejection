"""
RAG Engine for generating feedback emails.

Adapted from /Users/anuraag/Python/Customer Support Agent/rag_engine.py
for skills gap feedback generation.
"""

import os
import sys
import asyncio
from pathlib import Path
from typing import List, Dict, Optional, Any

from loguru import logger
from huggingface_hub import InferenceClient
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))

from .vector_store import ResumeVectorStore, get_vector_store
from .prompts import FEEDBACK_SYSTEM_PROMPT, FEEDBACK_EMAIL_TEMPLATE

# Try to import EnsembleRetriever
try:
    from langchain.retrievers import EnsembleRetriever
except ImportError:
    try:
        from langchain_classic.retrievers import EnsembleRetriever
    except ImportError:
        # Fallback implementation
        class EnsembleRetriever:
            def __init__(self, retrievers, weights):
                self.retrievers = retrievers
                self.weights = weights

            def invoke(self, query):
                all_docs = []
                for retriever in self.retrievers:
                    all_docs.extend(retriever.invoke(query))
                return all_docs[:10]


class CrossEncoderReranker:
    """Cross-encoder reranker for improving retrieval quality."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """Initialize the cross-encoder model."""
        logger.info(f"Loading cross-encoder model: {model_name}")
        self.model = CrossEncoder(model_name)
        logger.success("Cross-encoder model loaded successfully")

    def rerank(self, query: str, documents: List[Document], top_k: int = 3) -> List[Document]:
        """
        Rerank documents based on relevance to query.

        Args:
            query: Search query
            documents: List of documents to rerank
            top_k: Number of top documents to return

        Returns:
            Top_k reranked documents with scores
        """
        if not documents:
            return []

        # Create query-document pairs
        pairs = [[query, doc.page_content] for doc in documents]

        # Score all pairs
        scores = self.model.predict(pairs)

        # Sort by scores (descending)
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Get top_k documents
        reranked_docs = [doc for doc, score in scored_docs[:top_k]]

        # Add scores to metadata
        for i, (doc, score) in enumerate(scored_docs[:top_k]):
            doc.metadata['rerank_score'] = float(score)
            doc.metadata['rerank_position'] = i + 1

        return reranked_docs


class FeedbackRAGEngine:
    """
    RAG engine for generating feedback emails.

    Flow:
        1. Fetch applicant resume from ChromaDB (by applicant_id)
        2. Fetch job requirements (from Supabase or ChromaDB)
        3. Optional: Hybrid retrieval + cross-encoder reranking
        4. Generate personalized feedback email via LLM
    """

    LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"

    def __init__(
        self,
        hf_token: Optional[str] = None,
        vector_store: Optional[ResumeVectorStore] = None,
        llm_model: Optional[str] = None
    ):
        """
        Initialize feedback RAG engine.

        Args:
            hf_token: HuggingFace API token
            vector_store: Optional pre-initialized vector store
            llm_model: Optional LLM model name override
        """
        self.hf_token = self._get_hf_token(hf_token)
        self.llm_model = llm_model or self.LLM_MODEL
        self.vector_store = vector_store
        self.reranker = None
        self.hf_client = None
        self._initialized = False

    def _get_hf_token(self, token: Optional[str]) -> str:
        """Get and validate HF token."""
        token = token or os.environ.get("HF_TOKEN")
        if token:
            return token.strip().strip("'").strip('"')
        raise ValueError("HF_TOKEN not found. Set it in environment or pass to constructor.")

    async def initialize(self):
        """Initialize all RAG components asynchronously."""
        if self._initialized:
            return

        logger.info("Initializing Feedback RAG Engine...")

        # 1. Initialize vector store
        if self.vector_store is None:
            logger.info("Creating vector store...")
            self.vector_store = get_vector_store()

        # 2. Initialize reranker
        logger.info("Initializing cross-encoder reranker...")
        self.reranker = CrossEncoderReranker()

        # 3. Initialize LLM client
        logger.info(f"Initializing LLM client: {self.llm_model}")
        self.hf_client = InferenceClient(
            model=self.llm_model,
            token=self.hf_token
        )

        self._initialized = True
        logger.success("Feedback RAG Engine initialized successfully")

    def initialize_sync(self):
        """Synchronous initialization for non-async contexts."""
        asyncio.run(self.initialize())

    async def generate_feedback_email(
        self,
        applicant_id: str,
        job_id: str,
        applicant_name: str,
        job_title: str,
        match_score: float,
        db_client=None
    ) -> str:
        """
        Generate personalized rejection feedback email.

        Args:
            applicant_id: Applicant UUID
            job_id: Job UUID
            applicant_name: Candidate's name
            job_title: Job title
            match_score: Match score (0-1)
            db_client: Optional Supabase client for lazy loading

        Returns:
            Generated email content
        """
        if not self._initialized:
            await self.initialize()

        logger.info(f"Generating feedback for applicant {applicant_id}, job {job_id}")

        # 1. Get resume context
        resume_context = await self._get_resume_context(applicant_id, db_client)

        # 2. Get job requirements
        job_requirements = await self._get_job_context(job_id, db_client)

        # 3. Generate email using LLM
        email_content = await self._call_llm(
            resume_context=resume_context,
            job_requirements=job_requirements,
            candidate_name=applicant_name,
            job_title=job_title,
            match_score=match_score
        )

        logger.success(f"Feedback email generated for {applicant_name}")
        return email_content

    async def _get_resume_context(
        self,
        applicant_id: str,
        db_client=None
    ) -> str:
        """
        Get resume context from ChromaDB, with lazy loading from Supabase.

        Args:
            applicant_id: Applicant UUID
            db_client: Optional Supabase client

        Returns:
            Formatted resume context string
        """
        # Check if resume exists in ChromaDB
        if not self.vector_store.applicant_exists(applicant_id):
            logger.info(f"Resume not in ChromaDB, syncing from Supabase...")

            if db_client is None:
                # Lazy import to avoid circular imports
                from database.supabase_client_v2 import get_db_client
                db_client = get_db_client()

            # Sync from Supabase
            success = self.vector_store.sync_resume_from_supabase(applicant_id, db_client)
            if not success:
                logger.warning(f"Could not sync resume for {applicant_id}")
                return "Resume information not available."

        # Retrieve from ChromaDB
        documents = self.vector_store.get_applicant_resume(applicant_id)
        return self.vector_store.format_resume_context(documents)

    async def _get_job_context(
        self,
        job_id: str,
        db_client=None
    ) -> str:
        """
        Get job context from ChromaDB, with lazy loading from Supabase.

        Args:
            job_id: Job UUID
            db_client: Optional Supabase client

        Returns:
            Formatted job requirements string
        """
        # Check if job exists in ChromaDB
        if not self.vector_store.job_exists(job_id):
            logger.info(f"Job not in ChromaDB, syncing from Supabase...")

            if db_client is None:
                from database.supabase_client_v2 import get_db_client
                db_client = get_db_client()

            # Sync from Supabase
            success = self.vector_store.sync_job_from_supabase(job_id, db_client)
            if not success:
                logger.warning(f"Could not sync job for {job_id}")
                return "Job requirements not available."

        # Retrieve from ChromaDB
        documents = self.vector_store.get_job_requirements(job_id)
        return self.vector_store.format_job_context(documents)

    async def _call_llm(
        self,
        resume_context: str,
        job_requirements: str,
        candidate_name: str,
        job_title: str,
        match_score: float
    ) -> str:
        """
        Call LLM to generate feedback email.

        Args:
            resume_context: Formatted resume text
            job_requirements: Formatted job requirements
            candidate_name: Candidate's name
            job_title: Job title
            match_score: Match score (0-1)

        Returns:
            Generated email content
        """
        # Format the prompt
        user_prompt = FEEDBACK_EMAIL_TEMPLATE.format(
            resume_context=resume_context,
            job_requirements=job_requirements,
            candidate_name=candidate_name,
            job_title=job_title,
            match_score=round(match_score * 100, 1)
        )

        # Build messages
        messages = [
            {"role": "system", "content": FEEDBACK_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        try:
            logger.info(f"Calling LLM ({self.llm_model})...")

            # Run in thread to avoid blocking
            response = await asyncio.to_thread(
                self.hf_client.chat_completion,
                messages,
                max_tokens=1500,
                temperature=0.3
            )

            # Extract content from response
            try:
                content = response.choices[0].message["content"]
            except (KeyError, TypeError, AttributeError):
                try:
                    content = response.choices[0].message.content
                except Exception:
                    logger.error(f"Error parsing LLM response: {type(response)}")
                    content = str(response)

            logger.success("LLM response received")
            return content

        except Exception as e:
            logger.error(f"Error calling LLM: {type(e).__name__}: {str(e)}")
            return self._generate_fallback_email(candidate_name, job_title)

    def _generate_fallback_email(self, candidate_name: str, job_title: str) -> str:
        """Generate a generic fallback email if LLM fails."""
        return f"""Dear {candidate_name},

Thank you for your interest in the {job_title} position and for taking the time to apply.

After careful consideration, we have decided to move forward with candidates whose skills and experience more closely align with the current requirements of this role.

We encourage you to:
1. Continue developing your technical skills relevant to this field
2. Gain additional hands-on experience through projects or internships
3. Consider expanding your knowledge in emerging technologies

We appreciate your interest in our company and encourage you to apply for future positions that match your qualifications.

Best regards,
Hiring Team"""


# Factory function
def create_feedback_engine(
    hf_token: Optional[str] = None,
    vector_store: Optional[ResumeVectorStore] = None,
    llm_model: Optional[str] = None
) -> FeedbackRAGEngine:
    """
    Create a feedback RAG engine.

    Args:
        hf_token: HuggingFace API token
        vector_store: Optional pre-initialized vector store
        llm_model: Optional LLM model override

    Returns:
        Configured FeedbackRAGEngine
    """
    return FeedbackRAGEngine(
        hf_token=hf_token,
        vector_store=vector_store,
        llm_model=llm_model
    )


# Singleton instance
_engine: Optional[FeedbackRAGEngine] = None


def get_feedback_engine() -> FeedbackRAGEngine:
    """Get or create singleton feedback engine instance."""
    global _engine
    if _engine is None:
        _engine = create_feedback_engine()
    return _engine
