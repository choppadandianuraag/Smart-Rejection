"""
Vector Store for Resume and Job embeddings using ChromaDB.

Manages two collections:
1. resumes - indexed by applicant_id, stores resume sections
2. jobs - indexed by job_id, stores job requirement sections
"""

import os
from pathlib import Path
from typing import List, Dict, Optional
from loguru import logger

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document


class ResumeVectorStore:
    """
    ChromaDB store with resumes and jobs collections.

    Resumes are indexed by applicant_id for filtered retrieval.
    Jobs are indexed by job_id and synced from Supabase.
    """

    EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
    RESUME_SECTIONS = ["skills", "experience", "education", "projects", "certifications", "summary"]
    JOB_SECTIONS = ["requirements", "responsibilities", "qualifications", "overview"]

    def __init__(self, persist_dir: str = "chroma_db/feedback"):
        """
        Initialize vector store with ChromaDB.

        Args:
            persist_dir: Directory to persist ChromaDB data
        """
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initializing embeddings model: {self.EMBEDDING_MODEL}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # Initialize resumes collection
        logger.info("Initializing resumes collection...")
        self.resumes_collection = Chroma(
            persist_directory=str(self.persist_dir / "resumes"),
            embedding_function=self.embeddings,
            collection_name="resumes"
        )

        # Initialize jobs collection
        logger.info("Initializing jobs collection...")
        self.jobs_collection = Chroma(
            persist_directory=str(self.persist_dir / "jobs"),
            embedding_function=self.embeddings,
            collection_name="jobs"
        )

        logger.success("Vector store initialized successfully")

    # ========================================================================
    # Resume Operations
    # ========================================================================

    def add_resume(self, applicant_id: str, sections: Dict[str, str]) -> bool:
        """
        Add resume sections with applicant_id metadata.

        Args:
            applicant_id: Unique applicant UUID
            sections: Dict mapping section name to text content
                      e.g., {"skills": "Python, Java...", "experience": "..."}

        Returns:
            True if successful
        """
        try:
            documents = []
            metadatas = []
            ids = []

            for section_name, section_text in sections.items():
                if section_text and section_text.strip():
                    documents.append(section_text)
                    metadatas.append({
                        "applicant_id": applicant_id,
                        "section": section_name
                    })
                    ids.append(f"{applicant_id}_{section_name}")

            if documents:
                self.resumes_collection.add_texts(
                    texts=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                logger.info(f"Added {len(documents)} sections for applicant {applicant_id}")
                return True

            logger.warning(f"No valid sections to add for applicant {applicant_id}")
            return False

        except Exception as e:
            logger.error(f"Error adding resume for {applicant_id}: {e}")
            return False

    def get_applicant_resume(self, applicant_id: str) -> List[Document]:
        """
        Retrieve all sections for a specific applicant.

        Args:
            applicant_id: Applicant UUID

        Returns:
            List of Document objects with section content
        """
        try:
            # Use get() with where filter to retrieve by metadata
            results = self.resumes_collection.get(
                where={"applicant_id": applicant_id},
                include=["documents", "metadatas"]
            )

            if not results or not results.get("documents"):
                logger.warning(f"No resume found for applicant {applicant_id}")
                return []

            documents = []
            for i, doc_text in enumerate(results["documents"]):
                metadata = results["metadatas"][i] if results["metadatas"] else {}
                documents.append(Document(
                    page_content=doc_text,
                    metadata=metadata
                ))

            logger.info(f"Retrieved {len(documents)} sections for applicant {applicant_id}")
            return documents

        except Exception as e:
            logger.error(f"Error retrieving resume for {applicant_id}: {e}")
            return []

    def applicant_exists(self, applicant_id: str) -> bool:
        """Check if applicant's resume is already in the store."""
        try:
            results = self.resumes_collection.get(
                where={"applicant_id": applicant_id},
                include=[]
            )
            return bool(results and results.get("ids"))
        except Exception:
            return False

    def delete_applicant(self, applicant_id: str) -> bool:
        """Delete all sections for an applicant."""
        try:
            # Get IDs for this applicant
            results = self.resumes_collection.get(
                where={"applicant_id": applicant_id},
                include=[]
            )

            if results and results.get("ids"):
                self.resumes_collection.delete(ids=results["ids"])
                logger.info(f"Deleted resume for applicant {applicant_id}")
                return True
            return False

        except Exception as e:
            logger.error(f"Error deleting resume for {applicant_id}: {e}")
            return False

    # ========================================================================
    # Job Operations
    # ========================================================================

    def add_job(self, job_id: str, sections: Dict[str, str]) -> bool:
        """
        Add job sections with job_id metadata.

        Args:
            job_id: Unique job UUID
            sections: Dict mapping section name to text content
                      e.g., {"requirements": "5+ years Python...", "responsibilities": "..."}

        Returns:
            True if successful
        """
        try:
            documents = []
            metadatas = []
            ids = []

            for section_name, section_text in sections.items():
                if section_text and section_text.strip():
                    documents.append(section_text)
                    metadatas.append({
                        "job_id": job_id,
                        "section": section_name
                    })
                    ids.append(f"{job_id}_{section_name}")

            if documents:
                self.jobs_collection.add_texts(
                    texts=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                logger.info(f"Added {len(documents)} sections for job {job_id}")
                return True

            logger.warning(f"No valid sections to add for job {job_id}")
            return False

        except Exception as e:
            logger.error(f"Error adding job {job_id}: {e}")
            return False

    def get_job_requirements(self, job_id: str) -> List[Document]:
        """
        Retrieve job sections for a specific job.

        Args:
            job_id: Job UUID

        Returns:
            List of Document objects with job content
        """
        try:
            results = self.jobs_collection.get(
                where={"job_id": job_id},
                include=["documents", "metadatas"]
            )

            if not results or not results.get("documents"):
                logger.warning(f"No job found for job_id {job_id}")
                return []

            documents = []
            for i, doc_text in enumerate(results["documents"]):
                metadata = results["metadatas"][i] if results["metadatas"] else {}
                documents.append(Document(
                    page_content=doc_text,
                    metadata=metadata
                ))

            logger.info(f"Retrieved {len(documents)} sections for job {job_id}")
            return documents

        except Exception as e:
            logger.error(f"Error retrieving job {job_id}: {e}")
            return []

    def job_exists(self, job_id: str) -> bool:
        """Check if job is already in the store."""
        try:
            results = self.jobs_collection.get(
                where={"job_id": job_id},
                include=[]
            )
            return bool(results and results.get("ids"))
        except Exception:
            return False

    # ========================================================================
    # Sync Operations (from Supabase)
    # ========================================================================

    def sync_resume_from_supabase(self, applicant_id: str, db_client) -> bool:
        """
        Sync a single resume from Supabase to ChromaDB.

        Args:
            applicant_id: Applicant UUID
            db_client: Supabase client instance

        Returns:
            True if successful
        """
        try:
            from uuid import UUID

            # Fetch embedding data from Supabase
            embedding = db_client.get_applicant_embedding(UUID(applicant_id))

            if not embedding:
                logger.warning(f"No embedding found in Supabase for {applicant_id}")
                return False

            # Build sections dict from embedding data
            sections = {
                "skills": embedding.skills_text or "",
                "experience": embedding.work_experience_text or "",
                "education": embedding.education_text or "",
                "projects": embedding.projects_text or "",
                "certifications": embedding.certifications_text or "",
                "summary": embedding.summary_text or ""
            }

            return self.add_resume(applicant_id, sections)

        except Exception as e:
            logger.error(f"Error syncing resume from Supabase: {e}")
            return False

    def sync_job_from_supabase(self, job_id: str, db_client) -> bool:
        """
        Sync a job from Supabase to ChromaDB.

        Args:
            job_id: Job UUID
            db_client: Supabase client instance

        Returns:
            True if successful
        """
        try:
            from uuid import UUID

            # Fetch job description from Supabase
            job = db_client.get_job_description(UUID(job_id))

            if not job:
                logger.warning(f"No job found in Supabase for {job_id}")
                return False

            # Use raw_text as the full job description
            # In practice, you might want to segment this further
            sections = {
                "requirements": job.raw_text or job.description or "",
                "title": job.title or "",
                "overview": job.description or ""
            }

            return self.add_job(job_id, sections)

        except Exception as e:
            logger.error(f"Error syncing job from Supabase: {e}")
            return False

    def format_resume_context(self, documents: List[Document]) -> str:
        """
        Format resume documents into a single context string for LLM.

        Args:
            documents: List of Document objects

        Returns:
            Formatted string with section headers
        """
        if not documents:
            return "No resume information available."

        sections = {}
        for doc in documents:
            section_name = doc.metadata.get("section", "general")
            sections[section_name] = doc.page_content

        formatted = []
        section_order = ["summary", "skills", "experience", "education", "projects", "certifications"]

        for section in section_order:
            if section in sections and sections[section]:
                formatted.append(f"**{section.upper()}:**\n{sections[section]}")

        return "\n\n".join(formatted) if formatted else "No resume information available."

    def format_job_context(self, documents: List[Document]) -> str:
        """
        Format job documents into a single context string for LLM.

        Args:
            documents: List of Document objects

        Returns:
            Formatted string with section headers
        """
        if not documents:
            return "No job information available."

        sections = {}
        for doc in documents:
            section_name = doc.metadata.get("section", "general")
            sections[section_name] = doc.page_content

        formatted = []
        section_order = ["title", "overview", "requirements", "responsibilities", "qualifications"]

        for section in section_order:
            if section in sections and sections[section]:
                formatted.append(f"**{section.upper()}:**\n{sections[section]}")

        return "\n\n".join(formatted) if formatted else "No job information available."


# Singleton instance
_vector_store: Optional[ResumeVectorStore] = None


def get_vector_store(persist_dir: str = "chroma_db/feedback") -> ResumeVectorStore:
    """Get or create singleton vector store instance."""
    global _vector_store
    if _vector_store is None:
        _vector_store = ResumeVectorStore(persist_dir=persist_dir)
    return _vector_store
