"""
Embedding Service - Manages embedding generation and storage.
Phase 2: Separate from Phase 1 extraction logic.

Enhanced with:
- Text preprocessing (cleaning, normalization)
- Section-wise embeddings (skills, experience, education separately)
- Hybrid BERT + TF-IDF embeddings for better retrieval
"""

import sys
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from loguru import logger

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from embeddings.tfidf_embedder import TFIDFEmbedder
from embeddings.preprocessor import ResumePreprocessor
from embeddings.bert_embedder import BertEmbedder, SectionWiseEmbedder
from embeddings.hybrid_embedder import HybridEmbedder
from database.supabase_client import SupabaseClient, ResumeRepository


class EmbeddingService:
    """
    Service for generating and storing resume embeddings.
    Keeps Phase 2 (embeddings) separate from Phase 1 (extraction).
    
    Features:
    - Text preprocessing (cleaning, date standardization, normalization)
    - Section-wise embeddings (skills, experience, education, projects)
    - Hybrid BERT + TF-IDF embeddings for better retrieval
    """
    
    DEFAULT_MODEL_PATH = Path(__file__).parent / "models" / "tfidf_vectorizer.pkl"
    DEFAULT_HYBRID_PATH = Path(__file__).parent / "models" / "hybrid"
    
    # Section names for section-wise embedding
    SECTION_NAMES = ["skills", "experience", "education", "projects", "certifications"]
    
    def __init__(
        self,
        ngram_range: tuple = (1, 3),
        max_features: int = 5000,
        use_preprocessing: bool = True,
        bert_model: str = "all-MiniLM-L6-v2",
        bert_weight: float = 0.7,
        tfidf_weight: float = 0.3
    ):
        """
        Initialize the embedding service.
        
        Args:
            ngram_range: N-gram range for TF-IDF.
            max_features: Maximum vocabulary size.
            use_preprocessing: Whether to preprocess text.
            bert_model: BERT model to use for dense embeddings.
            bert_weight: Weight for BERT in hybrid embeddings.
            tfidf_weight: Weight for TF-IDF in hybrid embeddings.
        """
        self.embedder = TFIDFEmbedder(
            ngram_range=ngram_range,
            max_features=max_features
        )
        self.db = SupabaseClient().client
        self.repo = ResumeRepository()
        
        # Preprocessing
        self.use_preprocessing = use_preprocessing
        self.preprocessor = ResumePreprocessor() if use_preprocessing else None
        
        # BERT embedder (lazy initialized)
        self.bert_model_name = bert_model
        self.bert_embedder: Optional[BertEmbedder] = None
        
        # Hybrid embedder configuration
        self.bert_weight = bert_weight
        self.tfidf_weight = tfidf_weight
        self.hybrid_embedder: Optional[HybridEmbedder] = None
        
        # Ensure models directory exists
        self.DEFAULT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.DEFAULT_HYBRID_PATH.mkdir(parents=True, exist_ok=True)
    
    def _init_bert(self):
        """Lazy initialize BERT embedder."""
        if self.bert_embedder is None:
            self.bert_embedder = BertEmbedder(model_name=self.bert_model_name)
    
    def _init_hybrid(self):
        """Lazy initialize hybrid embedder."""
        if self.hybrid_embedder is None:
            self.hybrid_embedder = HybridEmbedder(
                bert_weight=self.bert_weight,
                tfidf_weight=self.tfidf_weight
            )
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess resume text.
        
        Args:
            text: Raw resume text.
            
        Returns:
            Preprocessed text.
        """
        if self.use_preprocessing and self.preprocessor:
            return self.preprocessor.preprocess(text)
        return text
    
    def extract_sections_for_embedding(self, raw_text: str, extracted_data: Dict[str, Any]) -> str:
        """
        Extract and combine Skills, Projects, and Experience sections for embedding.
        
        Args:
            raw_text: Full raw text from resume.
            extracted_data: Parsed structured data.
            
        Returns:
            Combined text focusing on skills, projects, and experience.
        """
        sections = []
        
        # Extract Skills section from raw text
        skills_text = self._extract_section(raw_text, ["SKILLS", "TECHNICAL SKILLS", "CORE SKILLS"])
        if skills_text:
            sections.append(f"SKILLS: {skills_text}")
        
        # Also add parsed skills from extracted_data
        if extracted_data.get('skills'):
            parsed_skills = " ".join(extracted_data['skills'])
            sections.append(f"PARSED SKILLS: {parsed_skills}")
        
        # Extract Projects section from raw text
        projects_text = self._extract_section(raw_text, ["PROJECTS", "PROJECT", "PERSONAL PROJECTS", "ACADEMIC PROJECTS"])
        if projects_text:
            sections.append(f"PROJECTS: {projects_text}")
        
        # Extract Experience section from raw text
        experience_text = self._extract_section(raw_text, ["EXPERIENCE", "WORK EXPERIENCE", "PROFESSIONAL EXPERIENCE", "INTERNSHIP", "INTERNSHIPS"])
        if experience_text:
            sections.append(f"EXPERIENCE: {experience_text}")
        
        # Extract Education section from raw text
        education_text = self._extract_section(raw_text, ["EDUCATION", "ACADEMIC", "QUALIFICATION"])
        if education_text:
            sections.append(f"EDUCATION: {education_text}")
        
        # Extract Certifications section
        certs_text = self._extract_section(raw_text, ["CERTIFICATES", "CERTIFICATIONS", "CERTIFICATION"])
        if certs_text:
            sections.append(f"CERTIFICATIONS: {certs_text}")
        
        # Combine all sections
        combined_text = "\n\n".join(sections)
        
        # If no sections found, fall back to full raw text
        if not combined_text.strip():
            logger.warning("No specific sections found, using full raw text")
            return raw_text
        
        logger.info(f"Extracted {len(sections)} sections for embedding")
        return combined_text
    
    def _extract_section(self, text: str, section_headers: List[str]) -> Optional[str]:
        """
        Extract a section from raw text based on header patterns.
        
        Args:
            text: Full raw text.
            section_headers: List of possible section header names.
            
        Returns:
            Extracted section text or None.
        """
        # Common section headers to detect end of current section
        all_headers = [
            "SKILLS", "TECHNICAL SKILLS", "CORE SKILLS",
            "PROJECTS", "PROJECT", "PERSONAL PROJECTS",
            "EXPERIENCE", "WORK EXPERIENCE", "PROFESSIONAL EXPERIENCE",
            "EDUCATION", "ACADEMIC", "QUALIFICATION",
            "CERTIFICATES", "CERTIFICATIONS", "CERTIFICATION",
            "SUMMARY", "OBJECTIVE", "PROFESSIONAL SUMMARY",
            "ACHIEVEMENTS", "AWARDS", "PUBLICATIONS",
            "LANGUAGES", "INTERESTS", "HOBBIES", "REFERENCES"
        ]
        
        text_upper = text.upper()
        
        for header in section_headers:
            # Find the start of this section
            pattern = rf'\b{re.escape(header)}\b'
            match = re.search(pattern, text_upper)
            
            if match:
                start_pos = match.end()
                
                # Find the next section header (end of current section)
                end_pos = len(text)
                for other_header in all_headers:
                    if other_header in section_headers:
                        continue
                    other_pattern = rf'\b{re.escape(other_header)}\b'
                    other_match = re.search(other_pattern, text_upper[start_pos:])
                    if other_match:
                        potential_end = start_pos + other_match.start()
                        if potential_end < end_pos:
                            end_pos = potential_end
                
                # Extract the section text
                section_text = text[start_pos:end_pos].strip()
                
                # Clean up the section text
                section_text = re.sub(r'\n{3,}', '\n\n', section_text)
                
                if len(section_text) > 20:  # Minimum content threshold
                    return section_text
        
        return None
    
    def fetch_all_resumes(self) -> List[Dict[str, Any]]:
        """
        Fetch all resumes from database.
        
        Returns:
            List of resume records.
        """
        result = self.db.table('resumes').select('id, raw_text, filename, extracted_data').execute()
        return result.data or []
    
    def fetch_resumes_without_embeddings(self) -> List[Dict[str, Any]]:
        """
        Fetch resumes that don't have embeddings yet.
        
        Returns:
            List of resume records without embeddings.
        """
        result = self.db.table('resumes')\
            .select('id, raw_text, filename, extracted_data')\
            .is_('embedding_vector', 'null')\
            .execute()
        return result.data or []
    
    def fit_on_all_resumes(self) -> Dict[str, Any]:
        """
        Fit the TF-IDF vectorizer on all resumes in the database.
        
        This should be called when you want to (re)train the vectorizer
        on the entire corpus.
        
        Returns:
            Statistics about the fitted model.
        """
        resumes = self.fetch_all_resumes()
        
        if not resumes:
            logger.warning("No resumes found in database to fit on")
            return {"status": "error", "message": "No resumes in database"}
        
        # Extract focused sections for each resume
        texts = []
        for r in resumes:
            if r.get('raw_text'):
                extracted_data = r.get('extracted_data', {})
                focused_text = self.extract_sections_for_embedding(r['raw_text'], extracted_data)
                texts.append(focused_text)
        
        logger.info(f"Fitting TF-IDF on {len(texts)} resume texts (focused on skills, projects, experience)")
        
        self.embedder.fit(texts)
        
        # Save the fitted model
        self.embedder.save(self.DEFAULT_MODEL_PATH)
        
        return self.embedder.get_stats()
    
    def load_model(self) -> bool:
        """
        Load a previously fitted model.
        
        Returns:
            True if loaded successfully, False otherwise.
        """
        if self.DEFAULT_MODEL_PATH.exists():
            self.embedder = TFIDFEmbedder.load(self.DEFAULT_MODEL_PATH)
            return True
        return False
    
    def generate_embedding(self, raw_text: str, extracted_data: Dict[str, Any] = None) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            raw_text: Raw text to embed.
            extracted_data: Parsed structured data (optional).
            
        Returns:
            Embedding vector as list of floats.
        """
        if not self.embedder.is_fitted:
            if not self.load_model():
                raise ValueError("No fitted model available. Call fit_on_all_resumes() first.")
        
        # Extract focused sections
        focused_text = self.extract_sections_for_embedding(raw_text, extracted_data or {})
        return self.embedder.embed_single(focused_text)
    
    def embed_resume(self, resume_id: str) -> Dict[str, Any]:
        """
        Generate and store embedding for a specific resume.
        
        Args:
            resume_id: UUID of the resume to embed.
            
        Returns:
            Result with embedding info.
        """
        # Fetch the resume
        result = self.db.table('resumes')\
            .select('id, raw_text, filename, extracted_data')\
            .eq('id', resume_id)\
            .execute()
        
        if not result.data:
            return {"status": "error", "message": f"Resume {resume_id} not found"}
        
        resume = result.data[0]
        extracted_data = resume.get('extracted_data', {})
        
        # Extract focused sections for embedding
        focused_text = self.extract_sections_for_embedding(resume['raw_text'], extracted_data)
        
        # Generate embedding
        embedding = self.embedder.embed_single(focused_text)
        
        # Store in database
        update_result = self.db.table('resumes')\
            .update({
                'embedding_vector': embedding,
                'embedding_model': self.embedder.model_name
            })\
            .eq('id', resume_id)\
            .execute()
        
        logger.success(f"Embedded resume: {resume['filename']}")
        
        return {
            "status": "success",
            "resume_id": resume_id,
            "filename": resume['filename'],
            "embedding_model": self.embedder.model_name,
            "vector_dimension": len(embedding),
            "non_zero_features": sum(1 for v in embedding if v > 0),
            "sections_embedded": ["skills", "projects", "experience", "education", "certifications"]
        }
    
    def embed_all_resumes(self, force: bool = False) -> Dict[str, Any]:
        """
        Generate and store embeddings for all resumes.
        
        Args:
            force: If True, re-embed all resumes. If False, only embed
                   resumes without embeddings.
                   
        Returns:
            Summary of embedding results.
        """
        if force:
            resumes = self.fetch_all_resumes()
        else:
            resumes = self.fetch_resumes_without_embeddings()
        
        if not resumes:
            logger.info("No resumes to embed")
            return {"status": "success", "embedded": 0, "message": "No resumes to embed"}
        
        # Ensure model is fitted
        if not self.embedder.is_fitted:
            if not self.load_model():
                logger.info("Fitting model on all resumes first...")
                self.fit_on_all_resumes()
        
        results = []
        for resume in resumes:
            try:
                result = self.embed_resume(resume['id'])
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to embed {resume['filename']}: {e}")
                results.append({
                    "status": "error",
                    "resume_id": resume['id'],
                    "error": str(e)
                })
        
        successful = sum(1 for r in results if r.get('status') == 'success')
        
        return {
            "status": "success",
            "total": len(resumes),
            "embedded": successful,
            "failed": len(resumes) - successful,
            "model": self.embedder.model_name,
            "results": results
        }
    
    def get_resume_embedding(self, resume_id: str) -> Optional[List[float]]:
        """
        Retrieve the stored embedding for a resume.
        
        Args:
            resume_id: UUID of the resume.
            
        Returns:
            Embedding vector or None if not found.
        """
        result = self.db.table('resumes')\
            .select('embedding_vector')\
            .eq('id', resume_id)\
            .execute()
        
        if result.data and result.data[0].get('embedding_vector'):
            return result.data[0]['embedding_vector']
        return None
    
    def get_top_terms(self, resume_id: str, top_n: int = 20) -> List[tuple]:
        """
        Get the top TF-IDF terms for a resume.
        
        Args:
            resume_id: UUID of the resume.
            top_n: Number of top terms to return.
            
        Returns:
            List of (term, weight) tuples.
        """
        embedding = self.get_resume_embedding(resume_id)
        
        if not embedding:
            return []
        
        import numpy as np
        return self.embedder.get_top_features(np.array(embedding), top_n)

    # =========================================================================
    # SECTION-WISE EMBEDDING METHODS
    # =========================================================================
    
    def extract_sections_dict(
        self,
        raw_text: str,
        extracted_data: Dict[str, Any] = None
    ) -> Dict[str, str]:
        """
        Extract resume sections as a dictionary.
        
        Args:
            raw_text: Full raw text from resume.
            extracted_data: Parsed structured data.
            
        Returns:
            Dictionary of section_name -> section_text.
        """
        sections = {}
        
        # Preprocess the text first
        text = self.preprocess_text(raw_text)
        
        # Extract Skills
        skills_text = self._extract_section(text, ["SKILLS", "TECHNICAL SKILLS", "CORE SKILLS"])
        if skills_text:
            sections["skills"] = skills_text
        elif extracted_data and extracted_data.get('skills'):
            sections["skills"] = " ".join(extracted_data['skills'])
        
        # Extract Projects
        projects_text = self._extract_section(text, ["PROJECTS", "PROJECT", "PERSONAL PROJECTS"])
        if projects_text:
            sections["projects"] = projects_text
        
        # Extract Experience
        experience_text = self._extract_section(text, ["EXPERIENCE", "WORK EXPERIENCE", "INTERNSHIP"])
        if experience_text:
            sections["experience"] = experience_text
        
        # Extract Education
        education_text = self._extract_section(text, ["EDUCATION", "ACADEMIC", "QUALIFICATION"])
        if education_text:
            sections["education"] = education_text
        
        # Extract Certifications
        certs_text = self._extract_section(text, ["CERTIFICATES", "CERTIFICATIONS"])
        if certs_text:
            sections["certifications"] = certs_text
        
        return sections
    
    def embed_sections_separately(
        self,
        raw_text: str,
        extracted_data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Generate separate embeddings for each resume section.
        
        Args:
            raw_text: Full raw text from resume.
            extracted_data: Parsed structured data.
            
        Returns:
            Dictionary with section embeddings and combined embeddings.
        """
        self._init_bert()
        
        # Extract sections
        sections = self.extract_sections_dict(raw_text, extracted_data)
        
        if not sections:
            logger.warning("No sections extracted, using full text")
            sections = {"full_text": self.preprocess_text(raw_text)}
        
        # Create section-wise embedder
        section_embedder = SectionWiseEmbedder(self.bert_embedder)
        
        # Generate section-wise embeddings
        result = section_embedder.embed_resume_sections(sections)
        
        logger.info(f"Generated embeddings for {len(result['embeddings'])} sections")
        
        return {
            "sections": list(result["embeddings"].keys()),
            "section_embeddings": {
                k: v.tolist() for k, v in result["embeddings"].items()
            },
            "combined_embedding": result["weighted_combined"].tolist() if result["weighted_combined"] is not None else None,
            "embedding_dim": self.bert_embedder.get_embedding_dimension()
        }
    
    # =========================================================================
    # HYBRID EMBEDDING METHODS (BERT + TF-IDF)
    # =========================================================================
    
    def generate_hybrid_embedding(
        self,
        raw_text: str,
        extracted_data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Generate hybrid BERT + TF-IDF embedding for a resume.
        
        Args:
            raw_text: Full raw text from resume.
            extracted_data: Parsed structured data.
            
        Returns:
            Dictionary with bert, tfidf, and combined embeddings.
        """
        self._init_hybrid()
        
        # Extract focused sections
        focused_text = self.extract_sections_for_embedding(raw_text, extracted_data or {})
        
        # Preprocess
        processed_text = self.preprocess_text(focused_text)
        
        # Load TF-IDF model if available
        if self.DEFAULT_MODEL_PATH.exists() and not self.hybrid_embedder._tfidf_initialized:
            self.hybrid_embedder.load_tfidf(str(self.DEFAULT_MODEL_PATH))
        
        # Generate hybrid embeddings
        embeddings = self.hybrid_embedder.embed(
            processed_text,
            bert_model=self.bert_model_name
        )
        
        return {
            "bert_embedding": embeddings["bert"][0].tolist(),
            "tfidf_embedding": embeddings["tfidf"][0].tolist(),
            "combined_embedding": embeddings["combined"][0].tolist(),
            "bert_dim": len(embeddings["bert"][0]),
            "tfidf_dim": len(embeddings["tfidf"][0]),
            "combined_dim": len(embeddings["combined"][0]),
            "bert_weight": self.bert_weight,
            "tfidf_weight": self.tfidf_weight
        }
    
    def embed_resume_hybrid(self, resume_id: str) -> Dict[str, Any]:
        """
        Generate and store hybrid embedding for a specific resume.
        
        Args:
            resume_id: UUID of the resume to embed.
            
        Returns:
            Result with embedding info.
        """
        # Fetch the resume
        result = self.db.table('resumes')\
            .select('id, raw_text, filename, extracted_data')\
            .eq('id', resume_id)\
            .execute()
        
        if not result.data:
            return {"status": "error", "message": f"Resume {resume_id} not found"}
        
        resume = result.data[0]
        extracted_data = resume.get('extracted_data', {})
        
        # Generate hybrid embedding
        hybrid_result = self.generate_hybrid_embedding(resume['raw_text'], extracted_data)
        
        # Store combined embedding in database
        update_result = self.db.table('resumes')\
            .update({
                'embedding_vector': hybrid_result['combined_embedding'],
                'embedding_model': f"hybrid-bert-tfidf-{self.bert_model_name}"
            })\
            .eq('id', resume_id)\
            .execute()
        
        logger.success(f"Generated hybrid embedding for: {resume['filename']}")
        
        return {
            "status": "success",
            "resume_id": resume_id,
            "filename": resume['filename'],
            "embedding_type": "hybrid",
            "bert_model": self.bert_model_name,
            "bert_dim": hybrid_result['bert_dim'],
            "tfidf_dim": hybrid_result['tfidf_dim'],
            "combined_dim": hybrid_result['combined_dim'],
            "bert_weight": self.bert_weight,
            "tfidf_weight": self.tfidf_weight
        }
    
    def embed_resume_section_wise(self, resume_id: str) -> Dict[str, Any]:
        """
        Generate and store section-wise embeddings for a specific resume.
        
        Args:
            resume_id: UUID of the resume to embed.
            
        Returns:
            Result with embedding info.
        """
        # Fetch the resume
        result = self.db.table('resumes')\
            .select('id, raw_text, filename, extracted_data')\
            .eq('id', resume_id)\
            .execute()
        
        if not result.data:
            return {"status": "error", "message": f"Resume {resume_id} not found"}
        
        resume = result.data[0]
        extracted_data = resume.get('extracted_data', {})
        
        # Generate section-wise embeddings
        section_result = self.embed_sections_separately(resume['raw_text'], extracted_data)
        
        # Store combined embedding in database
        if section_result['combined_embedding']:
            update_result = self.db.table('resumes')\
                .update({
                    'embedding_vector': section_result['combined_embedding'],
                    'embedding_model': f"section-wise-bert-{self.bert_model_name}"
                })\
                .eq('id', resume_id)\
                .execute()
        
        logger.success(f"Generated section-wise embeddings for: {resume['filename']}")
        
        return {
            "status": "success",
            "resume_id": resume_id,
            "filename": resume['filename'],
            "embedding_type": "section-wise",
            "sections_embedded": section_result['sections'],
            "embedding_dim": section_result['embedding_dim']
        }
    
    def save_hybrid_model(self):
        """Save the hybrid embedder to disk."""
        self._init_hybrid()
        self.hybrid_embedder.save(str(self.DEFAULT_HYBRID_PATH))
        logger.success(f"Saved hybrid model to {self.DEFAULT_HYBRID_PATH}")
    
    def load_hybrid_model(self) -> bool:
        """
        Load a previously saved hybrid model.
        
        Returns:
            True if loaded successfully, False otherwise.
        """
        hybrid_config_path = self.DEFAULT_HYBRID_PATH / "hybrid_config.pkl"
        if hybrid_config_path.exists():
            self.hybrid_embedder = HybridEmbedder.load(str(self.DEFAULT_HYBRID_PATH))
            return True
        return False
