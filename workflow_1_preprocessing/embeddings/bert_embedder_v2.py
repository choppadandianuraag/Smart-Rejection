"""
BERT Embedder Module
Pure BERT/Sentence-BERT embeddings for section-aware resume matching.
Uses all-mpnet-base-v2 model (768 dimensions).
"""

import os
from typing import List, Union, Optional
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from loguru import logger


class BERTEmbedder:
    """
    BERT embedder using Sentence-BERT for semantic embeddings.
    Generates 768-dimensional dense vectors using all-mpnet-base-v2.
    """
    
    # Model configuration
    DEFAULT_MODEL = "sentence-transformers/all-mpnet-base-v2"
    EMBEDDING_DIM = 768
    
    def __init__(
        self,
        model_name: str = None,
        device: str = None,
        cache_dir: str = None
    ):
        """
        Initialize BERT embedder.
        
        Args:
            model_name: HuggingFace model name (default: all-mpnet-base-v2)
            device: Device to use ('cuda', 'mps', 'cpu')
            cache_dir: Directory to cache downloaded models
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self.cache_dir = cache_dir or str(Path.home() / ".cache" / "sentence-transformers")
        
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        logger.info(f"Initializing BERT embedder with model: {self.model_name}")
        logger.info(f"Using device: {self.device}")
        
        # Load model
        try:
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device,
                cache_folder=self.cache_dir
            )
            logger.success(f"Model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def embed(
        self,
        text: Union[str, List[str]],
        normalize: bool = True,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Generate BERT embedding for text.
        
        Args:
            text: Single string or list of strings to embed
            normalize: Whether to L2-normalize embeddings (recommended for cosine similarity)
            show_progress: Show progress bar for batch embedding
            
        Returns:
            numpy array of shape (768,) for single text or (N, 768) for list
        """
        if isinstance(text, str):
            if not text.strip():
                logger.warning("Empty text provided, returning zero vector")
                return np.zeros(self.EMBEDDING_DIM, dtype=np.float32)
            
            texts = [text]
            single_input = True
        else:
            if not text or all(not t.strip() for t in text):
                logger.warning("All texts are empty, returning zero vectors")
                return np.zeros((len(text), self.EMBEDDING_DIM), dtype=np.float32)
            
            texts = text
            single_input = False
        
        try:
            # Generate embeddings
            embeddings = self.model.encode(
                texts,
                normalize_embeddings=normalize,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                batch_size=32
            )
            
            # Return single vector or batch
            if single_input:
                return embeddings[0]
            return embeddings
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise
    
    def embed_text(self, text: str, normalize: bool = True) -> np.ndarray:
        """
        Convenience method to embed a single text.

        Args:
            text: Text to embed
            normalize: Whether to L2-normalize embedding

        Returns:
            numpy array of shape (768,)
        """
        return self.embed(text, normalize=normalize, show_progress=False)

    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        normalize: bool = True,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Efficiently embed multiple texts in batches.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            normalize: Whether to normalize embeddings
            show_progress: Show progress bar
            
        Returns:
            numpy array of shape (N, 768)
        """
        if not texts:
            return np.zeros((0, self.EMBEDDING_DIM), dtype=np.float32)
        
        logger.info(f"Embedding {len(texts)} texts in batches of {batch_size}")
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        logger.success(f"Generated {len(embeddings)} embeddings")
        return embeddings
    
    def cosine_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector (768,)
            embedding2: Second embedding vector (768,)
            
        Returns:
            Cosine similarity score (0 to 1)
        """
        # Ensure both are 1D
        if embedding1.ndim > 1:
            embedding1 = embedding1.flatten()
        if embedding2.ndim > 1:
            embedding2 = embedding2.flatten()
        
        # Compute cosine similarity
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        
        # Clamp to [0, 1] (can be slightly outside due to numerical errors)
        return float(np.clip(similarity, 0.0, 1.0))
    
    def batch_cosine_similarity(
        self,
        embeddings1: np.ndarray,
        embeddings2: np.ndarray
    ) -> np.ndarray:
        """
        Compute pairwise cosine similarities between two sets of embeddings.
        
        Args:
            embeddings1: Array of shape (N, 768)
            embeddings2: Array of shape (M, 768)
            
        Returns:
            Similarity matrix of shape (N, M)
        """
        from sklearn.metrics.pairwise import cosine_similarity
        return cosine_similarity(embeddings1, embeddings2)
    
    @property
    def embedding_dimension(self) -> int:
        """Get embedding dimension."""
        return self.EMBEDDING_DIM
    
    def __repr__(self):
        return f"BERTEmbedder(model={self.model_name}, device={self.device}, dim={self.EMBEDDING_DIM})"


class SectionAwareEmbedder:
    """
    Higher-level embedder that handles section-aware embedding generation.
    Wraps BERTEmbedder with section-specific logic.
    """
    
    def __init__(self, model_name: str = None, device: str = None):
        """Initialize section-aware embedder."""
        self.embedder = BERTEmbedder(model_name=model_name, device=device)
        logger.info("Section-aware embedder initialized")
    
    def embed_sections(
        self,
        sections: List[dict],
        show_progress: bool = False
    ) -> List[dict]:
        """
        Embed multiple sections at once.
        
        Args:
            sections: List of section dicts with 'text' field
            show_progress: Show progress bar
            
        Returns:
            Same sections with added 'embedding' field
        """
        if not sections:
            return []
        
        # Extract texts
        texts = [s['text'] for s in sections]
        
        # Generate embeddings in batch
        embeddings = self.embedder.embed_batch(
            texts,
            show_progress=show_progress
        )
        
        # Add embeddings to sections
        for section, embedding in zip(sections, embeddings):
            section['embedding'] = embedding
        
        return sections
    
    def embed_section(self, section_text: str) -> np.ndarray:
        """Embed a single section."""
        return self.embedder.embed(section_text)

    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a single text (alias for embed_section).

        Args:
            text: Text to embed

        Returns:
            numpy array of shape (768,)
        """
        return self.embedder.embed(text)

    def compute_section_similarity(
        self,
        candidate_section: dict,
        job_section: dict
    ) -> float:
        """
        Compute similarity between a candidate section and job section.
        
        Args:
            candidate_section: Dict with 'embedding' key
            job_section: Dict with 'embedding' key
            
        Returns:
            Cosine similarity score (0 to 1)
        """
        return self.embedder.cosine_similarity(
            candidate_section['embedding'],
            job_section['embedding']
        )
    
    def compute_weighted_match_score(
        self,
        candidate_sections: List[dict],
        job_sections: List[dict],
        weights: dict
    ) -> dict:
        """
        Compute weighted match score between candidate and job.
        
        Args:
            candidate_sections: List of candidate section dicts with 'section_type' and 'embedding'
            job_sections: List of job section dicts with 'section_type' and 'embedding'
            weights: Dict mapping section pairs to weights
                    e.g., {"skills_to_requirements": 0.40, ...}
        
        Returns:
            Dict with overall score and section-wise scores
        """
        # Map sections by type
        candidate_map = {s['section_type']: s for s in candidate_sections}
        job_map = {s['section_type']: s for s in job_sections}
        
        # Define section pair mappings
        SECTION_MAPPINGS = [
            ('skills', 'requirements', 'skills_to_requirements'),
            ('work_experience', 'responsibilities', 'experience_to_responsibilities'),
            ('education', 'qualifications', 'education_to_qualifications'),
            ('summary', 'overview', 'summary_to_overview'),
        ]
        
        section_scores = {}
        applicable_weights = []
        weighted_scores = []
        
        for candidate_type, job_type, weight_key in SECTION_MAPPINGS:
            if candidate_type in candidate_map and job_type in job_map:
                # Compute similarity
                similarity = self.embedder.cosine_similarity(
                    candidate_map[candidate_type]['embedding'],
                    job_map[job_type]['embedding']
                )
                
                section_scores[weight_key] = similarity
                
                # Apply weight if available
                if weight_key in weights:
                    applicable_weights.append(weights[weight_key])
                    weighted_scores.append(similarity * weights[weight_key])
        
        # Handle missing sections: redistribute weights
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


# Singleton instance
_embedder_instance = None


def get_embedder(model_name: str = None, device: str = None) -> BERTEmbedder:
    """
    Get or create singleton BERTEmbedder instance.
    Avoids reloading model multiple times.
    """
    global _embedder_instance
    
    if _embedder_instance is None:
        _embedder_instance = BERTEmbedder(model_name=model_name, device=device)
    
    return _embedder_instance


def get_section_embedder(model_name: str = None, device: str = None) -> SectionAwareEmbedder:
    """Get or create singleton SectionAwareEmbedder instance."""
    return SectionAwareEmbedder(model_name=model_name, device=device)
