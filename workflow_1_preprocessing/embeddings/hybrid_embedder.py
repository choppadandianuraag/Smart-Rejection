"""
Hybrid Embedder combining BERT (dense) and TF-IDF (sparse) representations.
Provides better retrieval performance by leveraging both semantic and lexical matching.
"""

import os
import pickle
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path

import numpy as np
from loguru import logger
from scipy import sparse


class HybridEmbedder:
    """
    Combines dense (BERT) and sparse (TF-IDF) embeddings for hybrid retrieval.
    
    This approach leverages:
    - BERT: Semantic understanding, captures meaning beyond exact words
    - TF-IDF: Lexical matching, captures exact keyword matches
    """
    
    def __init__(
        self,
        bert_weight: float = 0.7,
        tfidf_weight: float = 0.3,
        normalize: bool = True
    ):
        """
        Initialize the hybrid embedder.
        
        Args:
            bert_weight: Weight for BERT embeddings (0-1).
            tfidf_weight: Weight for TF-IDF embeddings (0-1).
            normalize: Whether to normalize embeddings before combining.
        """
        self.bert_weight = bert_weight
        self.tfidf_weight = tfidf_weight
        self.normalize = normalize
        
        # These will be initialized lazily
        self.bert_embedder = None
        self.tfidf_embedder = None
        
        self._bert_initialized = False
        self._tfidf_initialized = False
    
    def _init_bert(self, model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None):
        """Initialize BERT embedder."""
        if self._bert_initialized:
            return
        
        from .bert_embedder import BertEmbedder
        self.bert_embedder = BertEmbedder(model_name=model_name, device=device)
        self._bert_initialized = True
    
    def _init_tfidf(self, **kwargs):
        """Initialize TF-IDF embedder."""
        if self._tfidf_initialized:
            return
        
        from .tfidf_embedder import TFIDFEmbedder
        self.tfidf_embedder = TFIDFEmbedder(**kwargs)
        self._tfidf_initialized = True
    
    def load_tfidf(self, model_path: str):
        """Load a pre-fitted TF-IDF model."""
        from .tfidf_embedder import TFIDFEmbedder
        self.tfidf_embedder = TFIDFEmbedder.load(model_path)
        self._tfidf_initialized = True
    
    def fit_tfidf(self, corpus: List[str]):
        """Fit TF-IDF on a corpus."""
        if not self._tfidf_initialized:
            self._init_tfidf()
        self.tfidf_embedder.fit(corpus)
    
    def embed(
        self,
        texts: Union[str, List[str]],
        return_separate: bool = False,
        bert_model: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None
    ) -> Union[Dict[str, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Generate hybrid embeddings for text(s).
        
        Args:
            texts: Single text or list of texts.
            return_separate: If True, return separate embeddings instead of combined.
            bert_model: BERT model to use.
            device: Device for BERT.
            
        Returns:
            Dictionary with 'bert', 'tfidf', and 'combined' embeddings,
            or tuple of (bert, tfidf, combined) if return_separate is True.
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Initialize embedders
        self._init_bert(model_name=bert_model, device=device)
        if not self._tfidf_initialized:
            self._init_tfidf()
            self.tfidf_embedder.fit(texts)
        
        # Generate BERT embeddings
        logger.info("Generating BERT embeddings...")
        bert_embeddings = self.bert_embedder.embed(texts)
        
        # Generate TF-IDF embeddings
        logger.info("Generating TF-IDF embeddings...")
        tfidf_embeddings = self.tfidf_embedder.transform(texts)
        
        # Convert sparse to dense if needed
        if sparse.issparse(tfidf_embeddings):
            tfidf_embeddings = tfidf_embeddings.toarray()
        
        # Normalize if requested
        if self.normalize:
            bert_embeddings = self._normalize(bert_embeddings)
            tfidf_embeddings = self._normalize(tfidf_embeddings)
        
        # Combine embeddings
        combined = self._combine_embeddings(bert_embeddings, tfidf_embeddings)
        
        if return_separate:
            return bert_embeddings, tfidf_embeddings, combined
        
        return {
            "bert": bert_embeddings,
            "tfidf": tfidf_embeddings,
            "combined": combined
        }
    
    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        """L2 normalize embeddings."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
        return embeddings / norms
    
    def _combine_embeddings(
        self,
        bert_emb: np.ndarray,
        tfidf_emb: np.ndarray
    ) -> np.ndarray:
        """
        Combine BERT and TF-IDF embeddings.
        
        Uses weighted concatenation for combined representation.
        """
        # Scale by weights
        bert_scaled = bert_emb * self.bert_weight
        tfidf_scaled = tfidf_emb * self.tfidf_weight
        
        # Concatenate
        combined = np.hstack([bert_scaled, tfidf_scaled])
        
        # Normalize the combined embedding
        if self.normalize:
            combined = self._normalize(combined)
        
        return combined
    
    def compute_similarity(
        self,
        query_embedding: Dict[str, np.ndarray],
        candidate_embeddings: List[Dict[str, np.ndarray]],
        method: str = "combined"
    ) -> np.ndarray:
        """
        Compute similarity between query and candidates.
        
        Args:
            query_embedding: Query embedding dict with 'bert', 'tfidf', 'combined'.
            candidate_embeddings: List of candidate embedding dicts.
            method: 'combined', 'bert', 'tfidf', or 'weighted_sum'.
            
        Returns:
            Array of similarity scores.
        """
        from sklearn.metrics.pairwise import cosine_similarity
        
        if method == "combined":
            query = query_embedding["combined"].reshape(1, -1)
            candidates = np.vstack([c["combined"] for c in candidate_embeddings])
            return cosine_similarity(query, candidates)[0]
        
        elif method == "bert":
            query = query_embedding["bert"].reshape(1, -1)
            candidates = np.vstack([c["bert"] for c in candidate_embeddings])
            return cosine_similarity(query, candidates)[0]
        
        elif method == "tfidf":
            query = query_embedding["tfidf"].reshape(1, -1)
            candidates = np.vstack([c["tfidf"] for c in candidate_embeddings])
            return cosine_similarity(query, candidates)[0]
        
        elif method == "weighted_sum":
            # Compute separate similarities and combine
            bert_query = query_embedding["bert"].reshape(1, -1)
            bert_candidates = np.vstack([c["bert"] for c in candidate_embeddings])
            bert_sim = cosine_similarity(bert_query, bert_candidates)[0]
            
            tfidf_query = query_embedding["tfidf"].reshape(1, -1)
            tfidf_candidates = np.vstack([c["tfidf"] for c in candidate_embeddings])
            tfidf_sim = cosine_similarity(tfidf_query, tfidf_candidates)[0]
            
            return self.bert_weight * bert_sim + self.tfidf_weight * tfidf_sim
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def save(self, save_dir: str):
        """Save the hybrid embedder configuration and models."""
        os.makedirs(save_dir, exist_ok=True)
        
        config = {
            "bert_weight": self.bert_weight,
            "tfidf_weight": self.tfidf_weight,
            "normalize": self.normalize
        }
        
        config_path = os.path.join(save_dir, "hybrid_config.pkl")
        with open(config_path, 'wb') as f:
            pickle.dump(config, f)
        
        # Save BERT config
        if self._bert_initialized:
            self.bert_embedder.save(save_dir)
        
        # Save TF-IDF model
        if self._tfidf_initialized:
            tfidf_path = os.path.join(save_dir, "tfidf_vectorizer.pkl")
            self.tfidf_embedder.save(tfidf_path)
        
        logger.info(f"Saved hybrid embedder to {save_dir}")
    
    @classmethod
    def load(cls, save_dir: str) -> "HybridEmbedder":
        """Load the hybrid embedder from disk."""
        config_path = os.path.join(save_dir, "hybrid_config.pkl")
        
        with open(config_path, 'rb') as f:
            config = pickle.load(f)
        
        embedder = cls(
            bert_weight=config["bert_weight"],
            tfidf_weight=config["tfidf_weight"],
            normalize=config.get("normalize", True)
        )
        
        # Load BERT config if exists
        bert_config_path = os.path.join(save_dir, "bert_config.pkl")
        if os.path.exists(bert_config_path):
            from .bert_embedder import BertEmbedder
            embedder.bert_embedder = BertEmbedder.load(save_dir)
            embedder._bert_initialized = True
        
        # Load TF-IDF model if exists
        tfidf_path = os.path.join(save_dir, "tfidf_vectorizer.pkl")
        if os.path.exists(tfidf_path):
            from .tfidf_embedder import TFIDFEmbedder
            embedder.tfidf_embedder = TFIDFEmbedder.load(tfidf_path)
            embedder._tfidf_initialized = True
        
        logger.info(f"Loaded hybrid embedder from {save_dir}")
        return embedder


def create_hybrid_embedding(
    text: str,
    bert_model: str = "all-MiniLM-L6-v2",
    bert_weight: float = 0.7,
    tfidf_weight: float = 0.3
) -> Dict[str, np.ndarray]:
    """
    Convenience function to create hybrid embeddings for a single text.
    
    Args:
        text: Text to embed.
        bert_model: BERT model to use.
        bert_weight: Weight for BERT embeddings.
        tfidf_weight: Weight for TF-IDF embeddings.
        
    Returns:
        Dictionary with 'bert', 'tfidf', and 'combined' embeddings.
    """
    embedder = HybridEmbedder(
        bert_weight=bert_weight,
        tfidf_weight=tfidf_weight
    )
    return embedder.embed(text, bert_model=bert_model)
