"""
TF-IDF Embedding Generator for Resume Data.
Phase 2: Generates embeddings using TF-IDF with n-grams.
"""

import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from loguru import logger


class TFIDFEmbedder:
    """
    TF-IDF based text embedder with n-gram support.
    
    Generates sparse vector embeddings from text using TF-IDF
    with configurable n-gram ranges.
    """
    
    def __init__(
        self,
        ngram_range: Tuple[int, int] = (1, 3),
        max_features: int = 5000,
        min_df: int = 1,
        max_df: float = 1.0,
        sublinear_tf: bool = True
    ):
        """
        Initialize the TF-IDF embedder.
        
        Args:
            ngram_range: Range of n-grams (min_n, max_n). Default (1,3) uses
                         unigrams, bigrams, and trigrams.
            max_features: Maximum number of features (vocabulary size).
            min_df: Minimum document frequency for terms.
            max_df: Maximum document frequency (as proportion) for terms.
            sublinear_tf: Apply sublinear tf scaling (1 + log(tf)).
        """
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.sublinear_tf = sublinear_tf
        
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.is_fitted = False
        
        self._model_name = f"tfidf-ngram{ngram_range[0]}-{ngram_range[1]}-{max_features}feat"
    
    @property
    def model_name(self) -> str:
        """Get the model identifier string."""
        return self._model_name
    
    def fit(self, texts: List[str]) -> 'TFIDFEmbedder':
        """
        Fit the TF-IDF vectorizer on a corpus of texts.
        
        Args:
            texts: List of text documents to fit on.
            
        Returns:
            Self for method chaining.
        """
        logger.info(f"Fitting TF-IDF vectorizer on {len(texts)} documents")
        logger.info(f"Config: ngram_range={self.ngram_range}, max_features={self.max_features}")
        
        self.vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            sublinear_tf=self.sublinear_tf,
            lowercase=True,
            strip_accents='unicode',
            stop_words='english',
            dtype=np.float64
        )
        
        self.vectorizer.fit(texts)
        self.is_fitted = True
        
        vocab_size = len(self.vectorizer.vocabulary_)
        logger.success(f"TF-IDF vectorizer fitted. Vocabulary size: {vocab_size}")
        
        return self
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts to TF-IDF vectors.
        
        Args:
            texts: List of text documents to transform.
            
        Returns:
            Dense numpy array of shape (n_documents, n_features).
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer not fitted. Call fit() first.")
        
        sparse_matrix = self.vectorizer.transform(texts)
        
        # Convert to dense array for storage
        dense_vectors = sparse_matrix.toarray()
        
        logger.info(f"Transformed {len(texts)} documents to vectors of shape {dense_vectors.shape}")
        
        return dense_vectors
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        Fit and transform in one step.
        
        Args:
            texts: List of text documents.
            
        Returns:
            Dense numpy array of TF-IDF vectors.
        """
        self.fit(texts)
        return self.transform(texts)
    
    def embed_single(self, text: str) -> List[float]:
        """
        Embed a single text document.
        
        Args:
            text: Text to embed.
            
        Returns:
            List of floats representing the TF-IDF vector.
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer not fitted. Call fit() first.")
        
        vector = self.vectorizer.transform([text]).toarray()[0]
        return vector.tolist()
    
    def get_feature_names(self) -> List[str]:
        """Get the feature names (vocabulary terms)."""
        if not self.is_fitted:
            return []
        return self.vectorizer.get_feature_names_out().tolist()
    
    def get_top_features(self, vector: np.ndarray, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Get the top N features by TF-IDF weight for a vector.
        
        Args:
            vector: TF-IDF vector.
            top_n: Number of top features to return.
            
        Returns:
            List of (feature_name, weight) tuples.
        """
        if not self.is_fitted:
            return []
        
        feature_names = self.get_feature_names()
        top_indices = np.argsort(vector)[-top_n:][::-1]
        
        return [
            (feature_names[i], float(vector[i]))
            for i in top_indices
            if vector[i] > 0
        ]
    
    def save(self, filepath: str | Path) -> None:
        """
        Save the fitted vectorizer to disk.
        
        Args:
            filepath: Path to save the vectorizer.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'config': {
                    'ngram_range': self.ngram_range,
                    'max_features': self.max_features,
                    'min_df': self.min_df,
                    'max_df': self.max_df,
                    'sublinear_tf': self.sublinear_tf
                },
                'is_fitted': self.is_fitted,
                'model_name': self._model_name
            }, f)
        
        logger.info(f"Saved TF-IDF embedder to {filepath}")
    
    @classmethod
    def load(cls, filepath: str | Path) -> 'TFIDFEmbedder':
        """
        Load a fitted vectorizer from disk.
        
        Args:
            filepath: Path to the saved vectorizer.
            
        Returns:
            Loaded TFIDFEmbedder instance.
        """
        filepath = Path(filepath)
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        embedder = cls(**data['config'])
        embedder.vectorizer = data['vectorizer']
        embedder.is_fitted = data['is_fitted']
        embedder._model_name = data['model_name']
        
        logger.info(f"Loaded TF-IDF embedder from {filepath}")
        
        return embedder
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the fitted vectorizer."""
        if not self.is_fitted:
            return {"status": "not fitted"}
        
        return {
            "status": "fitted",
            "model_name": self._model_name,
            "vocabulary_size": len(self.vectorizer.vocabulary_),
            "ngram_range": self.ngram_range,
            "max_features": self.max_features,
            "feature_count": len(self.get_feature_names())
        }
