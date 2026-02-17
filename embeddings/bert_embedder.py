"""
BERT-based Dense Embedder for Resume Data.
Uses sentence-transformers for generating dense embeddings.
"""

import os
import pickle
from typing import List, Dict, Optional, Union
from pathlib import Path

import numpy as np
from loguru import logger


class BertEmbedder:
    """
    Generates dense embeddings using sentence-transformers (BERT-based models).
    """
    
    # Default model - lightweight and efficient
    DEFAULT_MODEL = "all-MiniLM-L6-v2"
    
    # Alternative models for different use cases
    AVAILABLE_MODELS = {
        "all-MiniLM-L6-v2": {
            "description": "Fast, good for semantic similarity (384 dims)",
            "dimensions": 384
        },
        "all-mpnet-base-v2": {
            "description": "Higher quality, slower (768 dims)",
            "dimensions": 768
        },
        "paraphrase-MiniLM-L6-v2": {
            "description": "Optimized for paraphrase detection (384 dims)",
            "dimensions": 384
        },
        "multi-qa-MiniLM-L6-cos-v1": {
            "description": "Optimized for semantic search (384 dims)",
            "dimensions": 384
        }
    }
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None
    ):
        """
        Initialize the BERT embedder.
        
        Args:
            model_name: Name of the sentence-transformers model to use.
            device: Device to run on ('cpu', 'cuda', 'mps'). Auto-detected if None.
        """
        self.model_name = model_name
        self.model = None
        self.device = device
        
        # Lazy load the model
        self._initialized = False
    
    def _init_model(self):
        """Initialize the model (lazy loading)."""
        if self._initialized:
            return
        
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"Loading BERT model: {self.model_name}")
            
            # Auto-detect device if not specified
            if self.device is None:
                import torch
                if torch.cuda.is_available():
                    self.device = "cuda"
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    self.device = "mps"
                else:
                    self.device = "cpu"
            
            logger.info(f"Using device: {self.device}")
            
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self._initialized = True
            
            logger.info(f"Model loaded. Embedding dimensions: {self.model.get_sentence_embedding_dimension()}")
            
        except Exception as e:
            logger.error(f"Failed to load BERT model: {e}")
            raise
    
    def embed(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Generate dense embeddings for text(s).
        
        Args:
            texts: Single text or list of texts to embed.
            batch_size: Batch size for encoding.
            show_progress: Whether to show progress bar.
            
        Returns:
            Numpy array of embeddings. Shape: (n_texts, embedding_dim)
        """
        self._init_model()
        
        if isinstance(texts, str):
            texts = [texts]
        
        logger.info(f"Generating BERT embeddings for {len(texts)} text(s)")
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def embed_sections(
        self,
        sections: Dict[str, str],
        batch_size: int = 32
    ) -> Dict[str, np.ndarray]:
        """
        Generate embeddings for each section separately.
        
        Args:
            sections: Dictionary of section_name -> section_text.
            batch_size: Batch size for encoding.
            
        Returns:
            Dictionary of section_name -> embedding array.
        """
        self._init_model()
        
        section_embeddings = {}
        
        for section_name, section_text in sections.items():
            if section_text and section_text.strip():
                logger.debug(f"Embedding section: {section_name}")
                embedding = self.embed(section_text, batch_size=batch_size)
                section_embeddings[section_name] = embedding[0]  # Single text, get first
            else:
                logger.warning(f"Skipping empty section: {section_name}")
        
        return section_embeddings
    
    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension of the model."""
        self._init_model()
        return self.model.get_sentence_embedding_dimension()
    
    def save(self, save_dir: str):
        """
        Save the embedder configuration.
        
        Args:
            save_dir: Directory to save the configuration.
        """
        os.makedirs(save_dir, exist_ok=True)
        
        config = {
            "model_name": self.model_name,
            "device": self.device
        }
        
        config_path = os.path.join(save_dir, "bert_config.pkl")
        with open(config_path, 'wb') as f:
            pickle.dump(config, f)
        
        logger.info(f"Saved BERT embedder config to {config_path}")
    
    @classmethod
    def load(cls, save_dir: str) -> "BertEmbedder":
        """
        Load the embedder configuration.
        
        Args:
            save_dir: Directory containing the saved configuration.
            
        Returns:
            Loaded BertEmbedder instance.
        """
        config_path = os.path.join(save_dir, "bert_config.pkl")
        
        with open(config_path, 'rb') as f:
            config = pickle.load(f)
        
        logger.info(f"Loaded BERT embedder config from {config_path}")
        
        return cls(
            model_name=config["model_name"],
            device=config.get("device")
        )


class SectionWiseEmbedder:
    """
    Generates section-wise embeddings for resumes.
    Each section (skills, experience, education) gets its own embedding.
    """
    
    SECTION_WEIGHTS = {
        "skills": 1.5,
        "experience": 1.2,
        "education": 1.0,
        "projects": 1.3,
        "certifications": 0.8,
        "summary": 0.7
    }
    
    def __init__(self, bert_embedder: BertEmbedder):
        """
        Initialize the section-wise embedder.
        
        Args:
            bert_embedder: BertEmbedder instance to use.
        """
        self.bert_embedder = bert_embedder
    
    def embed_resume_sections(
        self,
        sections: Dict[str, str]
    ) -> Dict[str, Dict]:
        """
        Generate embeddings for each resume section.
        
        Args:
            sections: Dictionary of section_name -> section_text.
            
        Returns:
            Dictionary with section embeddings and metadata.
        """
        result = {
            "embeddings": {},
            "weighted_combined": None,
            "simple_combined": None
        }
        
        embeddings_list = []
        weights_list = []
        
        for section_name, section_text in sections.items():
            if not section_text or not section_text.strip():
                continue
            
            embedding = self.bert_embedder.embed(section_text)[0]
            result["embeddings"][section_name] = embedding
            
            # Get weight for this section
            weight = self.SECTION_WEIGHTS.get(section_name.lower(), 1.0)
            embeddings_list.append(embedding)
            weights_list.append(weight)
        
        if embeddings_list:
            # Simple average
            result["simple_combined"] = np.mean(embeddings_list, axis=0)
            
            # Weighted average
            weights_array = np.array(weights_list)
            weights_normalized = weights_array / weights_array.sum()
            result["weighted_combined"] = np.average(
                embeddings_list, 
                axis=0, 
                weights=weights_normalized
            )
        
        return result
    
    def compute_section_similarity(
        self,
        resume_sections: Dict[str, np.ndarray],
        job_sections: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """
        Compute similarity between resume and job sections.
        
        Args:
            resume_sections: Resume section embeddings.
            job_sections: Job description section embeddings.
            
        Returns:
            Dictionary of section -> similarity score.
        """
        from sklearn.metrics.pairwise import cosine_similarity
        
        similarities = {}
        
        for section_name in resume_sections:
            if section_name in job_sections:
                resume_emb = resume_sections[section_name].reshape(1, -1)
                job_emb = job_sections[section_name].reshape(1, -1)
                
                similarity = cosine_similarity(resume_emb, job_emb)[0][0]
                similarities[section_name] = float(similarity)
        
        return similarities
