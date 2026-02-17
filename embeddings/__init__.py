"""
Embeddings package initialization (Phase 2).

Modules:
- TFIDFEmbedder: Sparse TF-IDF embeddings
- BertEmbedder: Dense BERT embeddings
- HybridEmbedder: Combined BERT + TF-IDF
- ResumePreprocessor: Text cleaning and normalization
- EmbeddingService: Main service orchestrating all embedding methods
"""
from embeddings.tfidf_embedder import TFIDFEmbedder
from embeddings.bert_embedder import BertEmbedder, SectionWiseEmbedder
from embeddings.hybrid_embedder import HybridEmbedder
from embeddings.preprocessor import ResumePreprocessor, preprocess_resume
from embeddings.embedding_service import EmbeddingService

__all__ = [
    "TFIDFEmbedder",
    "BertEmbedder",
    "SectionWiseEmbedder", 
    "HybridEmbedder",
    "ResumePreprocessor",
    "preprocess_resume",
    "EmbeddingService"
]
