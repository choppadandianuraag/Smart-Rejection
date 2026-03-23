"""
Embeddings package for Workflow 1.

Modules:
- TFIDFEmbedder: Sparse TF-IDF embeddings
- BertEmbedder: Dense BERT embeddings (V1)
- BERTEmbedder / SectionAwareEmbedder: Dense BERT embeddings (V2)
- HybridEmbedder: Combined BERT + TF-IDF
- ResumePreprocessor: Text cleaning and normalization
"""
from .tfidf_embedder import TFIDFEmbedder
from .bert_embedder import BertEmbedder, SectionWiseEmbedder
from .bert_embedder_v2 import BERTEmbedder, SectionAwareEmbedder, get_embedder, get_section_embedder
from .hybrid_embedder import HybridEmbedder
from .preprocessor import ResumePreprocessor, preprocess_resume

__all__ = [
    "TFIDFEmbedder",
    "BertEmbedder",
    "SectionWiseEmbedder",
    "BERTEmbedder",
    "SectionAwareEmbedder",
    "get_embedder",
    "get_section_embedder",
    "HybridEmbedder",
    "ResumePreprocessor",
    "preprocess_resume",
]
