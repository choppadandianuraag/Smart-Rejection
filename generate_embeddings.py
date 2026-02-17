"""
Generate embeddings for resumes in the database.
Phase 2: Separate from Phase 1 extraction.

Usage:
    python generate_embeddings.py              # Embed resumes without embeddings
    python generate_embeddings.py --force      # Re-embed all resumes
    python generate_embeddings.py --view       # View embedding stats
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger
from embeddings.embedding_service import EmbeddingService


def main():
    parser = argparse.ArgumentParser(
        description="Generate TF-IDF embeddings for resumes (Phase 2)"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Re-embed all resumes, even those with existing embeddings"
    )
    parser.add_argument(
        "--view", "-v",
        action="store_true",
        help="View embedding statistics for stored resumes"
    )
    parser.add_argument(
        "--ngram-min",
        type=int,
        default=1,
        help="Minimum n-gram size (default: 1)"
    )
    parser.add_argument(
        "--ngram-max",
        type=int,
        default=3,
        help="Maximum n-gram size (default: 3)"
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=5000,
        help="Maximum vocabulary size (default: 5000)"
    )
    
    args = parser.parse_args()
    
    # Initialize embedding service
    service = EmbeddingService(
        ngram_range=(args.ngram_min, args.ngram_max),
        max_features=args.max_features
    )
    
    if args.view:
        view_embedding_stats(service)
        return
    
    print("=" * 60)
    print("   PHASE 2: TF-IDF EMBEDDING GENERATION")
    print("=" * 60)
    print()
    print(f"Configuration:")
    print(f"  N-gram range: ({args.ngram_min}, {args.ngram_max})")
    print(f"  Max features: {args.max_features}")
    print(f"  Force re-embed: {args.force}")
    print()
    
    # Step 1: Fit the vectorizer on all resumes
    print("-" * 60)
    print("Step 1: Fitting TF-IDF vectorizer on corpus...")
    print("-" * 60)
    
    fit_stats = service.fit_on_all_resumes()
    
    if fit_stats.get("status") == "error":
        print(f"Error: {fit_stats.get('message')}")
        return
    
    print(f"  Vocabulary size: {fit_stats.get('vocabulary_size')}")
    print(f"  Model name: {fit_stats.get('model_name')}")
    print()
    
    # Step 2: Generate embeddings for all resumes
    print("-" * 60)
    print("Step 2: Generating embeddings for resumes...")
    print("-" * 60)
    
    embed_results = service.embed_all_resumes(force=args.force)
    
    print(f"  Total resumes: {embed_results.get('total')}")
    print(f"  Embedded: {embed_results.get('embedded')}")
    print(f"  Failed: {embed_results.get('failed')}")
    print()
    
    # Step 3: Show results
    print("-" * 60)
    print("Step 3: Results")
    print("-" * 60)
    
    for result in embed_results.get('results', []):
        if result.get('status') == 'success':
            print(f"  ✅ {result.get('filename')}")
            print(f"     Vector dimension: {result.get('vector_dimension')}")
            print(f"     Non-zero features: {result.get('non_zero_features')}")
        else:
            print(f"  ❌ {result.get('resume_id')}: {result.get('error')}")
    
    print()
    print("=" * 60)
    print("   EMBEDDING GENERATION COMPLETE")
    print("=" * 60)


def view_embedding_stats(service: EmbeddingService):
    """View embedding statistics for stored resumes."""
    print("=" * 60)
    print("   EMBEDDING STATISTICS")
    print("=" * 60)
    print()
    
    # Fetch all resumes with embedding info
    result = service.db.table('resumes')\
        .select('id, filename, embedding_vector, embedding_model')\
        .execute()
    
    if not result.data:
        print("No resumes found in database.")
        return
    
    total = len(result.data)
    with_embeddings = sum(1 for r in result.data if r.get('embedding_vector'))
    
    print(f"Total resumes: {total}")
    print(f"With embeddings: {with_embeddings}")
    print(f"Without embeddings: {total - with_embeddings}")
    print()
    
    print("-" * 60)
    print("Resume Details:")
    print("-" * 60)
    
    for resume in result.data:
        has_embedding = bool(resume.get('embedding_vector'))
        status = "✅" if has_embedding else "❌"
        model = resume.get('embedding_model') or "None"
        
        vector_info = ""
        if has_embedding:
            vec = resume['embedding_vector']
            non_zero = sum(1 for v in vec if v > 0)
            vector_info = f" (dim={len(vec)}, non-zero={non_zero})"
        
        print(f"  {status} {resume['filename']}")
        print(f"     Model: {model}{vector_info}")
    
    # Show top terms if model is loaded
    if with_embeddings > 0 and service.load_model():
        print()
        print("-" * 60)
        print("Top TF-IDF Terms per Resume:")
        print("-" * 60)
        
        for resume in result.data:
            if resume.get('embedding_vector'):
                print(f"\n  📄 {resume['filename']}:")
                top_terms = service.get_top_terms(resume['id'], top_n=15)
                for term, weight in top_terms:
                    print(f"     • {term}: {weight:.4f}")


if __name__ == "__main__":
    main()
