"""
Resume Ranking Service
Rank resumes against a job description using hybrid embeddings.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from database.supabase_client import SupabaseClient
from embeddings.hybrid_embedder import HybridEmbedder


class ResumeRanker:
    """Rank resumes against a job description."""
    
    def __init__(self):
        self.db = SupabaseClient()
        self.embedder = HybridEmbedder()
        
    def generate_embeddings_for_all(self) -> int:
        """Generate embeddings for all resumes that don't have them."""
        # Get resumes without embeddings
        result = self.db._client.table("resumes").select("id, raw_text").is_("embedding_vector", "null").execute()
        
        if not result.data:
            print("All resumes already have embeddings!")
            return 0
        
        count = 0
        for resume in result.data:
            try:
                # Generate hybrid embedding
                embedding_dict = self.embedder.embed(resume["raw_text"])
                combined = embedding_dict["combined"].tolist()
                
                # Store in DB
                self.db._client.table("resumes").update({
                    "embedding_vector": combined
                }).eq("id", resume["id"]).execute()
                
                count += 1
                print(f"  ✅ Generated embedding for resume {count}")
            except Exception as e:
                print(f"  ❌ Failed for {resume['id']}: {e}")
        
        return count
    
    def rank_resumes(self, job_description: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Rank all resumes against a job description.
        
        Args:
            job_description: The job description text
            top_k: Number of top results to return
            
        Returns:
            List of ranked resumes with scores
        """
        # Generate embedding for job description
        print("Generating job description embedding...")
        job_embedding = self.embedder.embed(job_description)["combined"]
        
        # Fetch all resumes with embeddings
        print("Fetching resumes from database...")
        result = self.db._client.table("resumes").select(
            "id, filename, raw_text, embedding_vector"
        ).not_.is_("embedding_vector", "null").execute()
        
        if not result.data:
            print("No resumes with embeddings found!")
            return []
        
        print(f"Found {len(result.data)} resumes with embeddings")
        
        # Compute similarities
        rankings = []
        for resume in result.data:
            resume_embedding = np.array(resume["embedding_vector"])
            
            # Cosine similarity
            similarity = cosine_similarity(
                job_embedding.reshape(1, -1),
                resume_embedding.reshape(1, -1)
            )[0][0]
            
            rankings.append({
                "id": resume["id"],
                "filename": resume["filename"],
                "score": float(similarity),
                "raw_text_preview": resume["raw_text"][:200] + "..."
            })
        
        # Sort by score descending
        rankings.sort(key=lambda x: x["score"], reverse=True)
        
        return rankings[:top_k]


def main():
    ranker = ResumeRanker()
    
    # First, generate embeddings for all resumes
    print("\n📊 Step 1: Generating embeddings for resumes...")
    count = ranker.generate_embeddings_for_all()
    print(f"Generated {count} new embeddings\n")
    
    # Sample job description for Data Scientist
    job_description = """
    Data Scientist - Machine Learning
    
    We are looking for an experienced Data Scientist to join our team.
    
    Requirements:
    - 3+ years of experience in data science or machine learning
    - Strong proficiency in Python, including libraries like NumPy, Pandas, Scikit-learn
    - Experience with deep learning frameworks (TensorFlow, PyTorch)
    - Strong statistical analysis and mathematical modeling skills
    - Experience with NLP, computer vision, or recommendation systems
    - Familiarity with cloud platforms (AWS, GCP, Azure)
    - Excellent communication skills to present findings to stakeholders
    
    Responsibilities:
    - Build and deploy machine learning models to production
    - Analyze large datasets to extract actionable insights
    - Collaborate with engineering teams to implement ML solutions
    - Design and run A/B tests to measure model performance
    - Stay current with latest ML research and techniques
    
    Nice to have:
    - PhD in Computer Science, Statistics, or related field
    - Experience with MLOps and model monitoring
    - Publications in top ML conferences
    """
    
    print("📄 Step 2: Ranking resumes against job description...")
    print("-" * 50)
    
    rankings = ranker.rank_resumes(job_description, top_k=10)
    
    print("\n🏆 TOP 10 CANDIDATES:")
    print("=" * 60)
    
    for i, r in enumerate(rankings, 1):
        print(f"\n{i}. {r['filename']}")
        print(f"   Score: {r['score']:.4f}")
        print(f"   Preview: {r['raw_text_preview'][:100]}...")


if __name__ == "__main__":
    main()
