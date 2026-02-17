"""
ATS Resume Ranking System
Implements weighted scoring based on JD keyword matching with LLM extraction.
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from database.supabase_client import SupabaseClient
from embeddings.hybrid_embedder import HybridEmbedder


class ATSRankingSystem:
    """
    ATS-style resume ranking using:
    1. LLM keyword extraction from JD (categorized)
    2. LLM keyword extraction from Resume
    3. Embedding-based similarity per category
    4. Weighted scoring
    """
    
    # Category weights for final score (Skills: 60%, Experience: 25%, Education: 15%)
    SKILL_WEIGHTS = {
        "must_have": 0.50,      # 50% of skills weight
        "good_to_have": 0.30,   # 30% of skills weight
        "nice_to_have": 0.20    # 20% of skills weight
    }
    
    OVERALL_WEIGHTS = {
        "skills": 0.60,         # 60% weight for skills
        "experience": 0.25,     # 25% weight for experience
        "education": 0.15       # 15% weight for education
    }
    
    # Education level hierarchy (higher = better)
    EDUCATION_LEVELS = {
        "high school": 1,
        "diploma": 2,
        "associate": 3,
        "bachelor": 4,
        "master": 5,
        "mba": 5,
        "phd": 6,
        "doctorate": 6
    }
    
    def __init__(self):
        self.db = SupabaseClient()
        self.embedder = HybridEmbedder()
        
        # Initialize Groq LLM
        groq_api_key = os.getenv('GROQ_API_KEY')
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            groq_api_key=groq_api_key,
            temperature=0
        )
        
    def extract_jd_requirements_llm(self, job_description: str) -> Dict[str, Any]:
        """
        Extract comprehensive requirements from job description using LLM.
        Includes skills (categorized), experience, and education requirements.
        
        Returns:
            Dict with skills, experience, and education requirements
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert HR analyst. Extract ALL requirements from the job description.

Extract and return ONLY valid JSON in this exact format:
{{
    "skills": {{
        "must_have": ["skill1", "skill2"],
        "good_to_have": ["skill1", "skill2"],
        "nice_to_have": ["skill1", "skill2"]
    }},
    "experience": {{
        "min_years": 3,
        "preferred_years": 5,
        "relevant_domains": ["machine learning", "data science"]
    }},
    "education": {{
        "min_level": "bachelor",
        "preferred_level": "master",
        "fields": ["Computer Science", "Statistics", "Mathematics"]
    }}
}}

Rules:
- min_years: minimum required years (0 if not specified)
- preferred_years: ideal years (same as min if not specified)
- min_level: one of [high school, diploma, associate, bachelor, master, phd]
- preferred_level: one of [high school, diploma, associate, bachelor, master, phd]
- Extract specific skills, technologies as concise items"""),
            ("human", "{job_description}")
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({"job_description": job_description})
        
        # Parse JSON from response
        try:
            content = response.content
            start = content.find('{')
            end = content.rfind('}') + 1
            json_str = content[start:end]
            requirements = json.loads(json_str)
            
            # Ensure all keys exist with defaults
            if "skills" not in requirements:
                requirements["skills"] = {"must_have": [], "good_to_have": [], "nice_to_have": []}
            if "experience" not in requirements:
                requirements["experience"] = {"min_years": 0, "preferred_years": 0, "relevant_domains": []}
            if "education" not in requirements:
                requirements["education"] = {"min_level": "bachelor", "preferred_level": "bachelor", "fields": []}
            
            return requirements
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            return {
                "skills": {"must_have": [], "good_to_have": [], "nice_to_have": []},
                "experience": {"min_years": 0, "preferred_years": 0, "relevant_domains": []},
                "education": {"min_level": "bachelor", "preferred_level": "bachelor", "fields": []}
            }
    
    def extract_resume_profile_llm(self, resume_text: str) -> Dict[str, Any]:
        """
        Extract comprehensive profile from resume: skills, experience, education.
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert resume analyzer. Extract the candidate's complete profile.

Return ONLY valid JSON in this exact format:
{{
    "skills": ["Python", "SQL", "TensorFlow", ...],
    "experience": {{
        "total_years": 5,
        "job_titles": ["Data Scientist", "ML Engineer"],
        "domains": ["machine learning", "data analytics"]
    }},
    "education": {{
        "highest_level": "master",
        "field": "Computer Science",
        "degrees": ["M.S. Computer Science", "B.Tech Information Technology"]
    }}
}}

Rules:
- total_years: Calculate total professional experience (estimate from dates if needed)
- highest_level: one of [high school, diploma, associate, bachelor, master, phd]
- Extract ALL technical skills, tools, frameworks, languages
- Include certifications in skills array"""),
            ("human", "{resume_text}")
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({"resume_text": resume_text[:5000]})  # Limit text
        
        try:
            content = response.content
            start = content.find('{')
            end = content.rfind('}') + 1
            json_str = content[start:end]
            profile = json.loads(json_str)
            
            # Ensure all keys exist with defaults
            if "skills" not in profile:
                profile["skills"] = []
            if "experience" not in profile:
                profile["experience"] = {"total_years": 0, "job_titles": [], "domains": []}
            if "education" not in profile:
                profile["education"] = {"highest_level": "bachelor", "field": "", "degrees": []}
            
            return profile
        except Exception as e:
            print(f"Error parsing resume profile: {e}")
            return {
                "skills": [],
                "experience": {"total_years": 0, "job_titles": [], "domains": []},
                "education": {"highest_level": "bachelor", "field": "", "degrees": []}
            }
    
    def compute_category_score(
        self, 
        jd_requirements: List[str], 
        resume_text: str,
        resume_skills: List[str]
    ) -> Tuple[float, List[Dict]]:
        """
        Compute similarity score for a category.
        Uses embedding similarity between JD requirements and resume skills.
        
        Returns:
            score (0-1), list of matched items with scores
        """
        if not jd_requirements:
            return 1.0, []  # No requirements = full score
        
        matches = []
        total_score = 0
        
        # Create resume skills text for embedding
        resume_skills_text = " ".join(resume_skills) if resume_skills else resume_text[:2000]
        resume_embedding = self.embedder.embed(resume_skills_text)["combined"]
        
        for req in jd_requirements:
            # Generate embedding for requirement
            req_embedding = self.embedder.embed(req)["combined"]
            
            # Compute similarity
            similarity = cosine_similarity(
                req_embedding.reshape(1, -1),
                resume_embedding.reshape(1, -1)
            )[0][0]
            
            # Check for direct keyword matches (boost score)
            req_lower = req.lower()
            keyword_bonus = 0
            for skill in resume_skills:
                if isinstance(skill, str) and skill.lower() in req_lower:
                    keyword_bonus = 0.15  # 15% bonus for direct match
                    break
            
            final_score = min(1.0, similarity + keyword_bonus)
            total_score += final_score
            
            matches.append({
                "requirement": req[:50] + "..." if len(req) > 50 else req,
                "similarity": round(similarity, 3),
                "keyword_match": keyword_bonus > 0,
                "final_score": round(final_score, 3)
            })
        
        avg_score = total_score / len(jd_requirements)
        return avg_score, matches
    
    def compute_experience_score(
        self,
        jd_experience: Dict[str, Any],
        resume_experience: Dict[str, Any]
    ) -> Tuple[float, Dict]:
        """
        Score candidate's experience against JD requirements.
        
        Scoring logic:
        - Years: 40% (how well years match requirements)
        - Domain relevance: 60% (embedding similarity of domains)
        """
        details = {}
        
        # 1. Years scoring (40% of experience score)
        min_years = jd_experience.get("min_years", 0)
        preferred_years = jd_experience.get("preferred_years", min_years)
        candidate_years = resume_experience.get("total_years", 0)
        
        if min_years == 0:
            years_score = 1.0  # No requirement
        elif candidate_years >= preferred_years:
            years_score = 1.0  # Exceeds or meets preferred
        elif candidate_years >= min_years:
            # Between min and preferred - partial score
            years_score = 0.7 + 0.3 * ((candidate_years - min_years) / max(1, preferred_years - min_years))
        elif candidate_years >= min_years * 0.7:
            # Close to minimum (within 30%)
            years_score = 0.4 + 0.3 * (candidate_years / max(1, min_years))
        else:
            # Below threshold
            years_score = max(0.1, candidate_years / max(1, min_years))
        
        details["years"] = {
            "required_min": min_years,
            "required_preferred": preferred_years,
            "candidate_has": candidate_years,
            "score": round(years_score * 100, 1)
        }
        
        # 2. Domain relevance scoring (60% of experience score)
        jd_domains = jd_experience.get("relevant_domains", [])
        resume_domains = resume_experience.get("domains", [])
        resume_titles = resume_experience.get("job_titles", [])
        
        if not jd_domains:
            domain_score = 1.0  # No specific domain required
        else:
            # Create text from resume domains and titles
            resume_domain_text = " ".join(resume_domains + resume_titles)
            if not resume_domain_text:
                domain_score = 0.3  # No domain info found
            else:
                resume_embed = self.embedder.embed(resume_domain_text)["combined"]
                jd_domain_text = " ".join(jd_domains)
                jd_embed = self.embedder.embed(jd_domain_text)["combined"]
                
                domain_score = cosine_similarity(
                    resume_embed.reshape(1, -1),
                    jd_embed.reshape(1, -1)
                )[0][0]
        
        details["domain_relevance"] = {
            "required_domains": jd_domains[:3],
            "candidate_domains": resume_domains[:3],
            "candidate_titles": resume_titles[:3],
            "score": round(domain_score * 100, 1)
        }
        
        # Combined experience score
        final_score = (years_score * 0.4) + (domain_score * 0.6)
        
        return final_score, details
    
    def compute_education_score(
        self,
        jd_education: Dict[str, Any],
        resume_education: Dict[str, Any]
    ) -> Tuple[float, Dict]:
        """
        Score candidate's education against JD requirements.
        
        Scoring logic:
        - Degree level: 50% (meets/exceeds required level)
        - Field relevance: 50% (embedding similarity of fields)
        """
        details = {}
        
        # 1. Degree level scoring (50% of education score)
        min_level = jd_education.get("min_level", "bachelor").lower()
        preferred_level = jd_education.get("preferred_level", min_level).lower()
        candidate_level = resume_education.get("highest_level", "bachelor").lower()
        
        # Get numeric values
        min_level_val = self.EDUCATION_LEVELS.get(min_level, 4)
        preferred_level_val = self.EDUCATION_LEVELS.get(preferred_level, min_level_val)
        candidate_level_val = self.EDUCATION_LEVELS.get(candidate_level, 4)
        
        if candidate_level_val >= preferred_level_val:
            level_score = 1.0  # Meets or exceeds preferred
        elif candidate_level_val >= min_level_val:
            # Between min and preferred
            level_score = 0.7 + 0.3 * ((candidate_level_val - min_level_val) / 
                                        max(1, preferred_level_val - min_level_val))
        elif candidate_level_val == min_level_val - 1:
            level_score = 0.5  # One level below minimum
        else:
            level_score = 0.2  # Below minimum
        
        details["level"] = {
            "required_min": min_level,
            "required_preferred": preferred_level,
            "candidate_has": candidate_level,
            "score": round(level_score * 100, 1)
        }
        
        # 2. Field relevance scoring (50% of education score)
        jd_fields = jd_education.get("fields", [])
        candidate_field = resume_education.get("field", "")
        candidate_degrees = resume_education.get("degrees", [])
        
        if not jd_fields:
            field_score = 1.0  # No specific field required
        else:
            # Create text from candidate's education
            candidate_edu_text = f"{candidate_field} " + " ".join(candidate_degrees)
            if not candidate_edu_text.strip():
                field_score = 0.3  # No field info found
            else:
                candidate_embed = self.embedder.embed(candidate_edu_text)["combined"]
                jd_field_text = " ".join(jd_fields)
                jd_embed = self.embedder.embed(jd_field_text)["combined"]
                
                field_score = cosine_similarity(
                    candidate_embed.reshape(1, -1),
                    jd_embed.reshape(1, -1)
                )[0][0]
        
        details["field_relevance"] = {
            "required_fields": jd_fields[:3],
            "candidate_field": candidate_field,
            "candidate_degrees": candidate_degrees[:2],
            "score": round(field_score * 100, 1)
        }
        
        # Combined education score
        final_score = (level_score * 0.5) + (field_score * 0.5)
        
        return final_score, details
    
    def calculate_ats_score(
        self,
        resume_text: str,
        jd_requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate the comprehensive ATS score for a resume.
        Includes Skills (60%), Experience (25%), Education (15%)
        
        Returns:
            Dict with overall score and detailed breakdowns
        """
        # Extract complete profile from resume using LLM
        resume_profile = self.extract_resume_profile_llm(resume_text)
        
        resume_skills = resume_profile.get("skills", [])
        resume_experience = resume_profile.get("experience", {})
        resume_education = resume_profile.get("education", {})
        
        jd_skills = jd_requirements.get("skills", {})
        jd_experience = jd_requirements.get("experience", {})
        jd_education = jd_requirements.get("education", {})
        
        # ============ 1. SKILLS SCORING (60% of total) ============
        skill_category_scores = {}
        skill_details = {}
        
        for category in ["must_have", "good_to_have", "nice_to_have"]:
            requirements = jd_skills.get(category, [])
            score, matches = self.compute_category_score(
                requirements, 
                resume_text,
                resume_skills
            )
            skill_category_scores[category] = score
            skill_details[category] = {
                "score": round(score * 100, 1),
                "requirements_count": len(requirements),
                "matches": matches[:5]  # Limit for readability
            }
        
        # Weighted skills score
        skills_score = sum(
            skill_category_scores.get(cat, 0) * weight 
            for cat, weight in self.SKILL_WEIGHTS.items()
        )
        
        # ============ 2. EXPERIENCE SCORING (25% of total) ============
        experience_score, experience_details = self.compute_experience_score(
            jd_experience, resume_experience
        )
        
        # ============ 3. EDUCATION SCORING (15% of total) ============
        education_score, education_details = self.compute_education_score(
            jd_education, resume_education
        )
        
        # ============ FINAL ATS SCORE ============
        final_score = (
            skills_score * self.OVERALL_WEIGHTS["skills"] +
            experience_score * self.OVERALL_WEIGHTS["experience"] +
            education_score * self.OVERALL_WEIGHTS["education"]
        )
        
        return {
            "ats_score": round(final_score * 100, 1),
            "component_scores": {
                "skills": round(skills_score * 100, 1),
                "experience": round(experience_score * 100, 1),
                "education": round(education_score * 100, 1)
            },
            "skill_breakdown": {
                "must_have": round(skill_category_scores.get("must_have", 0) * 100, 1),
                "good_to_have": round(skill_category_scores.get("good_to_have", 0) * 100, 1),
                "nice_to_have": round(skill_category_scores.get("nice_to_have", 0) * 100, 1)
            },
            "weights": self.OVERALL_WEIGHTS,
            "resume_profile": {
                "skills_found": resume_skills[:15],
                "years_experience": resume_experience.get("total_years", 0),
                "education_level": resume_education.get("highest_level", ""),
                "job_titles": resume_experience.get("job_titles", [])[:3]
            },
            "details": {
                "skills": skill_details,
                "experience": experience_details,
                "education": education_details
            }
        }
    
    def rank_all_resumes(self, job_description: str, top_k: int = 10) -> List[Dict]:
        """
        Rank all resumes against a job description using LLM.
        Now includes Skills, Experience, and Education scoring.
        """
        print("Step 1: Extracting comprehensive requirements from JD using LLM...")
        jd_requirements = self.extract_jd_requirements_llm(job_description)
        
        jd_skills = jd_requirements.get("skills", {})
        jd_experience = jd_requirements.get("experience", {})
        jd_education = jd_requirements.get("education", {})
        
        print(f"\n📋 JD REQUIREMENTS EXTRACTED:")
        print(f"  Skills:")
        print(f"    - Must Have: {len(jd_skills.get('must_have', []))} items")
        print(f"    - Good to Have: {len(jd_skills.get('good_to_have', []))} items")
        print(f"    - Nice to Have: {len(jd_skills.get('nice_to_have', []))} items")
        print(f"  Experience:")
        print(f"    - Min Years: {jd_experience.get('min_years', 0)}")
        print(f"    - Preferred Years: {jd_experience.get('preferred_years', 0)}")
        print(f"    - Domains: {jd_experience.get('relevant_domains', [])[:3]}")
        print(f"  Education:")
        print(f"    - Min Level: {jd_education.get('min_level', 'bachelor')}")
        print(f"    - Preferred Level: {jd_education.get('preferred_level', 'bachelor')}")
        print(f"    - Fields: {jd_education.get('fields', [])[:3]}")
        
        # Fetch all resumes
        print("\nStep 2: Fetching resumes from database...")
        result = self.db._client.table("resumes").select(
            "id, filename, raw_text"
        ).execute()
        
        if not result.data:
            print("No resumes found!")
            return []
        
        print(f"  Found {len(result.data)} resumes")
        
        # Score each resume
        print("\nStep 3-6: Scoring resumes (Skills + Experience + Education)...")
        rankings = []
        
        for i, resume in enumerate(result.data, 1):
            print(f"  [{i:02d}/{len(result.data)}] Scoring {resume['filename']}...")
            
            score_result = self.calculate_ats_score(
                resume["raw_text"],
                jd_requirements
            )
            
            rankings.append({
                "id": resume["id"],
                "filename": resume["filename"],
                "ats_score": score_result["ats_score"],
                "skills_score": score_result["component_scores"]["skills"],
                "experience_score": score_result["component_scores"]["experience"],
                "education_score": score_result["component_scores"]["education"],
                "years_exp": score_result["resume_profile"]["years_experience"],
                "edu_level": score_result["resume_profile"]["education_level"],
                "skills_found": score_result["resume_profile"]["skills_found"],
                "job_titles": score_result["resume_profile"]["job_titles"]
            })
        
        # Sort by ATS score
        rankings.sort(key=lambda x: x["ats_score"], reverse=True)
        
        return rankings[:top_k]


def main():
    job_description = """
Key Responsibilities

Design, develop, and deploy machine learning models to solve real-world business problems
Perform exploratory data analysis (EDA) and statistical analysis on large datasets
Collaborate with the engineering team to integrate ML models into production systems
Build and maintain data pipelines for model training and evaluation
Communicate findings and insights to both technical and non-technical stakeholders
Monitor and optimize the performance of deployed models over time
Stay updated with the latest trends and techniques in machine learning and AI


Required Experience: 3-5 years of experience in data science or machine learning roles


Must Have (Required Skills)

Strong proficiency in Python (NumPy, Pandas, Scikit-learn)
Solid understanding of Machine Learning algorithms (Supervised & Unsupervised)
Experience with SQL and relational databases
Hands-on experience with data preprocessing, feature engineering, and model evaluation
Good understanding of statistical concepts (hypothesis testing, probability, distributions)
Experience working with cloud platforms (AWS / GCP / Azure)
Strong analytical and problem-solving skills
Bachelor's degree or above in Computer Science, Statistics, or a related field


Good to Have (Preferred Skills)

Experience with Deep Learning frameworks (TensorFlow / PyTorch)
Familiarity with NLP (Natural Language Processing) techniques
Experience with big data tools (Apache Spark, Hadoop)
Knowledge of MLOps practices and tools (MLflow, Docker, Kubernetes)
Experience with data visualization tools (Tableau / Power BI / Matplotlib)
Exposure to A/B testing and experimentation frameworks
Experience working in an agile/scrum environment
Master's degree preferred


Nice to Have (Bonus Skills)

Open source contributions in data science or ML
Experience with Recommendation Systems or Computer Vision
Knowledge of cloud-native ML services (AWS SageMaker, Google Vertex AI)
Published research papers or blog posts in data science or AI
Experience with API development and model serving (FastAPI, Flask)
Familiarity with Git and version control systems
"""

    print("=" * 70)
    print("🎯 ATS RESUME RANKING SYSTEM (Enhanced)")
    print("   Scoring: Skills (60%) + Experience (25%) + Education (15%)")
    print("=" * 70)
    
    ranker = ATSRankingSystem()
    rankings = ranker.rank_all_resumes(job_description, top_k=15)
    
    print("\n" + "=" * 90)
    print("🏆 TOP 15 CANDIDATES (COMPREHENSIVE ATS SCORES)")
    print("=" * 90)
    print(f"{'Rank':<5} {'Filename':<22} {'ATS':<7} {'Skills':<8} {'Exp':<7} {'Edu':<7} {'Yrs':<5} {'Level':<10}")
    print("-" * 90)
    
    for i, r in enumerate(rankings, 1):
        edu_display = r['edu_level'][:8] if r['edu_level'] else "N/A"
        print(f"{i:<5} {r['filename'][:21]:<22} {r['ats_score']:<7} "
              f"{r['skills_score']:<8} {r['experience_score']:<7} "
              f"{r['education_score']:<7} {r['years_exp']:<5} {edu_display:<10}")
    
    # Show top candidate details
    if rankings:
        print("\n" + "=" * 70)
        print(f"📋 TOP CANDIDATE DETAILS: {rankings[0]['filename']}")
        print("=" * 70)
        print(f"🎯 ATS Score: {rankings[0]['ats_score']}/100")
        print(f"📊 Component Scores:")
        print(f"   - Skills:     {rankings[0]['skills_score']}/100 (60% weight)")
        print(f"   - Experience: {rankings[0]['experience_score']}/100 (25% weight)")
        print(f"   - Education:  {rankings[0]['education_score']}/100 (15% weight)")
        print(f"\n👤 Profile:")
        print(f"   - Years of Experience: {rankings[0]['years_exp']}")
        print(f"   - Education Level: {rankings[0]['edu_level']}")
        print(f"   - Job Titles: {', '.join(rankings[0]['job_titles'][:3]) if rankings[0]['job_titles'] else 'N/A'}")
        print(f"   - Top Skills: {', '.join(rankings[0]['skills_found'][:8])}")


if __name__ == "__main__":
    main()
