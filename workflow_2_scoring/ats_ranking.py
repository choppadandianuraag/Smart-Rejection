"""
ATS Resume Ranking System
Implements weighted scoring based on JD keyword matching with LLM extraction.
"""

import os
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
from ranking.zone_classifier import ZoneClassifier
from ranking.education_matcher import check_education_match, normalize_education
# FeedbackGenerator and EmailService only needed for main() function
# from feedback.feedback_generator import FeedbackGenerator
# from mailer.email_service import EmailService


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

    # Combined scoring weights (Cosine + ATS)
    COMBINED_WEIGHTS = {
        "cosine_similarity": 0.40,  # 40% weight for embedding similarity
        "ats_score": 0.60           # 60% weight for ATS scoring
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

    def _extract_jd_requirements_fast(self, job_description: str) -> Dict[str, Any]:
        """
        Fast keyword-based extraction of job requirements without LLM.
        Used when fast_mode=True to avoid API rate limits.

        Returns:
            Dict with skills, experience, and education requirements
        """
        import re

        text = job_description.lower()

        # Extract skills based on common tech keywords
        common_skills = [
            "python", "java", "javascript", "sql", "aws", "docker", "kubernetes",
            "tensorflow", "pytorch", "machine learning", "deep learning", "ai", "ml",
            "react", "node", "api", "rest", "graphql", "mongodb", "postgresql",
            "git", "linux", "bash", "ci/cd", "jenkins", "microservices"
        ]

        found_skills = []
        for skill in common_skills:
            if skill in text:
                found_skills.append(skill.title())

        # Extract experience years using regex
        years_patterns = [
            r'(\d+)\+?\s*(?:years?|yrs?)',
            r'minimum\s+of\s+(\d+)\s+years?',
            r'at\s+least\s+(\d+)\s+years?'
        ]

        min_years = 0
        for pattern in years_patterns:
            matches = re.findall(pattern, text)
            if matches:
                min_years = max(min_years, int(matches[0]))

        # Extract education level
        education_level = "bachelor"  # default
        if "phd" in text or "ph.d" in text or "doctorate" in text:
            education_level = "phd"
        elif "master" in text or "msc" in text or "m.s" in text:
            education_level = "master"
        elif "bachelor" in text or "b.s" in text or "b.tech" in text:
            education_level = "bachelor"

        return {
            "skills": {
                "must_have": found_skills[:5],  # Top 5 skills as must-have
                "good_to_have": found_skills[5:10],  # Next 5 as good-to-have
                "nice_to_have": found_skills[10:]  # Rest as nice-to-have
            },
            "experience": {
                "min_years": min_years,
                "preferred_years": min_years + 2,  # Add 2 years as preferred
                "relevant_domains": ["technology", "software development"]
            },
            "education": {
                "min_level": education_level,
                "preferred_level": education_level,
                "fields": ["Computer Science", "Engineering", "Technology"]
            }
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

    def extract_resume_profile_from_db(self, applicant_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract resume profile from pre-computed applicant_embeddings data.
        Uses skills_text, work_experience_text, education_text instead of LLM.

        Args:
            applicant_data: Dict with skills_text, work_experience_text, education_text, etc.

        Returns:
            Dict with skills, experience, education profile
        """
        # Parse skills from skills_text
        skills_text = applicant_data.get("skills_text", "")
        skills = [s.strip() for s in skills_text.split(",") if s.strip()] if skills_text else []

        # Parse experience from work_experience_text
        work_exp_text = applicant_data.get("work_experience_text", "")
        # Extract years from text (simple heuristic - look for numbers)
        import re
        years_match = re.findall(r'(\d+)\+?\s*(?:years?|yrs?)', work_exp_text.lower())
        total_years = max([int(y) for y in years_match]) if years_match else 0

        # Extract job titles (simple heuristic - lines that look like titles)
        job_titles = []
        domains = []
        if work_exp_text:
            lines = work_exp_text.split('\n')
            for line in lines[:10]:  # Check first 10 lines
                line = line.strip()
                if line and len(line) < 50 and not any(c.isdigit() for c in line[:5]):
                    job_titles.append(line)
            domains = [work_exp_text[:200]]  # Use first part as domain context

        # Parse education from education_text
        edu_text = applicant_data.get("education_text", "").lower()

        # Determine education level
        highest_level = "bachelor"  # default
        if "phd" in edu_text or "ph.d" in edu_text or "doctorate" in edu_text:
            highest_level = "phd"
        elif "master" in edu_text or "m.s" in edu_text or "m.tech" in edu_text or "mba" in edu_text:
            highest_level = "master"
        elif "bachelor" in edu_text or "b.s" in edu_text or "b.tech" in edu_text or "b.e" in edu_text:
            highest_level = "bachelor"
        elif "diploma" in edu_text:
            highest_level = "diploma"
        elif "associate" in edu_text:
            highest_level = "associate"

        # Extract field
        field = ""
        fields_to_check = ["computer science", "engineering", "mathematics", "physics",
                          "statistics", "data science", "information technology", "electronics"]
        for f in fields_to_check:
            if f in edu_text:
                field = f.title()
                break

        return {
            "skills": skills[:30],  # Limit to 30 skills
            "experience": {
                "total_years": total_years,
                "job_titles": job_titles[:5],
                "domains": domains
            },
            "education": {
                "highest_level": highest_level,
                "field": field,
                "degrees": [applicant_data.get("education_text", "")[:100]]
            }
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
        Uses education_matcher for degree hierarchy matching.

        Scoring logic:
        - Degree level: 50% (uses education_matcher for hierarchy)
        - Field relevance: 50% (embedding similarity of fields)
        """
        details = {}

        # 1. Degree level scoring (50% of education score) - Using education_matcher
        min_level = jd_education.get("min_level", "bachelor")
        preferred_level = jd_education.get("preferred_level", min_level)
        candidate_level = resume_education.get("highest_level", "bachelor")

        # Use the new education_matcher for hierarchy-aware scoring
        matches_min, min_score = check_education_match(candidate_level, min_level)
        matches_pref, pref_score = check_education_match(candidate_level, preferred_level)

        # Calculate level score based on matches
        if matches_pref:
            # Meets or exceeds preferred level
            level_score = min(pref_score / 100, 1.0)  # Cap at 1.0, bonus handled separately
        elif matches_min:
            # Meets minimum but not preferred
            level_score = 0.7 + 0.3 * (min_score / 100)
        else:
            # Below minimum requirement
            level_score = max(0.2, min_score / 100)

        # Get standardized names for display
        cand_std, cand_lvl = normalize_education(candidate_level)
        min_std, _ = normalize_education(min_level)
        pref_std, _ = normalize_education(preferred_level)

        details["level"] = {
            "required_min": min_std or min_level,
            "required_preferred": pref_std or preferred_level,
            "candidate_has": cand_std or candidate_level,
            "matches_minimum": matches_min,
            "matches_preferred": matches_pref,
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
        jd_requirements: Dict[str, Any],
        precomputed_profile: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Calculate the comprehensive ATS score for a resume.
        Includes Skills (60%), Experience (25%), Education (15%)

        Args:
            resume_text: Raw resume text
            jd_requirements: Extracted JD requirements
            precomputed_profile: Optional pre-computed profile from applicant_embeddings
                                 If provided, skips LLM extraction

        Returns:
            Dict with overall score and detailed breakdowns
        """
        # Use pre-computed profile if provided, otherwise extract with LLM
        if precomputed_profile:
            resume_profile = precomputed_profile
        else:
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
        
        print(f"\n[JD] REQUIREMENTS EXTRACTED:")
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

    def rank_applicants_v2(
        self,
        job_id: str,
        job_description: str,
        use_precomputed: bool = True,
        top_k: int = 100
    ) -> List[Dict]:
        """
        Rank applicants from V2 schema (applicant_profiles + applicant_embeddings).
        Uses pre-computed skills_text, work_experience_text, education_text.

        Args:
            job_id: Job UUID
            job_description: Job description text
            use_precomputed: If True, use pre-computed data from applicant_embeddings
                            If False, use LLM extraction (slower)
            top_k: Number of top candidates to return

        Returns:
            List of ranked applicants with scores
        """
        print("Step 1: Extracting comprehensive requirements from JD using LLM...")
        jd_requirements = self.extract_jd_requirements_llm(job_description)

        jd_skills = jd_requirements.get("skills", {})
        jd_experience = jd_requirements.get("experience", {})
        jd_education = jd_requirements.get("education", {})

        print(f"\n[JD REQUIREMENTS]")
        print(f"  Skills:")
        print(f"    - Must Have: {len(jd_skills.get('must_have', []))} items")
        print(f"    - Good to Have: {len(jd_skills.get('good_to_have', []))} items")
        print(f"  Experience: {jd_experience.get('min_years', 0)}-{jd_experience.get('preferred_years', 0)} years")
        print(f"  Education: {jd_education.get('min_level', 'N/A')} minimum")

        # Fetch applicants from V2 schema
        print("\nStep 2: Fetching applicants from database...")

        # Get applicant profiles
        profiles_result = self.db._client.table("applicant_profiles").select(
            "applicant_id, name, email, raw_text"
        ).execute()

        if not profiles_result.data:
            print("No applicants found!")
            return []

        # Get applicant embeddings (pre-computed data)
        embeddings_result = self.db._client.table("applicant_embeddings").select(
            "applicant_id, skills_text, education_text, work_experience_text, projects_text, certifications_text"
        ).execute()

        # Create lookup for embeddings
        emb_lookup = {e["applicant_id"]: e for e in embeddings_result.data} if embeddings_result.data else {}

        print(f"  Found {len(profiles_result.data)} applicants, {len(emb_lookup)} with pre-computed data")

        # Score each applicant
        mode = "pre-computed" if use_precomputed else "LLM"
        print(f"\nStep 3: Scoring applicants using {mode} profile extraction...")

        rankings = []
        for i, applicant in enumerate(profiles_result.data, 1):
            applicant_id = applicant["applicant_id"]
            name = applicant.get("name", "Unknown")

            print(f"  [{i:02d}/{len(profiles_result.data)}] Scoring {name[:20]}...", end=" ")

            try:
                precomputed_profile = None

                if use_precomputed and applicant_id in emb_lookup:
                    # Use pre-computed data from applicant_embeddings
                    emb_data = emb_lookup[applicant_id]
                    precomputed_profile = self.extract_resume_profile_from_db(emb_data)

                resume_text = applicant.get("raw_text", "")
                if not resume_text and not precomputed_profile:
                    print("SKIPPED (no data)")
                    continue

                score_result = self.calculate_ats_score(
                    resume_text or "",
                    jd_requirements,
                    precomputed_profile=precomputed_profile
                )

                rankings.append({
                    "applicant_id": applicant_id,
                    "name": name,
                    "email": applicant.get("email", ""),
                    "ats_score": score_result["ats_score"],
                    "skills_score": score_result["component_scores"]["skills"],
                    "experience_score": score_result["component_scores"]["experience"],
                    "education_score": score_result["component_scores"]["education"],
                    "years_exp": score_result["resume_profile"]["years_experience"],
                    "edu_level": score_result["resume_profile"]["education_level"],
                    "skills_found": score_result["resume_profile"]["skills_found"],
                    "job_titles": score_result["resume_profile"]["job_titles"]
                })
                print(f"ATS: {score_result['ats_score']}")

            except Exception as e:
                print(f"ERROR: {e}")
                continue

        # Sort by ATS score
        rankings.sort(key=lambda x: x["ats_score"], reverse=True)

        return rankings[:top_k]

    def rank_applicants_combined(
        self,
        job_id: str,
        job_description: str,
        use_precomputed: bool = True,
        fast_mode: bool = True,
        top_k: int = 100
    ) -> List[Dict]:
        """
        Rank applicants using COMBINED scoring: Cosine Similarity (40%) + ATS (60%).

        Args:
            job_id: Job UUID
            job_description: Job description text
            use_precomputed: If True, use pre-computed data from applicant_embeddings
            fast_mode: If True, use faster but less accurate skill matching
            top_k: Number of top candidates to return

        Returns:
            List of ranked applicants with combined scores
        """
        print("=" * 80)
        print(f"COMBINED SCORING: Cosine (40%) + ATS (60%) [{'FAST' if fast_mode else 'FULL'} MODE]")
        print("=" * 80)

        # Step 1: Extract JD requirements
        if fast_mode:
            print("\nStep 1: Using fast mode - skipping LLM extraction...")
            # Use simple keyword extraction for fast mode
            jd_requirements = self._extract_jd_requirements_fast(job_description)
        else:
            print("\nStep 1: Extracting JD requirements with LLM...")
            jd_requirements = self.extract_jd_requirements_llm(job_description)

        jd_skills = jd_requirements.get("skills", {})
        jd_experience = jd_requirements.get("experience", {})
        jd_education = jd_requirements.get("education", {})

        print(f"\n[JD REQUIREMENTS]")
        print(f"  Skills: {len(jd_skills.get('must_have', []))} must-have, {len(jd_skills.get('good_to_have', []))} good-to-have")
        print(f"  Experience: {jd_experience.get('min_years', 0)}-{jd_experience.get('preferred_years', 0)} years")
        print(f"  Education: {jd_education.get('min_level', 'N/A')} minimum")

        # Step 2: Get job embedding
        print("\nStep 2: Loading job embedding...")
        job_emb_result = self.db._client.table("job_embeddings").select(
            "embedding_vector"
        ).eq("job_id", job_id).execute()

        if not job_emb_result.data:
            print("ERROR: No job embedding found!")
            return []

        job_embedding = np.array(json.loads(job_emb_result.data[0]["embedding_vector"]))
        print(f"  Job embedding loaded: {len(job_embedding)} dimensions")

        # Step 3: Get applicants
        print("\nStep 3: Fetching applicants...")
        profiles_result = self.db._client.table("applicant_profiles").select(
            "applicant_id, name, email, raw_text"
        ).execute()

        if not profiles_result.data:
            print("No applicants found!")
            return []

        # Get applicant embeddings
        embeddings_result = self.db._client.table("applicant_embeddings").select(
            "applicant_id, resume_embedding, skills_text, education_text, work_experience_text"
        ).execute()

        emb_lookup = {e["applicant_id"]: e for e in embeddings_result.data} if embeddings_result.data else {}

        print(f"  Found {len(profiles_result.data)} applicants, {len(emb_lookup)} with embeddings")

        # Step 4: Score each applicant
        mode = f"{'pre-computed' if use_precomputed else 'LLM'} + {'fast' if fast_mode else 'full'}"
        print(f"\nStep 4: Scoring applicants ({mode})...")

        rankings = []
        for i, applicant in enumerate(profiles_result.data, 1):
            applicant_id = applicant["applicant_id"]
            name = applicant.get("name", "Unknown")

            print(f"  [{i:02d}/{len(profiles_result.data)}] {name[:20]}...", end=" ")

            try:
                if applicant_id not in emb_lookup:
                    print("SKIP (no embedding)")
                    continue

                emb_data = emb_lookup[applicant_id]

                # 1. Compute Cosine Similarity
                resume_embedding_raw = emb_data.get("resume_embedding")
                if not resume_embedding_raw:
                    print("SKIP (no resume embedding)")
                    continue

                resume_embedding = np.array(json.loads(resume_embedding_raw) if isinstance(resume_embedding_raw, str) else resume_embedding_raw)

                cosine_sim = cosine_similarity(
                    job_embedding.reshape(1, -1),
                    resume_embedding.reshape(1, -1)
                )[0][0]

                cosine_score = cosine_sim * 100  # Convert to 0-100

                # 2. Compute ATS Score (optimized for fast mode)
                if fast_mode:
                    # Fast ATS: Simple keyword matching + basic scoring
                    ats_score = self._calculate_fast_ats_score(emb_data, jd_requirements)
                else:
                    # Full ATS: Complete embedding-based matching
                    precomputed_profile = None
                    if use_precomputed:
                        precomputed_profile = self.extract_resume_profile_from_db(emb_data)

                    resume_text = applicant.get("raw_text", "")
                    if not resume_text and not precomputed_profile:
                        print("SKIP (no data)")
                        continue

                    ats_result = self.calculate_ats_score(
                        resume_text or "",
                        jd_requirements,
                        precomputed_profile=precomputed_profile
                    )
                    ats_score = ats_result["ats_score"]

                # 3. Combine scores
                combined_score = (
                    cosine_score * self.COMBINED_WEIGHTS["cosine_similarity"] +
                    ats_score * self.COMBINED_WEIGHTS["ats_score"]
                )

                rankings.append({
                    "applicant_id": applicant_id,
                    "name": name,
                    "email": applicant.get("email", ""),
                    "combined_score": round(combined_score, 1),
                    "cosine_score": round(cosine_score, 1),
                    "ats_score": round(ats_score, 1),
                    "skills_score": ats_score if fast_mode else ats_result["component_scores"]["skills"],
                    "experience_score": 0 if fast_mode else ats_result["component_scores"]["experience"],
                    "education_score": 0 if fast_mode else ats_result["component_scores"]["education"],
                    "years_exp": 0 if fast_mode else ats_result["resume_profile"]["years_experience"],
                    "edu_level": "" if fast_mode else ats_result["resume_profile"]["education_level"],
                    "skills_found": [] if fast_mode else ats_result["resume_profile"]["skills_found"],
                    "job_titles": [] if fast_mode else ats_result["resume_profile"]["job_titles"]
                })

                print(f"Combined: {combined_score:.1f} (Cosine: {cosine_score:.1f}, ATS: {ats_score:.1f})")

            except Exception as e:
                print(f"ERROR: {e}")
                continue

        # Sort by combined score
        rankings.sort(key=lambda x: x["combined_score"], reverse=True)

        return rankings[:top_k]

    def _calculate_fast_ats_score(self, emb_data: Dict, jd_requirements: Dict) -> float:
        """
        Fast ATS scoring using simple keyword matching instead of embeddings.
        Much faster but less accurate than full ATS scoring.
        """
        # Get candidate skills
        skills_text = emb_data.get("skills_text", "").lower()
        work_text = emb_data.get("work_experience_text", "").lower()
        edu_text = emb_data.get("education_text", "").lower()

        combined_text = f"{skills_text} {work_text} {edu_text}"

        # Count matches for each skill category
        jd_skills = jd_requirements.get("skills", {})
        total_score = 0
        total_weight = 0

        for category, skills_list in jd_skills.items():
            if not skills_list:
                continue

            matches = 0
            for skill in skills_list:
                # Simple substring matching
                if any(keyword.lower() in combined_text for keyword in skill.lower().split()):
                    matches += 1

            # Calculate category score
            category_score = (matches / len(skills_list)) * 100 if skills_list else 0

            # Apply weights
            if category == "must_have":
                weight = 0.5
            elif category == "good_to_have":
                weight = 0.3
            else:  # nice_to_have
                weight = 0.2

            total_score += category_score * weight
            total_weight += weight

        # Return weighted average (skills only for fast mode)
        return total_score / total_weight if total_weight > 0 else 0

    def fetch_job_descriptions(self) -> List[Dict[str, Any]]:
        """
        Fetch all job descriptions from the database.

        Returns:
            List of job description records
        """
        result = self.db._client.table("job_descriptions").select(
            "id, title, company, location, description, requirements"
        ).execute()

        return result.data if result.data else []

    def get_job_description_text(self, jd_record: Dict[str, Any]) -> str:
        """
        Combine job description fields into a single text for analysis.

        Args:
            jd_record: Job description record from database

        Returns:
            Combined job description text
        """
        parts = []

        if jd_record.get("title"):
            parts.append(f"Job Title: {jd_record['title']}")

        if jd_record.get("company"):
            parts.append(f"Company: {jd_record['company']}")

        if jd_record.get("location"):
            parts.append(f"Location: {jd_record['location']}")

        if jd_record.get("description"):
            parts.append(f"\nJob Description:\n{jd_record['description']}")

        if jd_record.get("requirements"):
            parts.append(f"\nRequirements:\n{jd_record['requirements']}")

        return "\n".join(parts)


def main():
    print("=" * 70)
    print("[*] ATS RESUME RANKING SYSTEM (Enhanced)")
    print("   Scoring: Skills (60%) + Experience (25%) + Education (15%)")
    print("=" * 70)

    ranker = ATSRankingSystem()

    # Fetch job descriptions from database
    print("\nFetching job descriptions from database...")
    job_descriptions = ranker.fetch_job_descriptions()

    if not job_descriptions:
        print("ERROR: No job descriptions found in database!")
        print("Please add a job description to the 'job_descriptions' table first.")
        return

    # Use the first (or most recent) job description
    jd_record = job_descriptions[0]
    job_description = ranker.get_job_description_text(jd_record)

    print(f"[JD] Using Job Description:")
    print(f"  - Title: {jd_record.get('title', 'N/A')}")
    print(f"  - Company: {jd_record.get('company', 'N/A')}")
    print(f"  - Location: {jd_record.get('location', 'N/A')}")

    rankings = ranker.rank_all_resumes(job_description, top_k=15)
    
    print("\n" + "=" * 90)
    print("[TOP] TOP 15 CANDIDATES (COMPREHENSIVE ATS SCORES)")
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
        print(f"[DETAILS] TOP CANDIDATE DETAILS: {rankings[0]['filename']}")
        print("=" * 70)
        print(f"[SCORE] ATS Score: {rankings[0]['ats_score']}/100")
        print(f"[BREAKDOWN] Component Scores:")
        print(f"   - Skills:     {rankings[0]['skills_score']}/100 (60% weight)")
        print(f"   - Experience: {rankings[0]['experience_score']}/100 (25% weight)")
        print(f"   - Education:  {rankings[0]['education_score']}/100 (15% weight)")
        print(f"\n[PROFILE] Profile:")
        print(f"   - Years of Experience: {rankings[0]['years_exp']}")
        print(f"   - Education Level: {rankings[0]['edu_level']}")
        print(f"   - Job Titles: {', '.join(rankings[0]['job_titles'][:3]) if rankings[0]['job_titles'] else 'N/A'}")
        print(f"   - Top Skills: {', '.join(rankings[0]['skills_found'][:8])}")

    # ========== PHASE 2: ZONE CLASSIFICATION ==========
    if rankings:
        print("\n" + "=" * 70)
        print("[ZONE] SMART REJECTION - ZONE CLASSIFICATION")
        print("=" * 70)

        # Initialize zone classifier with thresholds
        classifier = ZoneClassifier(selected_threshold=75, borderline_threshold=40)

        # Classify all ranked candidates
        classified_results = classifier.batch_classify(rankings)

        # Get summary statistics
        stats = classifier.get_summary_stats(classified_results)

        # Display SELECTED candidates
        print(f"\n[OK] SELECTED ({stats['selected_count']} candidates) - Moving to interview round:")
        if classified_results['selected']:
            for candidate in classified_results['selected']:
                score = candidate.get('ats_score', candidate.get('score', 0))
                name = candidate.get('filename', candidate.get('name', 'Unknown'))
                print(f"     - {name}: {score}/100")
        else:
            print("     (No candidates met the selection threshold)")

        # Display BORDERLINE candidates (these get feedback emails)
        print(f"\n[!!] BORDERLINE ({stats['borderline_count']} candidates) - Will receive feedback emails:")
        if classified_results['borderline']:
            for candidate in classified_results['borderline']:
                score = candidate.get('ats_score', candidate.get('score', 0))
                name = candidate.get('filename', candidate.get('name', 'Unknown'))
                print(f"     - {name}: {score}/100")
        else:
            print("     (No candidates in borderline zone)")

        # Display REJECTED candidates
        print(f"\n[XX] REJECTED ({stats['rejected_count']} candidates) - No action needed:")
        if classified_results['rejected']:
            for candidate in classified_results['rejected']:
                score = candidate.get('ats_score', candidate.get('score', 0))
                name = candidate.get('filename', candidate.get('name', 'Unknown'))
                print(f"     - {name}: {score}/100")
        else:
            print("     (No candidates in rejected zone)")

        # Summary
        print(f"\n" + "-" * 70)
        print(f"[SUMMARY] Classification Summary:")
        print(f"   Total Candidates: {stats['total_candidates']}")
        print(f"   Selected: {stats['selected_count']} ({stats['selected_percentage']}%)")
        print(f"   Borderline: {stats['borderline_count']} ({stats['borderline_percentage']}%)")
        print(f"   Rejected: {stats['rejected_count']} ({stats['rejected_percentage']}%)")
        print(f"\n[EMAIL] Feedback Emails to Send: {stats['feedback_emails_to_send']}")

        # ========== PHASE 2.2: FEEDBACK GENERATION ==========
        if classified_results['borderline']:
            print("\n" + "=" * 70)
            print("[FEEDBACK] GENERATING PERSONALIZED FEEDBACK FOR BORDERLINE CANDIDATES")
            print("=" * 70)

            # Lazy import
            from feedback.feedback_generator import FeedbackGenerator
            feedback_gen = FeedbackGenerator()

            for candidate in classified_results['borderline']:
                # Generate feedback using candidate data and scores
                feedback = feedback_gen.generate_feedback(
                    candidate_data=candidate,
                    score_breakdown=candidate
                )

                # Print the feedback report
                print(feedback_gen.generate_text_report(feedback))

                # Store feedback in candidate dict for later use (e.g., email sending)
                candidate['feedback'] = feedback
                candidate['feedback_report'] = feedback_gen.generate_text_report(feedback)

            # ========== PHASE 3: EMAIL INTEGRATION ==========
            print("\n" + "=" * 70)
            print("[EMAIL] PHASE 3: SENDING FEEDBACK EMAILS")
            print("=" * 70)

            # Lazy import
            from mailer.email_service import EmailService
            # Initialize email service in print mode (for development)
            # Change to "sendgrid" or "smtp" for production
            email_service = EmailService(mode="print")

            # Get job details for email context
            job_title = jd_record.get('title', 'Position')
            company = jd_record.get('company', 'Company')

            # Send emails to all borderline candidates
            email_results = email_service.send_batch_emails(
                candidates=classified_results['borderline'],
                job_title=job_title,
                company=company
            )

            # Show email statistics
            email_stats = email_service.get_email_stats()
            print(f"\n[STATS] Email Statistics:")
            print(f"   Mode: {email_stats['mode']}")
            print(f"   Total Sent: {email_stats['total_sent']}")
            print(f"   Successful: {email_stats['successful']}")
            print(f"   Failed: {email_stats['failed']}")

        # Return borderline candidates for feedback generation (Phase 2.2)
        return classified_results['borderline']

    return []


if __name__ == "__main__":
    main()
