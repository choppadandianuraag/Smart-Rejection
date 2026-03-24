"""
Integrated Scoring Pipeline (V2)
Combines Cosine Similarity + ATS Scoring (Skills/Experience/Education)
Uses V2 database schema (applicant_profiles, applicant_embeddings)
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from uuid import UUID
from decimal import Decimal

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from loguru import logger

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "workflow_1_preprocessing"))

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from database.supabase_client_v2 import get_db_client
from database.models_v2 import (
    JobDescription,
    JobDescriptionCreate,
    MatchHistoryCreate,
    MatchResultResponse
)
from embeddings.bert_embedder_v2 import get_section_embedder
from ranking.education_matcher import check_education_match, normalize_education
from ranking.zone_classifier import ZoneClassifier


class IntegratedScoringPipeline:
    """
    Integrated scoring pipeline that provides:
    1. Cosine Similarity Score (from embeddings)
    2. ATS Score (LLM-based skills/experience/education matching)
    3. Combined Final Score

    Uses V2 database schema.
    """

    # Weights for combining cosine similarity and ATS scores
    SCORE_COMBINATION_WEIGHTS = {
        "cosine_similarity": 0.40,  # 40% weight for embedding similarity
        "ats_score": 0.60           # 60% weight for ATS scoring
    }

    # ATS component weights (within the 60% ATS score)
    ATS_WEIGHTS = {
        "skills": 0.60,         # 60% of ATS score
        "experience": 0.25,     # 25% of ATS score
        "education": 0.15       # 15% of ATS score
    }

    # Skill category weights
    SKILL_WEIGHTS = {
        "must_have": 0.50,
        "good_to_have": 0.30,
        "nice_to_have": 0.20
    }

    def __init__(self, use_llm: bool = True):
        """
        Initialize the integrated scoring pipeline.

        Args:
            use_llm: Whether to use LLM for ATS scoring (requires GROQ_API_KEY)
        """
        self.db = get_db_client()
        self.embedder = get_section_embedder()
        self.use_llm = use_llm

        # Initialize LLM if available
        self.llm = None
        if use_llm:
            groq_api_key = os.getenv('GROQ_API_KEY')
            if groq_api_key:
                self.llm = ChatGroq(
                    model="llama-3.3-70b-versatile",
                    groq_api_key=groq_api_key,
                    temperature=0
                )
                logger.info("LLM initialized for ATS scoring")
            else:
                logger.warning("GROQ_API_KEY not found - ATS scoring will use cosine similarity only")
                self.use_llm = False

        logger.info("Integrated Scoring Pipeline initialized (V2)")

    # ========================================================================
    # Job Description Processing
    # ========================================================================

    def process_job_description(
        self,
        job_text: str,
        title: str,
        company: Optional[str] = None
    ) -> Tuple[UUID, Dict[str, Any]]:
        """
        Process job description and extract requirements.

        Args:
            job_text: Full job description text
            title: Job title
            company: Company name

        Returns:
            Tuple of (job_id, jd_requirements)
        """
        logger.info(f"Processing job description: {title}")

        # Create job in database
        job_data = JobDescriptionCreate(
            title=title,
            company=company,
            description=job_text,
            raw_text=job_text
        )

        job = self.db.create_job_description(job_data)
        if not job:
            raise Exception("Failed to create job description")

        job_id = job.job_id

        # Extract requirements using LLM
        jd_requirements = {}
        if self.use_llm and self.llm:
            jd_requirements = self._extract_jd_requirements_llm(job_text)

        # Generate job embedding for cosine similarity
        job_embedding = self.embedder.embed_text(job_text)

        logger.success(f"Job processed with ID: {job_id}")

        return job_id, {
            "requirements": jd_requirements,
            "embedding": job_embedding,
            "title": title,
            "company": company
        }

    def _extract_jd_requirements_llm(self, job_description: str) -> Dict[str, Any]:
        """Extract requirements from job description using LLM."""
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

        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Rate limiting
                time.sleep(0.5)

                chain = prompt | self.llm
                response = chain.invoke({"job_description": job_description})

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
                error_str = str(e).lower()
                if "429" in error_str or "rate" in error_str:
                    wait_time = (attempt + 1) * 5
                    logger.warning(f"Rate limited. Waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
                    time.sleep(wait_time)
                    continue
                logger.error(f"Error parsing LLM response: {e}")
                break

        return {
            "skills": {"must_have": [], "good_to_have": [], "nice_to_have": []},
            "experience": {"min_years": 0, "preferred_years": 0, "relevant_domains": []},
            "education": {"min_level": "bachelor", "preferred_level": "bachelor", "fields": []}
        }

    # ========================================================================
    # Applicant Scoring
    # ========================================================================

    def score_single_resume(
        self,
        resume_text: str,
        jd_data: Dict[str, Any],
        applicant_name: str = "Single Candidate"
    ) -> Dict[str, Any]:
        """
        Score a single resume against a job description.
        Does NOT require the resume to be in the database.

        Args:
            resume_text: Raw resume text
            jd_data: Job data with requirements and embedding
            applicant_name: Optional name for display

        Returns:
            Scoring result dict
        """
        logger.info(f"Scoring single resume: {applicant_name}")

        jd_embedding = jd_data.get("embedding")
        jd_requirements = jd_data.get("requirements", {})

        # Generate resume embedding
        resume_embedding = self.embedder.embed_text(resume_text)

        # Calculate cosine similarity
        cosine_score = 0.0
        if resume_embedding is not None and jd_embedding is not None:
            cosine_score = self._cosine_similarity(
                np.array(resume_embedding),
                jd_embedding
            )

        # Calculate ATS score
        ats_result = {"ats_score": 0, "component_scores": {}, "resume_profile": {}}
        if self.use_llm and self.llm:
            ats_result = self._calculate_ats_score(resume_text, jd_requirements)

        # Combined score
        ats_score = ats_result.get("ats_score", 0) / 100  # Normalize to 0-1

        if self.use_llm:
            final_score = (
                cosine_score * self.SCORE_COMBINATION_WEIGHTS["cosine_similarity"] +
                ats_score * self.SCORE_COMBINATION_WEIGHTS["ats_score"]
            )
        else:
            final_score = cosine_score

        result = {
            "name": applicant_name,
            "final_score": round(final_score * 100, 1),
            "cosine_similarity": round(cosine_score * 100, 1),
            "ats_score": round(ats_score * 100, 1),
            "skills_score": ats_result.get("component_scores", {}).get("skills", 0),
            "experience_score": ats_result.get("component_scores", {}).get("experience", 0),
            "education_score": ats_result.get("component_scores", {}).get("education", 0),
            "years_exp": ats_result.get("resume_profile", {}).get("years_experience", 0),
            "edu_level": ats_result.get("resume_profile", {}).get("education_level", ""),
            "skills_found": ats_result.get("resume_profile", {}).get("skills_found", []),
            "job_titles": ats_result.get("resume_profile", {}).get("job_titles", [])
        }

        # Classify zone
        if result["final_score"] >= 75:
            result["zone"] = "SELECTED"
        elif result["final_score"] >= 40:
            result["zone"] = "BORDERLINE"
        else:
            result["zone"] = "POOR_MATCH"

        logger.success(f"Single resume scored: {result['final_score']}/100 ({result['zone']})")

        return result

    def print_single_result(self, result: Dict[str, Any], jd_title: str = "Job"):
        """Print formatted result for a single resume."""
        zone_colors = {
            "SELECTED": "\033[92m",      # Green
            "BORDERLINE": "\033[93m",    # Yellow
            "POOR_MATCH": "\033[91m"     # Red
        }
        reset = "\033[0m"

        print("\n" + "=" * 70)
        print(f"[SINGLE RESUME SCORING RESULT]")
        print(f"   Job: {jd_title}")
        print("=" * 70)

        zone = result.get("zone", "UNKNOWN")
        color = zone_colors.get(zone, "")

        print(f"\n   Candidate: {result.get('name', 'Unknown')}")
        print(f"   Zone: {color}{zone}{reset}")
        print(f"\n   FINAL SCORE: {result.get('final_score', 0):.1f}/100")
        print(f"   ├─ Cosine Similarity: {result.get('cosine_similarity', 0):.1f}/100 (40% weight)")
        print(f"   └─ ATS Score: {result.get('ats_score', 0):.1f}/100 (60% weight)")
        print(f"       ├─ Skills: {result.get('skills_score', 0):.1f}/100")
        print(f"       ├─ Experience: {result.get('experience_score', 0):.1f}/100 ({result.get('years_exp', 0)} years)")
        print(f"       └─ Education: {result.get('education_score', 0):.1f}/100 ({result.get('edu_level', 'N/A')})")

        if result.get('skills_found'):
            print(f"\n   Skills Found: {', '.join(result['skills_found'][:10])}")

        if result.get('job_titles'):
            print(f"   Job Titles: {', '.join(result['job_titles'][:3])}")

        print("=" * 70)

    def score_all_applicants(
        self,
        job_id: UUID,
        jd_data: Dict[str, Any],
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Score all applicants against a job description.

        Args:
            job_id: Job UUID
            jd_data: Job data with requirements and embedding
            limit: Optional limit on number of applicants

        Returns:
            List of ranked applicants with scores
        """
        logger.info(f"Scoring all applicants for job {job_id}")

        # Get all applicants from V2 database
        applicants = self.db.get_all_applicants(limit=limit)

        if not applicants:
            logger.warning("No applicants found in database")
            return []

        logger.info(f"Found {len(applicants)} applicants to score")

        results = []
        jd_embedding = jd_data.get("embedding")
        jd_requirements = jd_data.get("requirements", {})

        for i, applicant in enumerate(applicants, 1):
            try:
                logger.info(f"  [{i:02d}/{len(applicants)}] Scoring {applicant.name}...")

                # Get applicant embedding
                applicant_emb = self.db.get_applicant_embedding(applicant.applicant_id)

                # Calculate cosine similarity
                cosine_score = 0.0
                if applicant_emb and applicant_emb.resume_embedding and jd_embedding is not None:
                    resume_emb = np.array(applicant_emb.resume_embedding)
                    cosine_score = self._cosine_similarity(resume_emb, jd_embedding)

                # Calculate ATS score
                ats_result = {"ats_score": 0, "component_scores": {}, "resume_profile": {}}
                if self.use_llm and self.llm and applicant.raw_text:
                    ats_result = self._calculate_ats_score(
                        applicant.raw_text,
                        jd_requirements
                    )

                # Combined score
                ats_score = ats_result.get("ats_score", 0) / 100  # Normalize to 0-1

                if self.use_llm:
                    final_score = (
                        cosine_score * self.SCORE_COMBINATION_WEIGHTS["cosine_similarity"] +
                        ats_score * self.SCORE_COMBINATION_WEIGHTS["ats_score"]
                    )
                else:
                    # If no LLM, use only cosine similarity
                    final_score = cosine_score

                # Update score in database
                self.db.update_match_score(
                    applicant.applicant_id,
                    final_score,
                    job_id
                )

                # Store match history
                match_data = MatchHistoryCreate(
                    applicant_id=applicant.applicant_id,
                    job_id=job_id,
                    overall_score=Decimal(str(round(final_score, 4))),
                    section_scores={
                        "cosine_similarity": round(cosine_score, 4),
                        "ats_score": round(ats_score, 4),
                        "skills": ats_result.get("component_scores", {}).get("skills", 0) / 100,
                        "experience": ats_result.get("component_scores", {}).get("experience", 0) / 100,
                        "education": ats_result.get("component_scores", {}).get("education", 0) / 100
                    },
                    config_name="integrated_v2",
                    weights_used={
                        **self.SCORE_COMBINATION_WEIGHTS,
                        "ats_components": self.ATS_WEIGHTS
                    }
                )
                self.db.create_match_history(match_data)

                results.append({
                    "applicant_id": str(applicant.applicant_id),
                    "name": applicant.name,
                    "email": applicant.email,
                    "final_score": round(final_score * 100, 1),
                    "cosine_similarity": round(cosine_score * 100, 1),
                    "ats_score": round(ats_score * 100, 1),
                    "skills_score": ats_result.get("component_scores", {}).get("skills", 0),
                    "experience_score": ats_result.get("component_scores", {}).get("experience", 0),
                    "education_score": ats_result.get("component_scores", {}).get("education", 0),
                    "years_exp": ats_result.get("resume_profile", {}).get("years_experience", 0),
                    "edu_level": ats_result.get("resume_profile", {}).get("education_level", ""),
                    "skills_found": ats_result.get("resume_profile", {}).get("skills_found", [])
                })

            except Exception as e:
                logger.error(f"Error scoring applicant {applicant.applicant_id}: {e}")
                continue

        # Sort by final score (descending)
        results.sort(key=lambda x: x["final_score"], reverse=True)

        # Add rank
        for rank, result in enumerate(results, 1):
            result["rank"] = rank

        logger.success(f"Scored {len(results)} applicants successfully")

        return results

    def _calculate_ats_score(
        self,
        resume_text: str,
        jd_requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate ATS score using LLM extraction."""

        # Extract resume profile
        resume_profile = self._extract_resume_profile_llm(resume_text)

        resume_skills = resume_profile.get("skills", [])
        resume_experience = resume_profile.get("experience", {})
        resume_education = resume_profile.get("education", {})

        jd_skills = jd_requirements.get("skills", {})
        jd_experience = jd_requirements.get("experience", {})
        jd_education = jd_requirements.get("education", {})

        # 1. Skills scoring (60% of ATS)
        skills_score = self._compute_skills_score(jd_skills, resume_skills, resume_text)

        # 2. Experience scoring (25% of ATS)
        experience_score = self._compute_experience_score(jd_experience, resume_experience)

        # 3. Education scoring (15% of ATS)
        education_score = self._compute_education_score(jd_education, resume_education)

        # Final ATS score
        final_ats_score = (
            skills_score * self.ATS_WEIGHTS["skills"] +
            experience_score * self.ATS_WEIGHTS["experience"] +
            education_score * self.ATS_WEIGHTS["education"]
        )

        return {
            "ats_score": round(final_ats_score * 100, 1),
            "component_scores": {
                "skills": round(skills_score * 100, 1),
                "experience": round(experience_score * 100, 1),
                "education": round(education_score * 100, 1)
            },
            "resume_profile": {
                "skills_found": resume_skills[:15],
                "years_experience": resume_experience.get("total_years", 0),
                "education_level": resume_education.get("highest_level", ""),
                "job_titles": resume_experience.get("job_titles", [])[:3]
            }
        }

    def _extract_resume_profile_llm(self, resume_text: str) -> Dict[str, Any]:
        """Extract profile from resume using LLM."""
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

        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Rate limiting - wait before LLM call to avoid 429 errors
                time.sleep(0.5)  # 500ms delay between calls

                chain = prompt | self.llm
                response = chain.invoke({"resume_text": resume_text[:5000]})

                content = response.content
                start = content.find('{')
                end = content.rfind('}') + 1
                json_str = content[start:end]
                profile = json.loads(json_str)

                # Ensure all keys exist
                if "skills" not in profile:
                    profile["skills"] = []
                if "experience" not in profile:
                    profile["experience"] = {"total_years": 0, "job_titles": [], "domains": []}
                if "education" not in profile:
                    profile["education"] = {"highest_level": "bachelor", "field": "", "degrees": []}

                return profile

            except Exception as e:
                error_str = str(e).lower()
                if "429" in error_str or "rate" in error_str:
                    wait_time = (attempt + 1) * 5  # Exponential backoff: 5s, 10s, 15s
                    logger.warning(f"Rate limited. Waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
                    time.sleep(wait_time)
                    continue
                logger.error(f"Error parsing resume profile: {e}")
                break

        return {
            "skills": [],
            "experience": {"total_years": 0, "job_titles": [], "domains": []},
            "education": {"highest_level": "bachelor", "field": "", "degrees": []}
        }

    def _compute_skills_score(
        self,
        jd_skills: Dict[str, List[str]],
        resume_skills: List[str],
        resume_text: str
    ) -> float:
        """Compute skills match score."""
        category_scores = {}

        for category in ["must_have", "good_to_have", "nice_to_have"]:
            requirements = jd_skills.get(category, [])
            if not requirements:
                category_scores[category] = 1.0
                continue

            # Create embeddings for comparison
            resume_skills_text = " ".join(resume_skills) if resume_skills else resume_text[:2000]
            resume_emb = self.embedder.embed_text(resume_skills_text)

            total_score = 0
            for req in requirements:
                req_emb = self.embedder.embed_text(req)
                similarity = self._cosine_similarity(req_emb, resume_emb)

                # Keyword bonus
                req_lower = req.lower()
                keyword_bonus = 0
                for skill in resume_skills:
                    if isinstance(skill, str) and skill.lower() in req_lower:
                        keyword_bonus = 0.15
                        break

                total_score += min(1.0, similarity + keyword_bonus)

            category_scores[category] = total_score / len(requirements)

        # Weighted average
        return sum(
            category_scores.get(cat, 0) * weight
            for cat, weight in self.SKILL_WEIGHTS.items()
        )

    def _compute_experience_score(
        self,
        jd_experience: Dict[str, Any],
        resume_experience: Dict[str, Any]
    ) -> float:
        """Compute experience match score."""
        min_years = jd_experience.get("min_years", 0)
        preferred_years = jd_experience.get("preferred_years", min_years)
        candidate_years = resume_experience.get("total_years", 0)

        # Years scoring (40%)
        if min_years == 0:
            years_score = 1.0
        elif candidate_years >= preferred_years:
            years_score = 1.0
        elif candidate_years >= min_years:
            years_score = 0.7 + 0.3 * ((candidate_years - min_years) / max(1, preferred_years - min_years))
        elif candidate_years >= min_years * 0.7:
            years_score = 0.4 + 0.3 * (candidate_years / max(1, min_years))
        else:
            years_score = max(0.1, candidate_years / max(1, min_years))

        # Domain relevance (60%)
        jd_domains = jd_experience.get("relevant_domains", [])
        resume_domains = resume_experience.get("domains", [])
        resume_titles = resume_experience.get("job_titles", [])

        if not jd_domains:
            domain_score = 1.0
        else:
            resume_domain_text = " ".join(resume_domains + resume_titles)
            if not resume_domain_text:
                domain_score = 0.3
            else:
                resume_emb = self.embedder.embed_text(resume_domain_text)
                jd_emb = self.embedder.embed_text(" ".join(jd_domains))
                domain_score = self._cosine_similarity(resume_emb, jd_emb)

        return (years_score * 0.4) + (domain_score * 0.6)

    def _compute_education_score(
        self,
        jd_education: Dict[str, Any],
        resume_education: Dict[str, Any]
    ) -> float:
        """Compute education match score."""
        min_level = jd_education.get("min_level", "bachelor")
        preferred_level = jd_education.get("preferred_level", min_level)
        candidate_level = resume_education.get("highest_level", "bachelor")

        # Use education matcher
        matches_min, min_score = check_education_match(candidate_level, min_level)
        matches_pref, pref_score = check_education_match(candidate_level, preferred_level)

        if matches_pref:
            level_score = min(pref_score / 100, 1.0)
        elif matches_min:
            level_score = 0.7 + 0.3 * (min_score / 100)
        else:
            level_score = max(0.2, min_score / 100)

        # Field relevance (50%)
        jd_fields = jd_education.get("fields", [])
        candidate_field = resume_education.get("field", "")
        candidate_degrees = resume_education.get("degrees", [])

        if not jd_fields:
            field_score = 1.0
        else:
            candidate_edu_text = f"{candidate_field} " + " ".join(candidate_degrees)
            if not candidate_edu_text.strip():
                field_score = 0.3
            else:
                candidate_emb = self.embedder.embed_text(candidate_edu_text)
                jd_emb = self.embedder.embed_text(" ".join(jd_fields))
                field_score = self._cosine_similarity(candidate_emb, jd_emb)

        return (level_score * 0.5) + (field_score * 0.5)

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        if vec1 is None or vec2 is None:
            return 0.0

        vec1 = np.array(vec1).reshape(1, -1)
        vec2 = np.array(vec2).reshape(1, -1)

        similarity = cosine_similarity(vec1, vec2)[0][0]
        return float(np.clip(similarity, 0.0, 1.0))

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def get_job_descriptions(self) -> List[Dict[str, Any]]:
        """Fetch all job descriptions from database."""
        result = self.db.client.table("job_descriptions").select(
            "job_id, title, company, description, raw_text, is_active"
        ).eq("is_active", True).execute()

        return result.data if result.data else []

    def print_rankings(self, results: List[Dict[str, Any]], top_k: int = 15):
        """Print formatted ranking results."""
        print("\n" + "=" * 100)
        print("[RANKING] INTEGRATED SCORING RESULTS")
        print("       Scoring: Cosine (40%) + ATS (60%) [Skills 60% + Exp 25% + Edu 15%]")
        print("=" * 100)

        header = f"{'Rank':<5} {'Name':<25} {'Final':<8} {'Cosine':<8} {'ATS':<7} {'Skills':<8} {'Exp':<7} {'Edu':<7} {'Yrs':<5}"
        print(header)
        print("-" * 100)

        for r in results[:top_k]:
            edu_display = r.get('edu_level', '')[:6] if r.get('edu_level') else "N/A"
            name_display = r.get('name', 'Unknown')[:24]
            print(
                f"{r.get('rank', '-'):<5} "
                f"{name_display:<25} "
                f"{r.get('final_score', 0):<8.1f} "
                f"{r.get('cosine_similarity', 0):<8.1f} "
                f"{r.get('ats_score', 0):<7.1f} "
                f"{r.get('skills_score', 0):<8.1f} "
                f"{r.get('experience_score', 0):<7.1f} "
                f"{r.get('education_score', 0):<7.1f} "
                f"{r.get('years_exp', 0):<5}"
            )

        print("=" * 100)

        # Top candidate details
        if results:
            top = results[0]
            print(f"\n[TOP CANDIDATE] {top.get('name', 'Unknown')}")
            print(f"   Final Score: {top.get('final_score', 0):.1f}/100")
            print(f"   Cosine Similarity: {top.get('cosine_similarity', 0):.1f}/100")
            print(f"   ATS Score: {top.get('ats_score', 0):.1f}/100")
            print(f"   - Skills: {top.get('skills_score', 0):.1f}/100")
            print(f"   - Experience: {top.get('experience_score', 0):.1f}/100 ({top.get('years_exp', 0)} years)")
            print(f"   - Education: {top.get('education_score', 0):.1f}/100 ({top.get('edu_level', 'N/A')})")
            if top.get('skills_found'):
                print(f"   Top Skills: {', '.join(top['skills_found'][:8])}")

    # ========================================================================
    # Zone Classification
    # ========================================================================

    def classify_candidates(
        self,
        results: List[Dict[str, Any]],
        selected_threshold: float = 75.0,
        borderline_threshold: float = 40.0
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Classify candidates into zones based on their scores.

        Args:
            results: List of scored candidates
            selected_threshold: Minimum score for SELECTED zone (default: 75)
            borderline_threshold: Minimum score for BORDERLINE zone (default: 40)

        Returns:
            Dict with 'selected', 'borderline', 'poor_match' lists
        """
        classifier = ZoneClassifier(
            selected_threshold=selected_threshold,
            borderline_threshold=borderline_threshold
        )

        # Prepare candidates for classification (use final_score)
        candidates_for_classification = []
        for r in results:
            candidate = r.copy()
            candidate['score'] = r.get('final_score', 0)  # ZoneClassifier looks for 'score'
            candidates_for_classification.append(candidate)

        classified = classifier.batch_classify(candidates_for_classification)

        return classified

    def print_zone_classification(
        self,
        classified_results: Dict[str, List[Dict[str, Any]]]
    ) -> None:
        """Print zone classification report."""
        classifier = ZoneClassifier()
        classifier.print_classification_report(classified_results)

    def get_borderline_candidates(
        self,
        classified_results: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Get borderline candidates (for feedback generation)."""
        return classified_results.get("borderline", [])


def main():
    """Main entry point for integrated scoring."""
    print("=" * 70)
    print("[*] INTEGRATED SCORING PIPELINE (V2)")
    print("   Combines: Cosine Similarity (40%) + ATS Score (60%)")
    print("   ATS: Skills (60%) + Experience (25%) + Education (15%)")
    print("=" * 70)

    # Initialize pipeline
    pipeline = IntegratedScoringPipeline(use_llm=True)

    # Check for existing job descriptions
    existing_jobs = pipeline.get_job_descriptions()

    if existing_jobs:
        print(f"\nFound {len(existing_jobs)} existing job description(s)")
        jd_record = existing_jobs[0]
        print(f"Using: {jd_record.get('title', 'N/A')} @ {jd_record.get('company', 'N/A')}")

        # Extract requirements from existing JD
        job_id = UUID(jd_record['job_id'])
        jd_data = {
            "requirements": pipeline._extract_jd_requirements_llm(jd_record['raw_text']) if pipeline.use_llm else {},
            "embedding": pipeline.embedder.embed_text(jd_record['raw_text']),
            "title": jd_record.get('title'),
            "company": jd_record.get('company')
        }
    else:
        print("\nNo job descriptions found. Please provide one.")
        print("Example: python integrated_scoring.py --jd 'path/to/job_description.txt'")
        return

    # Print JD requirements
    if jd_data.get("requirements"):
        req = jd_data["requirements"]
        print(f"\n[JD REQUIREMENTS]")
        print(f"  Skills: Must-Have={len(req.get('skills', {}).get('must_have', []))}, "
              f"Good-to-Have={len(req.get('skills', {}).get('good_to_have', []))}, "
              f"Nice-to-Have={len(req.get('skills', {}).get('nice_to_have', []))}")
        print(f"  Experience: {req.get('experience', {}).get('min_years', 0)}-{req.get('experience', {}).get('preferred_years', 0)} years")
        print(f"  Education: {req.get('education', {}).get('min_level', 'N/A')} - {req.get('education', {}).get('preferred_level', 'N/A')}")

    # Score all applicants
    print("\n[SCORING] Processing applicants...")
    results = pipeline.score_all_applicants(job_id, jd_data)

    if not results:
        print("No applicants found or scored.")
        return results

    # Print rankings
    pipeline.print_rankings(results)

    # Zone classification
    print("\n[ZONE CLASSIFICATION]")
    classified = pipeline.classify_candidates(results)
    pipeline.print_zone_classification(classified)

    # Get borderline candidates for feedback (for later use)
    borderline = pipeline.get_borderline_candidates(classified)
    if borderline:
        print(f"\n[INFO] {len(borderline)} borderline candidates identified for feedback emails")

    return results, classified


if __name__ == "__main__":
    main()
