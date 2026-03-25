"""
Prompt templates for feedback email generation.
"""

FEEDBACK_SYSTEM_PROMPT = """You are a technical recruiter analyzing resumes and providing personalized feedback.

IMPORTANT: You must carefully read BOTH the resume AND job requirements, then identify what is SPECIFICALLY missing for THIS candidate. Do NOT give generic feedback - each candidate is different."""

FEEDBACK_EMAIL_TEMPLATE = """Analyze this candidate's resume against the job requirements and write personalized feedback.

=== CANDIDATE'S RESUME ===
{resume_context}
=== END RESUME ===

=== JOB REQUIREMENTS ===
{job_requirements}
=== END JOB REQUIREMENTS ===

Candidate Name: {candidate_name}
Position: {job_title}

STEP 1 - First, identify what technologies/skills THIS candidate HAS from their resume:
(List 3-5 specific technologies you found in their resume)

STEP 2 - Compare against job requirements and identify what's MISSING:
(List specific technologies from the JD that are NOT in the resume)

STEP 3 - Now write the email using your analysis above:

Dear {candidate_name},

Thank you for applying to the {job_title} position. Your experience with [MENTION 1-2 SPECIFIC SKILLS FROM STEP 1] shows a solid foundation.

To strengthen your application for similar roles, we recommend gaining experience with:
- [FIRST MISSING SKILL FROM STEP 2]
- [SECOND MISSING SKILL FROM STEP 2]
- [THIRD MISSING SKILL FROM STEP 2 IF ANY]

Consider: [Relevant certification or course based on the missing skills]

We encourage you to apply again once you've built experience in these areas.

Best regards,
The Hiring Team

IMPORTANT:
- Your STEP 1 and STEP 2 analysis must be specific to THIS resume
- The skills you mention must come from YOUR analysis, not defaults
- Do NOT always say "Kubernetes, TensorFlow" - analyze what's actually missing
- If candidate has FastAPI but lacks Docker, say Docker not FastAPI
- Output ONLY the final email (Dear... to ...Hiring Team), not the analysis steps"""

SKILLS_COMPARISON_PROMPT = """Analyze the skills gap between the candidate's resume and the job requirements.

CANDIDATE SKILLS AND EXPERIENCE:
{resume_context}

JOB REQUIREMENTS:
{job_requirements}

Provide a brief analysis:
1. Matching Skills: List skills the candidate has that match the job
2. Missing Skills: List key skills required by the job that the candidate lacks
3. Partially Matching: Skills that are related but need strengthening

Format as JSON:
{{
    "matching_skills": ["skill1", "skill2"],
    "missing_skills": ["skill1", "skill2"],
    "partial_skills": ["skill1 (reason)", "skill2 (reason)"]
}}"""
