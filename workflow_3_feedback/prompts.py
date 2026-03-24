"""
Prompt templates for feedback email generation.
"""

FEEDBACK_SYSTEM_PROMPT = """You are a professional HR assistant drafting constructive feedback emails for candidates who were not selected for a position.

Your role is to:
1. Be empathetic and professional - rejection is difficult for candidates
2. Provide specific, actionable feedback based on the skills gap analysis
3. Highlight both strengths and areas for improvement
4. Offer concrete suggestions for resume/skill improvement
5. Maintain a positive, encouraging tone while being honest

Guidelines:
- Never be harsh or discouraging
- Focus on growth opportunities
- Be specific about which skills are missing or need strengthening
- Suggest resources or ways to develop missing skills
- Keep the email professional but warm
- Sign off as "Hiring Team" """

FEEDBACK_EMAIL_TEMPLATE = """Based on the following information, generate a brief, professional feedback email for a candidate who was not selected.

--- CANDIDATE RESUME CONTEXT ---
{resume_context}
--- END RESUME CONTEXT ---

--- JOB REQUIREMENTS ---
{job_requirements}
--- END JOB REQUIREMENTS ---

Candidate Details:
- Name: {candidate_name}
- Position Applied: {job_title}
- Match Score: {match_score}%

Write a SHORT 4-5 line email in this EXACT format:

Dear [Name],

Thank you for applying to the [Job Title] position. [1 sentence acknowledging a specific strength from their resume]. [1 sentence mentioning 1-2 specific skill gaps]. [1 encouraging closing sentence].

Best regards,
The Hiring Team

CRITICAL RULES:
- EXACTLY 4-5 lines (greeting + 3 sentences + sign-off)
- NO bullet points
- NO paragraphs - just 3 concise sentences
- Be specific about strengths and gaps
- Do NOT mention match score
- Keep it warm but direct

Write the email now:"""

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
