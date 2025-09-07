# llm_engine.py
import re
import difflib
from collections import Counter
import os
import requests
import json

# ----------------------------
# Base role keywords (expandable)
# ----------------------------
ROLE_KEYWORDS = {
    "data scientist": [
        "python", "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch",
        "machine learning", "data analysis", "sql", "statistics", "nlp"
    ],
    "backend developer": [
        "python", "java", "node", "node.js", "express", "spring", "django", "flask",
        "rest api", "api", "postgres", "postgresql", "mysql", "mongodb", "sql",
        "docker", "kubernetes", "aws", "azure", "git", "ci/cd", "redis"
    ],
    "product manager": [
        "roadmap", "stakeholder", "requirements", "agile", "kanban", "metrics",
        "prioritization", "product strategy"
    ],
}

# ----------------------------
# Synonyms / normalization
# ----------------------------
NORMALIZE = {
    "nodejs": "node",
    "node.js": "node",
    "postgre": "postgresql",
    "postgres": "postgresql",
    "postgresql": "postgresql",
    "py": "python",
    "tf": "tensorflow",
    "torch": "pytorch",
    "nlp": "nlp",
    "rest": "rest api",
    "restapi": "rest api",
    "ci/cd": "ci/cd",
    "k8s": "kubernetes",
    "db": "database",
    "mysql": "mysql",
    "mongo": "mongodb",
}

# ----------------------------
# Helper functions
# ----------------------------
def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^\w\s\-/+\.]", " ", text)
    return [t.strip() for t in text.split() if t.strip()]

def normalize_token(tok):
    return NORMALIZE.get(tok.lower(), tok.lower())

def extract_jd_keywords(job_description, top_n=20):
    if not job_description:
        return []
    tokens = tokenize(job_description)
    freq = Counter([normalize_token(t) for t in tokens if len(t) > 2])
    return [w for w, _ in freq.most_common(top_n)]

def has_keyword_in_text(keyword, resume_tokens):
    kw_norm = normalize_token(keyword)
    if keyword.lower() in resume_tokens["raw_text_lower"]:
        return True
    for tok in resume_tokens["tokens"]:
        tok_norm = normalize_token(tok)
        if tok_norm == kw_norm:
            return True
        if difflib.SequenceMatcher(a=tok_norm, b=kw_norm).ratio() > 0.85:
            return True
    return False

def build_resume_tokens(resume_text):
    return {
        "raw_text_lower": resume_text.lower(),
        "tokens": tokenize(resume_text)
    }

# ----------------------------
# Rule-based Resume Review
# ----------------------------
def review_resume(resume_text, job_role, job_description=None):
    if not resume_text or not job_role:
        return 0, "No resume text or job role provided."

    role_key = job_role.strip().lower()
    default_keywords = ROLE_KEYWORDS.get(role_key, [])
    jd_keywords = extract_jd_keywords(job_description or "", top_n=30)
    merged_keywords = list({normalize_token(k) for k in default_keywords + jd_keywords})

    SECTIONS = {
        "education": ["education", "educational background", "qualifications"],
        "experience": ["experience", "work experience", "professional experience"],
        "skills": ["skills", "technical skills", "technologies"],
        "projects": ["project", "projects", "personal projects"],
        "certifications": ["certificate", "certifications", "credentials"],
        "internship": ["internship", "internships", "training"]
    }

    tokens_struct = build_resume_tokens(resume_text)
    score, strengths, weaknesses = 0, [], []
    matched_keywords, missing_keywords = [], []

    # Section detection
    for sec, variants in SECTIONS.items():
        found = any(re.search(r"\b" + re.escape(v) + r"\b", tokens_struct["raw_text_lower"]) for v in variants)
        if found:
            score += 8
            strengths.append(f"Found section: {sec.capitalize()}")
        else:
            weaknesses.append(f"Missing or weak section: {sec.capitalize()}")

    # Keyword matching
    role_norm_set = {normalize_token(k) for k in default_keywords}
    for kw in merged_keywords:
        if len(kw) <= 2:
            continue
        found = has_keyword_in_text(kw, tokens_struct)
        if found:
            matched_keywords.append(kw)
            score += 6 if kw in role_norm_set else 2
        else:
            missing_keywords.append(kw)

    # Achievements detection
    nums = re.findall(r"\b\d{1,3}%|\b\d{2,6}\b", resume_text)
    if nums:
        score += 10
        strengths.append("Resume includes measurable achievements (numbers/percentages).")
    else:
        weaknesses.append("No measurable achievements found — add metrics.")

    # Word count
    wc = len(tokens_struct["tokens"])
    if wc >= 400:
        score += 15
    elif wc >= 200:
        score += 8
    elif wc >= 100:
        score += 4
    else:
        weaknesses.append("Resume is short — add more detail about projects and impact.")

    score = min(max(int(score), 0), 100)

    # Feedback construction
    feedback = []
    feedback.append(f"**Role evaluated:** {job_role}")
    feedback.append(f"**Resume Score:** {score}/100\n")

    if strengths:
        feedback.append("**Strengths:**")
        for s in strengths:
            feedback.append("- " + s)
        feedback.append("")

    if weaknesses:
        feedback.append("**Areas to Improve:**")
        for w in weaknesses:
            feedback.append("- " + w)
        feedback.append("")

    if matched_keywords:
        feedback.append("**Matched Keywords/Skills:**")
        feedback.append("- " + ", ".join(sorted(set(matched_keywords))) + "\n")

    if missing_keywords:
        missing_role = [k for k in missing_keywords if k in role_norm_set]
        missing_jd = [k for k in missing_keywords if k not in role_norm_set]
        top_missing = missing_role[:10] + missing_jd[:10]
        if top_missing:
            feedback.append("**Important Missing Keywords (consider adding):**")
            feedback.append("- " + ", ".join(top_missing))
            feedback.append("")

    suggestions = [
        "Add measurable outcomes to your bullet points.",
        "Highlight exact technologies used in projects (frameworks, DB, cloud).",
        "Include a short Projects section with 2–3 bullets describing impact and tech stack.",
        "Tailor the resume summary and skills to match job description keywords."
    ]
    feedback.append("**Suggestions:**")
    for s in suggestions:
        feedback.append("- " + s)

    return score, "\n".join(feedback)

# ----------------------------
# AI-based Resume Review using OpenRouter
# ----------------------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = "https://openrouter.ai/api/v1"
HEADERS = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}

def ai_review_resume(resume_text, job_role, job_description=None):
    if not OPENROUTER_API_KEY:
        return "API key not found.", 0

    prompt = f"""
You are a resume reviewer. Evaluate the following resume for the role: {job_role}.
Resume:
{resume_text}

Optional Job Description:
{job_description if job_description else 'N/A'}

Give detailed feedback with strengths, weaknesses, and suggestions.
"""

    data = {
        "model": "mistral",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 800,
        "temperature": 0.7
    }

    try:
        response = requests.post(f"{BASE_URL}/chat/completions", headers=HEADERS, data=json.dumps(data))
        response.raise_for_status()
        result = response.json()
        ai_feedback = result["choices"][0]["message"]["content"]
        return ai_feedback, 100
    except Exception as e:
        return f"AI Review Failed: {e}", 0
