import os
import json
import re
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile") # Default fallback


if not LLM_API_KEY:
    print("Warning: LLM_API_KEY not found in environment variables.")

# Initialize OpenAI client (works with Groq, Together, Ollama etc.)
client = AsyncOpenAI(
    api_key=LLM_API_KEY,
    base_url=LLM_BASE_URL
)

def clean_json_text(text: str) -> str:
    """Removes markdown code blocks and whitespace to extract JSON."""
    text = text.strip()
    # Remove ```json ... ``` or just ``` ... ```
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()

async def generate_json(prompt: str) -> dict:
    """Helper to generate JSON from LLM."""
    try:
        response = await client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant that outputs ONLY valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=4096,
            response_format={"type": "json_object"} # Enforce JSON mode if supported
        )
        content = response.choices[0].message.content
        cleaned_text = clean_json_text(content)
        return json.loads(cleaned_text)
    except Exception as e:
        print(f"Error generating JSON: {e}")
        # Fallback: try to parse without strict json_object mode if it failed
        return None

async def run_ats_analyzer(resume_text: str, job_description: str = None) -> dict:
    prompt = f"""
    You are an ATS_ANALYZER_AGENT.
    Goal: Parse the resume and estimate an ATS score.
    
    Resume Text:
    {resume_text}
    
    Job Description:
    {job_description if job_description else "Not provided. Infer role from resume."}
    
    Output must be a JSON object with the exact structure:
    {{
      "parsedResume": {{
        "name": "<string>",
        "contact": {{ "email": "...", "phone": "...", "location": "...", "linkedin": "...", "portfolio": "..." }},
        "summary": "...",
        "skills": [ {{ "name": "...", "category": "...", "proficiencyLevel": "..." }} ],
        "experience": [ {{ "title": "...", "company": "...", "startDate": "...", "endDate": "...", "descriptionBullets": ["..."] }} ],
        "projects": [ {{ "name": "...", "role": "...", "descriptionBullets": ["..."], "technologies": ["..."] }} ],
        "education": [ {{ "degree": "...", "institution": "...", "startYear": "...", "endYear": "..." }} ],
        "certifications": [ {{ "name": "...", "issuer": "...", "year": "..." }} ],
        "extraSections": []
      }},
      "atsScore": {{
        "score": <0-100>,
        "scoreBreakdown": {{ "keywordMatch": <0-100>, "sectionStructure": <0-100>, "readability": <0-100>, "roleAlignment": <0-100> }},
        "summary": "..."
      }},
      "jobSuitability": {{
        "match": "<High | Medium | Low>",
        "percentage": <0-100>,
        "reasoning": "..."
      }},
      "careerSuggestions": {{
        "recommendedRoles": ["<Role 1>", "<Role 2>", "<Role 3>"],
        "marketOutlook": "<Description of current demand and typical openings for these roles>",
        "topCompaniesToTarget": ["<Company 1>", "<Company 2>"]
      }},
      "resumePersona": {{
        "tone": "<e.g. Leader, Doer, Academic, Creative>",
        "impression": "<short description of the vibe>"
      }},
      "salaryEstimation": {{
        "range": "<e.g. $80k - $100k or ₹10L - ₹15L>",
        "currency": "<inferred from location>"
      }},
      "keywordAnalysis": {{
        "jobRoleInferred": "...",
        "matchedKeywords": ["..."],
        "missingImportantKeywords": ["..."],
        "niceToHaveKeywords": ["..."]
      }},
      "strengths": ["..."],
      "weaknesses": ["..."]
    }}
    
    Return ONLY valid JSON. Do not include any other text.
    """
    return await generate_json(prompt)

async def run_ats_optimizer(resume_text: str, job_description: str = None) -> dict:
    prompt = f"""
    You are an ATS_OPTIMIZER_AGENT.
    Goal: Suggest improvements to increase ATS score.
    
    Resume Text:
    {resume_text}
    
    Job Description:
    {job_description if job_description else "Not provided."}
    
    Output must be a JSON object with the exact structure:
    {{
      "overallStrategy": "...",
      "sectionLevelSuggestions": [
        {{ "section": "...", "issue": "...", "suggestion": "...", "exampleRewrite": "..." }}
      ],
      "keywordSuggestions": {{
        "addTheseKeywords": [ {{ "keyword": "...", "reason": "...", "whereToAdd": "..." }} ],
        "removeOrReduceTheseKeywords": [ {{ "keyword": "...", "reason": "..." }} ]
      }},
      "skillGapLearningPath": [
        {{ "skill": "<missing skill>", "learningTopics": ["<topic1>", "<topic2>"] }}
      ],
      "formattingAndStructureTips": ["..."],
      "estimatedImprovedAtsScore": {{ "score": <0-100>, "assumptions": "..." }}
    }}
    
    Return ONLY valid JSON.
    """
    return await generate_json(prompt)

async def run_interview_coach(resume_text: str, job_description: str = None) -> dict:
    prompt = f"""
    You are an INTERVIEW_COACH_AGENT.
    Goal: Generate 30 interview questions (10 Easy, 10 Medium, 10 Hard).
    
    Resume Text:
    {resume_text}
    
    Job Description:
    {job_description if job_description else "Not provided."}
    
    Output must be a JSON object with the exact structure:
    {{
      "targetRole": "...",
      "difficultyDistribution": {{ "easy": 10, "medium": 10, "hard": 10 }},
      "questions": [
        {{ "id": "Q1", "difficulty": "Easy", "category": "...", "question": "...", "basedOn": {{ "resumeSection": "...", "keywords": ["..."] }}, "followUpHint": "..." }}
      ]
    }}
    
    Return ONLY valid JSON.
    """
    return await generate_json(prompt)

async def run_interview_answer_generator(resume_text: str, job_description: str, questions: list) -> dict:
    questions_str = json.dumps(questions)
    prompt = f"""
    You are an INTERVIEW_PREP_EXPERT.
    Goal: Generate model answers for the provided interview questions, tailored to the candidate's resume.
    
    Resume Text:
    {resume_text}
    
    Job Description:
    {job_description if job_description else "Not provided."}
    
    Questions:
    {questions_str}
    
    Output must be a JSON object with the exact structure:
    {{
      "answers": [
        {{ "questionId": "<id from input>", "question": "...", "answer": "<STAR method answer or technical explanation>" }}
      ]
    }}
    
    Return ONLY valid JSON.
    """
    return await generate_json(prompt)

