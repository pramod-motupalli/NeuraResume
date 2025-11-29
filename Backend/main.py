import os
import json
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import uvicorn
import io
from pypdf import PdfReader

# Load environment variables
load_dotenv()

# Import agents (will be created in agents.py)
from agents import run_ats_analyzer, run_ats_optimizer, run_interview_coach, run_interview_answer_generator, LLM_MODEL


print(f"Loaded LLM Model: {LLM_MODEL}")

app = FastAPI(title="NeuraResume Backend")


# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "NeuraResume API is running"}

@app.post("/analyze")
async def analyze_resume(
    resumeText: Optional[str] = Form(None),
    jobDescription: Optional[str] = Form(None),
    tasks: str = Form(...),
    resumeFile: Optional[UploadFile] = File(None)
):
    """
    Analyze resume based on requested tasks. Accepts text or PDF file.
    """
    try:
        # Parse tasks JSON
        try:
            tasks_dict = json.loads(tasks)
            print(f"Received tasks: {tasks_dict}")
        except json.JSONDecodeError:
            print("Error decoding tasks JSON")
            raise HTTPException(status_code=400, detail="Invalid JSON for tasks")

        final_resume_text = ""

        # Handle File Upload (PDF)
        if resumeFile:
            print(f"Received file: {resumeFile.filename}, content_type: {resumeFile.content_type}")
            if resumeFile.content_type == "application/pdf":
                try:
                    contents = await resumeFile.read()
                    pdf_stream = io.BytesIO(contents)
                    reader = PdfReader(pdf_stream)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text() + "\n"
                    final_resume_text = text
                    print(f"Extracted text length: {len(final_resume_text)}")
                except Exception as e:
                    print(f"Error reading PDF: {e}")
                    raise HTTPException(status_code=400, detail=f"Error reading PDF: {str(e)}")
            else:
                # Fallback for other files if needed, or error
                raise HTTPException(status_code=400, detail="Only PDF files are supported for upload.")
        
        # Fallback to text input if no file or file failed (though we raised error above)
        if not final_resume_text and resumeText:
            final_resume_text = resumeText
            print(f"Received resume text length: {len(final_resume_text)}")

        if not final_resume_text.strip():
             print("No resume text provided")
             raise HTTPException(status_code=400, detail="No resume text provided (either via file or text input).")

        results = {
            "atsAnalyzer": None,
            "atsOptimizer": None,
            "interviewCoach": None
        }

        if tasks_dict.get("runAtsAnalyzer"):
            print("Running ATS Analyzer...")
            results["atsAnalyzer"] = await run_ats_analyzer(final_resume_text, jobDescription)
            print("ATS Analyzer done.")

        if tasks_dict.get("runAtsOptimizer"):
            print("Running ATS Optimizer...")
            results["atsOptimizer"] = await run_ats_optimizer(final_resume_text, jobDescription)
            print("ATS Optimizer done.")

        if tasks_dict.get("runInterviewCoach"):
            print("Running Interview Coach...")
            results["interviewCoach"] = await run_interview_coach(final_resume_text, jobDescription)
            print("Interview Coach done.")

        return results

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Internal Server Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class GenerateAnswersRequest(BaseModel):
    resumeText: str
    jobDescription: Optional[str] = None
    questions: List[dict]

@app.post("/generate-answers")
async def generate_answers(request: GenerateAnswersRequest):
    """
    Generate answers for the provided questions.
    """
    try:
        print("Generating answers...")
        # Extract just the question text and ID for the agent to save tokens
        simplified_questions = [{"id": q.get("id"), "question": q.get("question")} for q in request.questions]
        
        result = await run_interview_answer_generator(
            request.resumeText, 
            request.jobDescription, 
            simplified_questions
        )
        return result
    except Exception as e:
        print(f"Error generating answers: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
