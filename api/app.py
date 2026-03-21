from fastapi import FastAPI, UploadFile, File
from typing import List
import os
import json
from dotenv import load_dotenv
from resume_model.text_extractor import extract_text_and_links
from resume_model.resume_api_integration import call_gemini_api
from resume_model.jd_api_integration import call_jd_gemini_api
from resume_model.llm_fit_scorer import call_llm_fit_scorer

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

app = FastAPI()


@app.post("/evaluate_resumes/")
async def evaluate_resumes(
    jd_file: UploadFile = File(...),
    resumes: List[UploadFile] = File(...)
):
    # Step 1: Parse JD
    jd_text = (await jd_file.read()).decode("utf-8")
    jd_json_str = call_jd_gemini_api(jd_text, api_key)
    jd_json = json.loads(jd_json_str)

    # Step 2: Parse all resumes
    parsed_resumes = []
    for resume_file in resumes:
        content = await resume_file.read()
        temp_path = f"/tmp/{resume_file.filename}"
        with open(temp_path, "wb") as f:
            f.write(content)
        text, _ = extract_text_and_links(temp_path)
        parsed_json_str = call_gemini_api(text, resume_file.filename, api_key)
        try:
            parsed_json = json.loads(parsed_json_str)
        except Exception:
            parsed_json = {"candidate_id": resume_file.filename, "error": "Failed to parse resume"}
        parsed_resumes.append(parsed_json)
        os.remove(temp_path)

    # Step 3: Comparative scoring — single LLM call for all resumes
    results = call_llm_fit_scorer(parsed_resumes, jd_json, api_key)

    return results


@app.get("/")
def read_root():
    return {"message": "Resume Screening API is running."}