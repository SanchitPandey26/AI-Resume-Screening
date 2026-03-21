# AI Resume Screening System

An AI-powered resume screening system that evaluates and ranks multiple candidates against a job description — delivering structured insights with zero manual effort.

## The Problem

Recruiters receive dozens of resumes for every open role. Manually reading, comparing, and shortlisting candidates is slow, inconsistent, and prone to unconscious bias. This system automates that entire process in seconds.

## The Solution

Upload a job description and up to 10 resumes. The system:

1. Parses each resume and strips all PII (name, email, phone) to reduce bias
2. Parses the job description and strips company-specific information
3. Sends all anonymized candidates to an LLM in a single call for comparative evaluation
4. Returns a ranked list with scores, strengths, gaps, and hiring recommendations

## Demo

> Streamlit Cloud deployment link here

## Sample Output

| Rank | Candidate | Score | Recommendation |
|------|-----------|-------|----------------|
| 1 | Candidate-6 | 95 | Strong Fit |
| 2 | Candidate-10 | 92 | Strong Fit |
| 3 | Candidate-4 | 88 | Strong Fit |
| 4 | Candidate-7 | 85 | Strong Fit |
| 5 | Candidate-1 | 82 | Strong Fit |
| 6 | Candidate-3 | 78 | Moderate Fit |
| 7 | Candidate-9 | 75 | Moderate Fit |
| 8 | Candidate-8 | 70 | Moderate Fit |
| 9 | Candidate-5 | 65 | Moderate Fit |
| 10 | Candidate-2 | 55 | Moderate Fit |

Each candidate also gets:
- **Key Strengths** — 2–3 specific points on why they fit the role
- **Key Gaps** — 2–3 specific points on what they are missing
- **Summary** — a 2-sentence objective evaluation

## Approach

### Why comparative scoring?
Most resume screening tools evaluate each candidate in isolation and often return the same score for very different profiles. This system passes all resumes to the LLM in a single call, forcing it to rank candidates *relative to each other* — producing genuinely differentiated scores.

### Why PII removal?
Stripping names, emails, and contact details before evaluation reduces the chance of unconscious bias influencing the LLM's assessment. Candidates are identified only by their filename.

### Why not RAG or embeddings?
At the scale this problem requires (5–10 resumes, 1 JD), semantic embeddings add complexity without adding value. The entire context fits comfortably within the LLM's context window, so direct evaluation is faster, simpler, and more accurate.

### Pipeline

```
JD (.txt)
    └── JD Parser (Gemini) ──────────────────────┐
                                                  │
Resumes (.pdf / .docx)                            ▼
    └── Text Extractor                     Comparative LLM Scorer
    └── Resume Parser (Gemini, PII removed) ────► (single API call)
                                                  │
                                                  ▼
                                          Ranked JSON Results
                                                  │
                                                  ▼
                                        Streamlit Frontend
```

## Tech Stack

| Layer | Tool |
|-------|------|
| LLM | Google Gemini 3.1 Flash Lite |
| Backend | FastAPI |
| Frontend | Streamlit |
| PDF Parsing | PyMuPDF (fitz) |
| DOCX Parsing | python-docx |
| Deployment | Streamlit Cloud |

---

## Project Structure

```
Resume-Screening/
├── resume_model/
│   ├── text_extractor.py          # PDF and DOCX text extraction
│   ├── resume_api_integration.py  # Resume parser — extracts structured, anonymized JSON
│   ├── jd_api_integration.py      # JD parser — extracts role requirements JSON
│   └── llm_fit_scorer.py          # Comparative scorer — single LLM call for all candidates
├── api/
│   └── app.py                     # FastAPI endpoint (for API access)
├── streamlit_app.py                         # Streamlit frontend
├── requirements.txt
└── .env                           # GEMINI_API_KEY goes here
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/SanchitPandey26/Resume-Screening.git
cd Resume-Screening
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up your API key

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your-google-gemini-api-key-here
```

Get a free API key at [aistudio.google.com](https://aistudio.google.com)

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

## Using the App

1. Upload your **Job Description** as a `.txt` file
2. Upload **1–10 candidate resumes** in `.pdf` or `.docx` format
3. Click **Analyze Candidates**
4. View the ranked results — filter by Strong Fit, Moderate Fit, or Not Fit
5. Expand the **Parsed Job Description** section to verify extraction

## Running the FastAPI Backend (optional)

If you want to integrate this into another system via API:

```bash
uvicorn api.app:app --reload
```

API available at `http://127.0.0.1:8000`
Interactive docs at `http://127.0.0.1:8000/docs`

**Endpoint:** `POST /evaluate_resumes/`

| Parameter | Type | Description |
|-----------|------|-------------|
| `jd_file` | file | Job description as `.txt` |
| `resumes` | files | One or more `.pdf` or `.docx` resumes |

## API Rate Limits

This project uses Gemini 3.1 Flash Lite on the free tier:

| Limit | Value |
|-------|-------|
| Requests per minute | 15 RPM |
| Tokens per minute | 250K TPM |
| Requests per day | 500 RPD |

For 10 resumes, the system makes **12 API calls total** — well within all limits.

## Potential Improvements

- **Async processing** — parse all resumes in parallel to reduce total time from ~60s to ~10s for 10 resumes
- **Export to CSV** — download the ranked results as a spreadsheet
- **Multi-role support** — screen the same resume pool against multiple JDs simultaneously
- **Confidence scoring** — flag candidates where the LLM is uncertain and recommend human review
- **ATS integration** — connect directly to Greenhouse, Lever, or Workday via API

## License

MIT License © 2025 Sanchit Pandey