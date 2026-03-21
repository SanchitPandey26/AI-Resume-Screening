import os
from google import genai
from google.genai import types


def call_jd_gemini_api(jd_text: str, api_key: str) -> str:
    client = genai.Client(api_key=api_key)
    model = "gemini-3.1-flash-lite-preview"

    prompt_text = f"""Extract the following fields from the job description below and return them as a single JSON object.

Fields to extract:
- job_title: the title of the role
- required_skills: list of skills explicitly stated as required or must-have
- nice_to_have_skills: list of skills stated as preferred, bonus, or optional
- experience_level: string describing required experience (e.g. "2+ years", "Entry Level", "Senior")
- education: required degree or qualification as a string
- responsibilities: list of key job responsibilities

Rules:
- DO NOT include company name, office address, salary, benefits, perks, or culture info
- DO NOT include anything about the application process
- Separate required vs nice-to-have skills carefully — only mark as required if explicitly stated
- Keep responsibilities concise, one action per item

Job Description:
{jd_text}"""

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt_text),
            ],
        ),
    ]

    generate_content_config = types.GenerateContentConfig(
        temperature=0.1,
        thinking_config=types.ThinkingConfig(
            thinking_level="MINIMAL",
        ),
        response_mime_type="application/json",
        response_schema=genai.types.Schema(
            type=genai.types.Type.OBJECT,
            required=["job_title", "required_skills", "nice_to_have_skills", "experience_level", "education", "responsibilities"],
            properties={
                "job_title": genai.types.Schema(
                    type=genai.types.Type.STRING,
                ),
                "required_skills": genai.types.Schema(
                    type=genai.types.Type.ARRAY,
                    items=genai.types.Schema(
                        type=genai.types.Type.STRING,
                    ),
                ),
                "nice_to_have_skills": genai.types.Schema(
                    type=genai.types.Type.ARRAY,
                    items=genai.types.Schema(
                        type=genai.types.Type.STRING,
                    ),
                ),
                "experience_level": genai.types.Schema(
                    type=genai.types.Type.STRING,
                ),
                "education": genai.types.Schema(
                    type=genai.types.Type.STRING,
                ),
                "responsibilities": genai.types.Schema(
                    type=genai.types.Type.ARRAY,
                    items=genai.types.Schema(
                        type=genai.types.Type.STRING,
                    ),
                ),
            },
        ),
        system_instruction=[
            types.Part.from_text(text="""You are an expert HR assistant and job description parsing specialist.
Your job is to extract structured, anonymized information from job descriptions.
You must remove all company-specific information such as company name, office location, benefits, perks, culture descriptions, and application process details.
Always return output as a valid JSON object following the requested schema exactly.
Focus only on role requirements — skills, experience, education, and responsibilities.
If a field is missing or not found, return null for strings and empty array [] for lists.
Do not include any commentary, explanation, or markdown — only the raw JSON."""),
        ],
    )

    response_text = ""
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        response_text += chunk.text

    return response_text