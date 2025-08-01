from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import fitz  # PyMuPDF
import os
import json
import tempfile

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# FastAPI app
app = FastAPI()

# PDF to text function
def extract_text_from_pdf(file_path: str) -> str:
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# LangChain prompt
template = """
You are an expert resume parser.

Extract structured information from the resume text below and return it as a **valid and complete JSON** in the format:

{{
  "resume": {{
    "Name": "...",
    "Email": "...",
    "Phone": "...",
    "Links": {{
      "LinkedIn": "...",
      "GitHub": "...",
      "Portfolio": "...",
      "OtherLinks": [],
      "Projects": [
        {{
          "Title": "...",
          "Description": "...",
          "LiveLink": "...",
          "GitHubLink": "...",
          "StartDate": "...",
          "EndDate": "...",
          "Role": "...",
          "description": "...",
          "bullets": [],
          "TechStack": []
        }}
      ]
    }},
    "Summary": "...",
    "Skills": {{
      "Tools": [],
      "soft skills": [],
      "TechStack": [],
      "Languages": [],
      "Others": []
    }},
    "WorkExperience": [
      {{
        "JobTitle": "...",
        "Company": "...",
        "Duration": "...",
        "Location": "...",
        "Responsibilities": [],
        "TechStack": []
      }}
    ],
    "Education": [
      {{
        "Degree": "...",
        "Mark": "...",
        "Institution": "...",
        "Location": "...",
        "StartYear": "...",
        "EndYear": "..."
      }}
    ],
    "Certifications": [],
    "Languages": [],
    "Achievements": []
    "Awards": [],
    "VolunteerExperience": []
    "Hobbies": [],
    "Interests": []
    "References": []
  }},
  "summary": "One-paragraph professional summary (same as Summary above)"
}}

- Do NOT guess or hallucinate links.
- Skip missing fields entirely (do not insert nulls or empty strings).
- Ensure JSON is strictly valid.

---

Resume Text:
{resume_text}
"""

prompt = PromptTemplate.from_template(template)
llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-3.5-turbo", temperature=0)
parser = StrOutputParser()
chain = prompt | llm | parser

@app.post("/parse-resume")
async def parse_resume(file: UploadFile = File(...)):
    try:
        # Save uploaded file temporarily
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            temp_path = tmp.name

        # Extract text
        resume_text = extract_text_from_pdf(temp_path)

        # Invoke LLM
        result_str = chain.invoke({"resume_text": resume_text})

        # Parse JSON safely
        try:
            result_json = json.loads(result_str)
        except json.JSONDecodeError as je:
            raise HTTPException(status_code=500, detail=f"Invalid JSON returned by LLM: {je}")

        # Cleanup
        os.remove(temp_path)

        return JSONResponse(content=result_json)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
