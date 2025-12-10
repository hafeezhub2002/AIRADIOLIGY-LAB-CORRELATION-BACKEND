from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import io
from PyPDF2 import PdfReader

# Initialize FastAPI
app = FastAPI()

# CORS configuration — allow exact frontend URLs
origins = [
    "http://localhost:3000",  # for local testing
    "https://airadioligy-lab-correlation-f-r3bc.vercel.app",  # deployed frontend
    "https://airadioligy-lab-correlation-f.vercel.app",       # any other deployed frontend
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # exact allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for structured data
class MedicalData(BaseModel):
    radiology_report: str
    lab_values: str
    clinical_notes: str

# Mock RAG function (replace with real RAG logic)
def retrieve_relevant_facts(text: str) -> str:
    return "Relevant medical knowledge found."

# Mock LLM client (replace with Gemini or any LLM)
class LLMClient:
    def models_generate_content(self, model: str, contents: str) -> str:
        return (
            "Discrepancy: Yes\n"
            "Summary: Radiology vs Labs mismatch.\n"
            "Diagnostic Explanation: Further evaluation needed."
        )

client = LLMClient()

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Backend is running successfully!"}

# Structured data endpoint
@app.post("/analyze")
async def analyze_data(data: MedicalData):
    try:
        combined_text = (
            f"Radiology: {data.radiology_report}\n"
            f"Labs: {data.lab_values}\n"
            f"Notes: {data.clinical_notes}"
        )

        rag_context = retrieve_relevant_facts(combined_text)

        prompt = f"""
RAG Context:
{rag_context}

Patient Data:
{combined_text}

Provide:
- Discrepancy Yes/No
- Summary
- Diagnostic explanation
        """

        response_text = client.models_generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        result = {}
        for line in response_text.split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                result[key.strip().lower().replace(" ", "_")] = value.strip()

        return {"analysis_result": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# PDF upload endpoint
@app.post("/analyze_pdf")
async def analyze_pdf(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        reader = PdfReader(io.BytesIO(contents))
        text = ""

        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"

        data = MedicalData(
            radiology_report="Extracted from PDF",
            lab_values="Extracted from PDF",
            clinical_notes=text,
        )

        return await analyze_data(data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
