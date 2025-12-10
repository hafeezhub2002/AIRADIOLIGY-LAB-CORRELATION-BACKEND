from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import io
from PyPDF2 import PdfReader

# Initialize FastAPI
app = FastAPI()

origins = [
    "http://localhost:3000",
    "https://your-frontend-vercel-url.vercel.app",
]


# ✅ Allow CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for incoming data
class MedicalData(BaseModel):
    radiology_report: str
    lab_values: str
    clinical_notes: str

# Example RAG retrieval function
def retrieve_relevant_facts(text: str) -> str:
    return "Relevant medical knowledge context related to the patient's data."

# Example LLM client stub
class LLMClient:
    def models_generate_content(self, model: str, contents: str) -> str:
        # Mock response for demo purposes
        return """Discrepancy: Yes
Summary: Radiology report shows mild infiltrates but lab values indicate no infection.
Diagnostic Explanation: The radiology findings and labs are contradictory; further investigation is needed."""

client = LLMClient()

# ✅ Simple GET endpoint to check server status
@app.get("/")
async def root():
    return {"message": "FastAPI server is running!"}

# ✅ POST endpoint for medical analysis (structured input)
@app.post("/analyze")
async def analyze_data(data: MedicalData):
    try:
        combined_text = (
            f"Radiology: {data.radiology_report}\n"
            f"Labs: {data.lab_values}\n"
            f"Notes: {data.clinical_notes}"
        )

        rag_context = retrieve_relevant_facts(combined_text)

        prompt = (
            "You are a medical contradiction detection system.\n\n"
            f"### RAG Context:\n{rag_context}\n\n"
            f"### Patient Data:\n{combined_text}\n\n"
            "Provide a structured response:\n"
            "- Discrepancy (Yes/No)\n"
            "- Summary of contradiction\n"
            "- Diagnostic explanation"
        )

        response_text = client.models_generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        # Parse LLM output
        result = {}
        for line in response_text.split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                result[key.strip().lower().replace(" ", "_")] = value.strip()

        return {"analysis_result": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ✅ New POST endpoint to handle PDF uploads
@app.post("/analyze_pdf")
async def analyze_pdf(file: UploadFile = File(...)):
    try:
        # Read PDF file
        contents = await file.read()
        pdf_reader = PdfReader(io.BytesIO(contents))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"

        # For simplicity, we put all text in clinical_notes
        data = {
            "radiology_report": "Extracted from PDF",
            "lab_values": "Extracted from PDF",
            "clinical_notes": text
        }

        # Reuse existing analyze_data endpoint
        result = await analyze_data(MedicalData(**data))
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
