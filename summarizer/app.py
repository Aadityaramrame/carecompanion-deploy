from fastapi import FastAPI
from pydantic import BaseModel
from summarizer import MedicalSummary

app = FastAPI()
summarizer = MedicalSummary()

class SummarizationRequest(BaseModel):
    text: str
    lang: str = 'en'

@app.get("/")
def root():
    return {"message": "Care Companion Summarizer is running 🚀"}

@app.post("/summarize")
def summarize(request: SummarizationRequest):
    result = summarizer.summarize_text(request.text, request.lang)
    return {"summary": result}
