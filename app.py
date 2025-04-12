from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from typing import List
from OCR.ocr_processor import OCRProcessor  # Correct import of OCRProcessor class
from summarizer.Summarizer import Summarizer
from KeywordExtraction.MedicalKeywordExtractor import extract_medical_keywords
from summarizer.translator_module import translate_to_english
from chatbot.chatbot_function import DataProcessor, Chatbot

app = FastAPI(
    title="CareCompanion API",
    description="Unified API for chatbot, OCR, summarizer, keyword extractor, and translation",
    version="1.0"
)

# Load components at startup
data_processor = DataProcessor()
chatbot = Chatbot(data_processor)
summarizer = Summarizer()

# Request/Response models
class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    matched_question: str
    answer: str
    source: str

class SummaryRequest(BaseModel):
    text: str

class KeywordRequest(BaseModel):
    text: str

class TranslateRequest(BaseModel):
    text: str
    source_lang: str

# Routes
@app.post("/chat", response_model=ChatResponse)
async def get_chat_response(req: ChatRequest):
    matched_question, answer, source = chatbot.get_response(req.question)
    return {"matched_question": matched_question, "answer": answer, "source": source}

@app.post("/ocr")
async def process_ocr(file: UploadFile = File(...)):
    image_bytes = await file.read()
    
    # Instantiate the OCRProcessor class
    ocr_processor = OCRProcessor()
    
    # Pass image bytes to the OCR processor
    extracted_text = ocr_processor.extract_text_from_image(image_bytes)
    return {"extracted_text": extracted_text}

@app.post("/summarize")
async def summarize_text(req: SummaryRequest):
    summary = summarizer.summarize(req.text)
    return {"summary": summary}

@app.post("/keywords")
async def extract_keywords(req: KeywordRequest):
    keywords = extract_medical_keywords(req.text)
    return {"keywords": keywords}

@app.post("/translate")
async def translate(req: TranslateRequest):
    translated = translate_to_english(req.text, req.source_lang)
    return {"translated_text": translated}
