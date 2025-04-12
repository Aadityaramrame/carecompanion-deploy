from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from typing import List
from OCR.ocr_processor import OCRProcessor  # Correct import of OCRProcessor class
from summarizer.Summarizer import Summarizer
from KeywordExtraction.MedicalKeywordExtractor import MedicalKeywordExtractor
from summarizer.translator_module import TextTranslator
from Chatbot.chatbot_function import DataProcessor, Chatbot

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
    try:
        matched_question, answer, source = chatbot.get_response(req.question)
        return ChatResponse(
            matched_question=matched_question,
            answer=answer,
            source=source
        )
    except Exception as e:
        return ChatResponse(
            matched_question="ERROR",
            answer=f"Something went wrong: {str(e)}",
            source="System"
        )

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
    extractor = MedicalKeywordExtractor()
    keywords = extractor.extract_keywords(req.text)
    categorized = extractor.categorize_keywords(keywords)
    return {"keywords": categorized}

@app.post("/translate")
async def translate(req: TranslateRequest):
    translator = TextTranslator()
    translated = translator.translate_to_english(req.text)
    return {"translated_text": translated}
