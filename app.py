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

# Logging startup process
import logging
logging.basicConfig(level=logging.INFO)

# Try initializing components and log any errors
try:
    logging.info("Initializing components...")
    
    # Initialize components
    data_processor = DataProcessor()
    chatbot = Chatbot(data_processor)
    summarizer = Summarizer()
    
    logging.info("✅ All components loaded successfully")
    
except Exception as e:
    logging.error(f"❌ Error during initialization: {str(e)}")
    raise e  # Re-raise the error to stop app startup

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
@app.get("/")
async def root():
    return {"status": "Care Companion backend is live!"}

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
        logging.error(f"❌ Error in /chat route: {str(e)}")
        return ChatResponse(
            matched_question="ERROR",
            answer=f"Something went wrong: {str(e)}",
            source="System"
        )

@app.post("/ocr")
async def process_ocr(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        
        # Instantiate the OCRProcessor class
        ocr_processor = OCRProcessor()
        
        # Pass image bytes to the OCR processor
        extracted_text = ocr_processor.extract_text_from_image(image_bytes)
        return {"extracted_text": extracted_text}
    except Exception as e:
        logging.error(f"❌ Error in /ocr route: {str(e)}")
        return {"error": f"Something went wrong: {str(e)}"}

@app.post("/summarize")
async def summarize_text(req: SummaryRequest):
    try:
        summary = summarizer.summarize(req.text)
        return {"summary": summary}
    except Exception as e:
        logging.error(f"❌ Error in /summarize route: {str(e)}")
        return {"error": f"Something went wrong: {str(e)}"}

@app.post("/keywords")
async def extract_keywords(req: KeywordRequest):
    try:
        extractor = MedicalKeywordExtractor()
        keywords = extractor.extract_keywords(req.text)
        categorized = extractor.categorize_keywords(keywords)
        return {"keywords": categorized}
    except Exception as e:
        logging.error(f"❌ Error in /keywords route: {str(e)}")
        return {"error": f"Something went wrong: {str(e)}"}

@app.post("/translate")
async def translate(req: TranslateRequest):
    try:
        translator = TextTranslator()
        translated = translator.translate_to_english(req.text)
        return {"translated_text": translated}
    except Exception as e:
        logging.error(f"❌ Error in /translate route: {str(e)}")
        return {"error": f"Something went wrong: {str(e)}"}
