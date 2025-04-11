import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException, Header
import os
import pytesseract
from ocr_processor import OCRProcessor, MedicalDataExtractor

# Configure Tesseract path for Render
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

app = FastAPI()
API_KEY = os.getenv("OCR_API_KEY")  # From Render environment variables

# Constants
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_CONTENT_TYPES = ["image/jpeg", "image/png", "image/bmp"]

@app.post("/extract/")
async def extract(
    authorization: str = Header(None),
    file: UploadFile = File(...)
):
    try:
        # 1. Authentication
        if authorization != f"Bearer {API_KEY}":
            raise HTTPException(status_code=401, detail="Invalid API key")

        # 2. File Validation
        if file.content_type not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(400, "Only JPEG/PNG/BMP images are allowed")
        
        if file.size > MAX_FILE_SIZE:
            raise HTTPException(413, f"Image too large (max {MAX_FILE_SIZE//1024//1024}MB)")

        # 3. Image Processing
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(400, "Invalid image data")

        # 4. OCR Processing
        text = OCRProcessor().extract_text(img)
        if not text.strip():
            raise HTTPException(422, "No text detected in image")

        # 5. Data Extraction
        data = MedicalDataExtractor().extract_medical_data(text)
        if not data:
            raise HTTPException(500, "Failed to extract medical data")

        return {
            "status": "success",
            "text": text,
            "structured": data
        }

    except HTTPException:
        raise  # Re-throw FastAPI exceptions
    except Exception as e:
        raise HTTPException(500, f"Internal server error: {str(e)}")

# Health check endpoint
@app.get("/")
async def health_check():
    return {"status": "alive", "ocr_ready": True}
