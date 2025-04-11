import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException, Header
import os
import pytesseract
import shutil
import logging
from ocr_processor import OCRProcessor, MedicalDataExtractor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== TESSERACT CONFIGURATION (CRITICAL FOR RENDER) =====
def verify_tesseract():
    """Ensure Tesseract is properly installed and accessible"""
    # Check system PATH first
    tesseract_path = shutil.which("tesseract") or '/usr/bin/tesseract'
    
    # Fallback paths if default doesn't exist
    possible_paths = [
        '/usr/bin/tesseract',
        '/usr/local/bin/tesseract',
        '/bin/tesseract',
        '/app/.apt/usr/bin/tesseract'  # Render-specific path
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            logger.info(f"✅ Tesseract configured at: {path}")
            return
            
    raise RuntimeError("""
    Tesseract OCR not found! Verify:
    1. build.sh installs tesseract-ocr
    2. Render build logs show successful installation
    3. Paths exist: /usr/bin/tesseract
    """)

verify_tesseract()  # Fail fast during startup

# ===== FASTAPI APP SETUP =====
app = FastAPI()
API_KEY = os.getenv("OCR_API_KEY")  # From Render environment variables

# Constants
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB (Render free tier has memory limits)
ALLOWED_CONTENT_TYPES = ["image/jpeg", "image/png"]  # BMP removed for stability

# ===== CORE ENDPOINTS =====
@app.post("/extract/")
async def extract(
    authorization: str = Header(None),
    file: UploadFile = File(...)
):
    """Process medical document image and extract structured data"""
    try:
        # 1. Authentication
        if not authorization or authorization != f"Bearer {API_KEY}":
            logger.warning("Unauthorized access attempt")
            raise HTTPException(401, "Invalid API key")

        # 2. File Validation
        if file.content_type not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(400, "Only JPEG/PNG images supported")
        
        if file.size > MAX_FILE_SIZE:
            raise HTTPException(413, f"Max file size: {MAX_FILE_SIZE//1024//1024}MB")

        # 3. Image Processing
        logger.info(f"Processing file: {file.filename} ({file.size} bytes)")
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None or img.size == 0:
            raise HTTPException(400, "Invalid or empty image")

        # 4. OCR Processing (with performance logging)
        logger.info("Starting OCR processing...")
        try:
            text = OCRProcessor().extract_text(img)
            if not text.strip():
                raise HTTPException(422, "No text detected")
        except Exception as ocr_error:
            logger.error(f"OCR failed: {str(ocr_error)}")
            raise HTTPException(500, "OCR processing error")

        # 5. Data Extraction
        data = MedicalDataExtractor().extract_medical_data(text)
        if not data:
            raise HTTPException(500, "Data extraction failed")

        return {
            "status": "success",
            "text": text[:500] + "..." if len(text) > 500 else text,  # Truncate long text
            "structured": data
        }

    except HTTPException as he:
        logger.error(f"Client error: {he.detail}")
        raise
    except Exception as e:
        logger.critical(f"Server error: {str(e)}", exc_info=True)
        raise HTTPException(500, "Internal processing error")

@app.get("/", methods=["GET", "HEAD"])
async def health_check():
    """Endpoint for health checks and Tesseract verification"""
    return {
        "status": "alive",
        "ocr_ready": os.path.exists(pytesseract.pytesseract.tesseract_cmd)
    }
