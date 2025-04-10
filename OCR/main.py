from fastapi import FastAPI, File, UploadFile, HTTPException, Header
import os
from ocr_processor import OCRProcessor, MedicalDataExtractor  # your OG CODE modules

app = FastAPI()
API_KEY = os.getenv("OCR_API_KEY")  # will come from env var

@app.post("/extract/")
async def extract(
    authorization: str = Header(None),
    file: UploadFile = File(...)
):
    # simple Bearer‑token check
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Invalid API key")
    # read image bytes and convert to array
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # run OCR & extraction
    text = OCRProcessor().extract_text(img)
    data = MedicalDataExtractor().extract_medical_data(text)
    return {"text": text, "structured": data}
