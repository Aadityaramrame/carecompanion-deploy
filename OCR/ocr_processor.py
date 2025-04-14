import cv2
import pytesseract
import torch
import numpy as np
from PIL import Image
import re
import json
import glob
import os
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
# --- Enhanced Medical Regex Patterns ---
MEDICAL_PATTERNS = {
    'patient': {
        # Pattern 1: e.g., "PATIENT (M) /13Y"
        'age_gender_pattern1': r'PATIENT\s*\(\s*(?P<gender1>M|F|Male|Female)\s*\)\s*/\s*(?P<age1>\d{1,3})(?=Y\b)',
        # Pattern 2: e.g., ", 42/M"
        'age_gender_pattern2': r',\s*(?P<age2>\d{1,3})\s*/\s*(?P<gender2>M|F|Male|Female)\b'
    },
    'patient_extra': {
        'weight': r'Weight\s*\(Kg\)\s*:\s*(\d+)',
        'health_card': r'Health\s*Card[:\s]*Exp[:\s]*(\d{4}[\/\-]\d{2}[\/\-]\d{2})'
    },
    'clinical': {
        'diagnosis': r'(?i)Diagnosis[:\s-]+([\s\S]+?)(?=\n\s*\n|Medicine Name)',
        'vitals': {
            'bp': r'(?i)(?:BP|Blood\s*Pressure)[\s:]*(\d{2,3}\s*/\s*\d{2,3})\s*(?:mmHg)?',
            'pulse': r'(?i)(?:Pulse|Heart\s*Rate)[\s:]*(\d{2,3})\s*(?:bpm)?',
            'temp': r'(?i)(?:Temp|Temperature)[\s:]*(\d{2}\.?\d*)\s*°?[CF]?',
            'rr': r'(?i)(?:RR|Respiratory\s*Rate)[\s:]*(\d{2})\s*(?:/min)?',
            'spo2': r'(?i)(?:SpO2|Oxygen\s*Saturation)[\s:]*(\d{2,3})\s*%?'
        },
        'complaints': r'(?i)Chief\s*Complaints[:\s-]+([\s\S]+?)(?=\n)',
        'reactions': r'(?i)(?:Adverse\s*Reactions)[\s:]+([\s\S]+?)(?=\n)',
        'investigations': r'(?i)(?:Investigations|Tests)[:\s-]+([\s\S]+?)(?=\n\s*\n|Medicine|Advice|$)'
    },
    'medications': {
        'pattern': r'(?m)^\s*\d+\)\s*((?:(?!^\s*\d+\)).)+)'
    },
    'advice': r'(?i)Advice[:\s-]+([\s\S]+?)(?=\n\s*(?:Follow\s*Up|Next\s*Visit)|$)',
    'follow_up': r'(?i)Follow\s*Up[:\s-]+(\d{2}[\/\-]\d{2}[\/\-]\d{2,4})'
}


class OCRProcessor:
    """Extracts text from an image using Tesseract OCR."""
    
    def extract_text_from_image(self, image_bytes: bytes) -> str:  # Change input type hint
    try:
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # Read image from bytes
        
        if img is None:
            raise ValueError("Failed to decode image")
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Now works with proper numpy array
        return pytesseract.image_to_string(gray)
    
    except Exception as e:
        raise ValueError(f"Image processing failed: {str(e)}")


class MedicalDataExtractor:
    """Extracts structured medical data from OCR text."""
    
    def extract_age_gender(self, text):
        # Try pattern 1
        m1 = re.search(MEDICAL_PATTERNS['patient']['age_gender_pattern1'], text)
        if m1:
            return m1.group('age1').strip(), m1.group('gender1').strip()
        # Else try pattern 2
        m2 = re.search(MEDICAL_PATTERNS['patient']['age_gender_pattern2'], text)
        if m2:
            return m2.group('age2').strip(), m2.group('gender2').strip()
        return None, None

    def extract_vitals(self, text):
        """Extract all vital signs from text."""
        vitals = {}
        for vital, pattern in MEDICAL_PATTERNS['clinical']['vitals'].items():
            match = re.search(pattern, text)
            if match:
                value = match.group(1).strip()
                # Clean up values
                if vital == 'bp':
                    value = re.sub(r'\s+', '', value)
                elif vital in ['temp', 'spo2']:
                    value = value.replace(' ', '')
                vitals[vital] = value
        return vitals

    def extract_medical_data(self, text):
        # Normalize whitespace while preserving newlines
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)

        result = {
            "patient": {},
            "vitals": {},
            "diagnosis": [],
            "medications": [],
            "investigations": [],
            "advice": [],
            "follow_up": {}
        }

        try:
            # Extract Age and Gender
            age, gender = self.extract_age_gender(text)
            if age and gender:
                result["patient"]["age"] = age
                result["patient"]["gender"] = gender

            # Extract Weight
            weight = re.search(MEDICAL_PATTERNS['patient_extra']['weight'], text, re.I)
            if weight:
                result["patient"]["weight"] = f"{weight.group(1).strip()} kg"

            # Extract Vitals
            result["vitals"] = self.extract_vitals(text)

            # Extract Diagnosis
            diagnosis = re.search(MEDICAL_PATTERNS['clinical']['diagnosis'], text)
            if diagnosis:
                result["diagnosis"] = [line.strip() for line in diagnosis.group(1).split('\n') if line.strip()]

            # Extract Investigations
            inv = re.search(MEDICAL_PATTERNS['clinical']['investigations'], text)
            if inv:
                result["investigations"] = [line.strip() for line in inv.group(1).split('\n') if line.strip()]

            # Extract Medications
            meds = re.findall(MEDICAL_PATTERNS['medications']['pattern'], text, re.DOTALL | re.MULTILINE)
            if meds:
                result["medications"].extend([re.sub(r'\s+', ' ', m).strip() for m in meds if m.strip()])

            # Extract Advice
            advice = re.search(MEDICAL_PATTERNS['advice'], text)
            if advice:
                result["advice"] = [line.strip() for line in advice.group(1).split('\n') if line.strip()]

            # Extract Follow Up date
            follow_up = re.search(MEDICAL_PATTERNS['follow_up'], text)
            if follow_up:
                result["follow_up"] = {"date": follow_up.group(1).strip()}

        except Exception as e:
            print(f"Extraction error: {str(e)}")
            return None

        return result


class MedicalOCRApp:
    """Main application class for processing medical images."""
    
    def __init__(self, input_folder):
        self.input_folder = input_folder
        self.ocr_processor = OCRProcessor()
        self.data_extractor = MedicalDataExtractor()

    def _print_separator(self):
        print("-" * 50)

    def process_images(self):
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(self.input_folder, ext)))

        if not image_paths:
            print(f"No images found in {self.input_folder}")
            return

        print(f"Found {len(image_paths)} images:")

        for idx, img_path in enumerate(image_paths, 1):
            print(f"\n## Processing Image {idx}/{len(image_paths)}: {os.path.basename(img_path)}\n")
            try:
                # OCR Extraction
                extracted_text = self.ocr_processor.extract_text(img_path)
                print("### Raw OCR Text:")
                print("\n" + extracted_text + "\n")

                # Structured Data Extraction
                structured_data = self.data_extractor.extract_medical_data(extracted_text)
                print("### Structured Medical Data:")
                print("json\n" + json.dumps(structured_data, indent=2, ensure_ascii=False) + "\n")

                self._print_separator()
            except Exception as e:
                print(f"**Error processing {os.path.basename(img_path)}:** {str(e)}")
                self._print_separator()

    def run(self):
        print("Medical Prescription Structured Data Extractor")
        print(f"Ensure that your images are placed in the '{self.input_folder}' folder before running the application!\n")
        self.process_images()


#if __name__ == "__main__":
    # Update the folder path as needed; for example, "images" if you have a local folder named images.
    #INPUT_FOLDER = "images"
    #app = MedicalOCRApp(INPUT_FOLDER)
    #app.run()
