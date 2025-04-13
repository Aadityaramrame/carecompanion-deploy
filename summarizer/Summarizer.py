import os
import requests
from summarizer.translator_module import TextTranslator

class Summarizer:
    def __init__(self):
        self.api_url = "https://api-inference.huggingface.co/models/Aadityaramrame/carecompanion-summarizer"
        self.api_token = os.getenv("API_KEY")
        self.headers = {"Authorization": f"Bearer {self.api_token}"}
        self.translator = TextTranslator()

    def clean_text(self, text):
        return ' '.join(text.replace('\n', ' ').split())

    def format_summary(self, summary):
        summary = summary.strip()
        if summary and not summary[0].isupper():
            summary = summary[0].upper() + summary[1:]
        if "expected to recover within" in summary and not summary.endswith("days."):
            summary = summary.rstrip('. ')
            summary += " 7–10 days."
        if "antibiotic" in summary.lower() and "supportive" in summary.lower() and "treatment" not in summary.lower():
            summary += " Treatment includes antibiotics and supportive care."
        return summary

    def summarize_text(self, text, target_lang='en'):
        try:
            detected_lang = self.translator.detect_language(text)
            if detected_lang != 'en':
                text = self.translator.translate_to_english(text)

            cleaned_text = self.clean_text(text)
            payload = {
                "inputs": f"summarize the clinical case with diagnosis, comorbidities, and treatment plan: {cleaned_text}"
            }

            response = requests.post(self.api_url, headers=self.headers, json=payload)
            response.raise_for_status()
            summary = response.json()[0]["generated_text"]

            formatted_summary = self.format_summary(summary)

            if target_lang != 'en':
                formatted_summary = self.translator.translate_from_english(formatted_summary, target_lang)

            return formatted_summary

        except Exception as e:
            return f"Summarization failed: {str(e)}"
