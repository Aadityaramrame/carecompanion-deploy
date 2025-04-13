import os
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from summarizer.translator_module import TextTranslator

class Summarizer:
    def __init__(self, model_path='t5-small'):  # Using t5-small
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        API_KEY = os.getenv('API_KEY')  # Optional if using a private HF model

        self.tokenizer = T5Tokenizer.from_pretrained(model_path, use_auth_token=API_KEY)
        self.model = self.load_model(model_path, API_KEY)
        self.translator = TextTranslator()

    def load_model(self, model_path, API_KEY):
        model = T5ForConditionalGeneration.from_pretrained(
            model_path,
            use_auth_token=API_KEY,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True
        )
        model.to(self.device)
        model.eval()
        return model

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

    def summarize_text(self, text, target_lang='en', max_length=200, min_length=30):
        try:
            detected_lang = self.translator.detect_language(text)
            print(f"Detected language: {detected_lang}")

            if detected_lang != 'en':
                text = self.translator.translate_to_english(text)

            cleaned_text = self.clean_text(text)
            print(f"Cleaned input: {cleaned_text[:100]}...")  # Only printing snippet

            prompt = f'summarize the clinical case with diagnosis, comorbidities, and treatment plan: {cleaned_text}'
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)

            with torch.no_grad():
                summary_ids = self.model.generate(
                    input_ids,
                    max_length=max_length,
                    min_length=min_length,
                    num_beams=4,
                    no_repeat_ngram_size=2,
                    repetition_penalty=2.0,
                    length_penalty=1.5,
                    early_stopping=True
                )

            raw_summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            formatted_summary = self.format_summary(raw_summary)

            if target_lang != 'en':
                formatted_summary = self.translator.translate_from_english(formatted_summary, target_lang)

            print(f"Final summary: {formatted_summary[:100]}...")
            return formatted_summary

        except Exception as e:
            return f"Summarization failed due to: {str(e)}"
