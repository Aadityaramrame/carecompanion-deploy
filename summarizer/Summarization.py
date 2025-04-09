import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from googletrans import Translator
class MedicalSummary:
    def __init__(self, model_path='/Users/aditi/Desktop/CareCompanion/fine_tuned_model'):
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = T5Tokenizer.from_pretrained('t5-base')
        self.model = self.load_model()

    def load_model(self):
        model = T5ForConditionalGeneration.from_pretrained(self.model_path)
        model.to(self.device)
        model.eval()
        print("Model loaded successfully!")
        return model

    def clean_text(self, text):
        """Remove newlines and extra spaces."""
        return ' '.join(text.replace('\n', ' ').split())

    def format_summary(self, summary):
        """Add final touches: capitalization, periods, prognosis completion."""
        summary = summary.strip()

        # Ensure sentence starts with capital letter
        if summary and not summary[0].isupper():
            summary = summary[0].upper() + summary[1:]

        # Ensure the summary ends with a proper prognosis
        if "expected to recover within" in summary and not summary.endswith("days."):
            summary = summary.rstrip('. ')
            summary += " 7–10 days."

        # Optional: auto-add treatment snippet if missing
        if "antibiotic" in summary.lower() and "supportive" in summary.lower() and "treatment" not in summary.lower():
            summary += " Treatment includes antibiotics and supportive care."

        return summary

    def summarize_text(self, text, max_length=500, min_length=100):
        try:
            cleaned_text = self.clean_text(text)
            prompt = f'summarize the clinical case with diagnosis, comorbidities, and treatment plan: {cleaned_text}'

            input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)

            summary_ids = self.model.generate(
                input_ids,
                max_length=max_length,
                min_length=min_length,
                num_beams=8,
                no_repeat_ngram_size=3,
                repetition_penalty=2.5,
                length_penalty=2.0,
                early_stopping=True
            )

            raw_summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            formatted_summary = self.format_summary(raw_summary)
            return formatted_summary

        except Exception as e:
            print(f"Summarization failed: {e}")
            return None
