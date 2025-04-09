from googletrans import Translator

def translate_medical_summary(summary_text, target_language):
    try:
        translator = Translator()
        translated_text = translator.translate(summary_text, dest=target_language).text
        return {'status': 'success', 'original': summary_text, 'translated': translated_text}
    except Exception as e:
        print(f"Translation failed: {e}")
        return {'status': 'error', 'error': str(e)}
