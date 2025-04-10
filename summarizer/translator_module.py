from googletrans import Translator

class TextTranslator:
    def __init__(self):
        self.translator = Translator()

    def detect_language(self, text):
        try:
            return self.translator.detect(text).lang
        except Exception as e:
            print(f"Language detection failed: {e}")
            return 'en'

    def translate_to_english(self, text):
        try:
            return self.translator.translate(text, dest='en').text
        except Exception as e:
            print(f"Translation to English failed: {e}")
            return text

    def translate_from_english(self, text, target_lang):
        try:
            return self.translator.translate(text, dest=target_lang).text
        except Exception as e:
            print(f"Translation from English failed: {e}")
            return text
