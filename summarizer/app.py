from pydantic import BaseModel
from summarizer import MedicalSummary
from flask import Flask, request, jsonify
from Summarizer import generate_summary
from translator_module import translate_text

app = Flask(__name__)

@app.route('/')
def home():
    return "CareCompanion Summarizer API is Live 🎉"

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "Text is required"}), 400
    summary = generate_summary(text)
    return jsonify({"summary": summary})

@app.route('/translate', methods=['POST'])
def translate():
    data = request.get_json()
    text = data.get("text", "")
    source = data.get("source_lang", "auto")
    target = data.get("target_lang", "en")
    if not text:
        return jsonify({"error": "Text is required"}), 400
    translated = translate_text(text, source, target)
    return jsonify({"translation": translated})

if __name__ == '__main__':
    app.run()
