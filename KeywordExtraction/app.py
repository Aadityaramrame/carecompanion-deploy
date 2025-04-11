from flask import Flask, request, jsonify
from MedicalKeywordExtractor import MedicalKeywordProcessor, MedicalKeywordExtractor

app = Flask(__name__)
extractor = MedicalKeywordExtractor()

@app.route("/extract_keywords", methods=["POST"])
def extract_keywords():
    data = request.get_json()
    summary = data.get("summary", "")
    if not summary:
        return jsonify({"error": "No summary provided"}), 400

    keywords = extractor.extract_keywords(summary)
    categorized = extractor.categorize_keywords(keywords)
    return jsonify(categorized)

@app.route("/", methods=["GET"])
def home():
    return "Medical Keyword Extractor is running!"

if __name__ == "__main__":
    app.run(debug=True)
