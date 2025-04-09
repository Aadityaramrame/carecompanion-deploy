import spacy
from collections import Counter
import re
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer


class MedicalKeywordExtractor:
    def __init__(self):
        # Load English language model
        self.nlp = spacy.load("en_core_web_sm")

        # Add medical terms to the vocabulary
        self.medical_terms = {
            "symptoms": ["pain", "fever", "headache", "nausea", "vomiting", "fatigue", "cough", "shortness of breath"],
            "conditions": ["diabetes", "hypertension", "asthma", "arthritis", "cancer", "infection", "inflammation"],
            "medications": ["antibiotics", "insulin", "antidepressants", "painkillers", "steroids"],
            "procedures": ["surgery", "biopsy", "x-ray", "mri", "ct scan", "blood test"],
            "body_parts": ["heart", "lungs", "brain", "liver", "kidneys", "stomach", "bones"],
            "measurements": ["blood pressure", "temperature", "heart rate", "weight", "height"]
        }

        # Add custom patterns for medical terms
        if "entity_ruler" not in self.nlp.pipe_names:
            ruler = self.nlp.add_pipe("entity_ruler", before="ner")
        else:
            ruler = self.nlp.get_pipe("entity_ruler")

        patterns = []
        for category, terms in self.medical_terms.items():
            for term in terms:
                patterns.append({"label": category.upper(), "pattern": term})
        ruler.add_patterns(patterns)

    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess the input text."""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters but keep medical terms
        text = re.sub(r'[^\w\s]', ' ', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text

    def extract_keywords(self, text: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Extract keywords from the input text using multiple methods.
        Returns a list of (keyword, score) tuples.
        """
        # Preprocess text
        text = self.preprocess_text(text)

        # Method 1: TF-IDF
        vectorizer = TfidfVectorizer(max_features=100)
        tfidf_matrix = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.toarray()[0]

        # Method 2: Named Entity Recognition
        doc = self.nlp(text)
        ner_entities = [(ent.text, 1.0) for ent in doc.ents]

        # Method 3: Part of Speech tagging for medical terms
        pos_keywords = []
        for token in doc:
            if token.pos_ in ['NOUN', 'ADJ'] and len(token.text) > 2:
                pos_keywords.append((token.text, 0.8))

        # Combine all keywords and their scores
        all_keywords = {}

        # Add TF-IDF keywords
        for word, score in zip(feature_names, tfidf_scores):
            if score > 0:
                all_keywords[word] = score

        # Add NER entities
        for word, score in ner_entities:
            all_keywords[word] = max(all_keywords.get(word, 0), score)

        # Add POS-based keywords
        for word, score in pos_keywords:
            all_keywords[word] = max(all_keywords.get(word, 0), score)

        # Sort keywords by score and return top N
        sorted_keywords = sorted(all_keywords.items(), key=lambda x: x[1], reverse=True)
        return sorted_keywords[:top_n]

    def categorize_keywords(self, keywords: List[Tuple[str, float]]) -> Dict[str, List[Tuple[str, float]]]:
        """
        Categorize the extracted keywords into medical categories.
        """
        categorized = {category: [] for category in self.medical_terms.keys()}
        categorized['other'] = []

        medical_terms_set = {term for category in self.medical_terms.values() for term in category}

        for keyword, score in keywords:
            if keyword in medical_terms_set:
                for category, terms in self.medical_terms.items():
                    if keyword in terms:
                        categorized[category].append((keyword, score))
                        break
            else:
                categorized['other'].append((keyword, score))

        return categorized


class MedicalSummaryInput:
    def __init__(self):
        self.summary = ""

    def get_input(self):
        """Collects medical summary from user input (until an empty line is entered)."""
        print("Enter your medical summary (press Enter on an empty line when done):")
        while True:
            line = input()
            if line.strip() == "":
                break
            self.summary += line + " "

        if not self.summary.strip():
            print("No summary provided. Exiting.")
            return None
        return self.summary


class MedicalKeywordProcessor:
    def __init__(self):
        self.extractor = MedicalKeywordExtractor()
        self.input_handler = MedicalSummaryInput()

    def process(self):
        # Get medical summary input from user
        summary = self.input_handler.get_input()
        if not summary:
            return

        # Extract keywords
        keywords = self.extractor.extract_keywords(summary)

        # Categorize keywords
        categorized_keywords = self.extractor.categorize_keywords(keywords)

        # Print results
        print("\nExtracted Keywords:")
        print("-" * 50)
        for category, words in categorized_keywords.items():
            if words:  # Only print categories that have keywords
                print(f"\n{category.upper()}:")
                for word, score in words:
                    print(f"  - {word} (score: {score:.4f})")


def main():
    processor = MedicalKeywordProcessor()
    processor.process()


if __name__ == "__main__":
    main()
