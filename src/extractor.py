import spacy
from collections import Counter

# Load the small English spaCy model
nlp = spacy.load("en_core_web_sm")

class LinguisticExtractor:
    def __init__(self):
        self.nlp = nlp

    def extract_linguistics(self, text: str) -> dict:
        """Process text through spaCy pipeline to extract POS, Entities, and Chunks."""
        doc = self.nlp(text)
        
        # 1. POS Tag Distribution & Analysis
        pos_counts = Counter([token.pos_ for token in doc if not token.is_punct and not token.is_space])
        token_analysis = [
            {"text": token.text, "lemma": token.lemma_, "pos": token.pos_, "tag": token.tag_, "dep": token.dep_}
            for token in doc if not token.is_punct and not token.is_space
        ]
        
        # 2. Named Entity Recognition (NER)
        entities = [
            {"text": ent.text, "label": ent.label_, "explanation": spacy.explain(ent.label_)}
            for ent in doc.ents
        ]
        
        # 3. Noun Chunks (Shallow Parsing)
        noun_chunks = [chunk.text.strip() for chunk in doc.noun_chunks if len(chunk.text.strip()) > 2]
        
        return {
            "token_analysis": token_analysis,
            "pos_distribution": dict(pos_counts),
            "entities": entities,
            "noun_chunks": noun_chunks
        }
