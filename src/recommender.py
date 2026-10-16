import os
import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from preprocessor import TextPreprocessor

MODEL_DIR = "models"

class NewsRecommender:
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
        self.corpus_df = pd.read_pickle(os.path.join(MODEL_DIR, "indexed_corpus.pkl"))
        
        # Precompute matrix for the saved corpus
        self.corpus_matrix = self.vectorizer.transform(self.corpus_df['cleaned_text'])

    def recommend(self, input_text: str, top_k: int = 4):
        """Find the top-k most semantically related articles in the BBC dataset."""
        cleaned_input = self.preprocessor.tokenize_and_lemmatize(input_text)
        query_vec = self.vectorizer.transform([cleaned_input])
        
        # Calculate Cosine Similarities
        similarities = cosine_similarity(query_vec, self.corpus_matrix).flatten()
        
        # Top-K matching indices
        top_indices = similarities.argsort()[::-1][:top_k]
        
        results = []
        text_col = 'text' if 'text' in self.corpus_df.columns else 'Article'
        label_col = 'category' if 'category' in self.corpus_df.columns else 'Category'
        
        for idx in top_indices:
            results.append({
                "category": self.corpus_df.iloc[idx][label_col],
                "snippet": self.corpus_df.iloc[idx][text_col][:250] + "...",
                "similarity_score": round(float(similarities[idx]), 4)
            })
            
        return results
