import os
import sys
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, accuracy_score

# Ensure src can import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessor import TextPreprocessor

DATA_PATH = os.path.join("data", "bbc-text.csv")
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def train_and_evaluate():
    print("[*] Loading BBC News Dataset...")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Missing {DATA_PATH}. Place your BBC dataset there first.")

    df = pd.read_csv(DATA_PATH)
    text_col = 'text' if 'text' in df.columns else 'Article'
    label_col = 'category' if 'category' in df.columns else 'Category'

    print(f"[*] Total articles loaded: {len(df)}")
    print("[*] Running Preprocessing and Lemmatization Pipeline...")
    preprocessor = TextPreprocessor()
    df['cleaned_text'] = preprocessor.transform_series(df[text_col])

    # Save cleaned corpus for similarity lookups
    df.to_pickle(os.path.join(MODEL_DIR, "indexed_corpus.pkl"))

    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_text'], df[label_col], test_size=0.2, random_state=42, stratify=df[label_col]
    )

    print("[*] Generating TF-IDF Vector Space (Unigrams + Bigrams)...")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, sublinear_tf=True)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Models benchmark tournament
    models = {
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.1),
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0),
        "Linear SVM": CalibratedClassifierCV(LinearSVC(C=1.0, random_state=42))
    }

    best_model = None
    best_acc = 0.0
    best_model_name = ""

    print("\n--- MODEL BENCHMARK TOURNAMENT ---")
    for name, clf in models.items():
        clf.fit(X_train_vec, y_train)
        preds = clf.predict(X_test_vec)
        acc = accuracy_score(y_test, preds)
        print(f"-> {name} Accuracy: {acc * 100:.2f}%")
        
        if acc > best_acc:
            best_acc = acc
            best_model = clf
            best_model_name = name

    print(f"\n[✓] Winner: {best_model_name} with {best_acc * 100:.2f}% Accuracy")
    print("\nDetailed Performance Report:")
    print(classification_report(y_test, best_model.predict(X_test_vec)))

    # Persist artifacts
    joblib.dump(best_model, os.path.join(MODEL_DIR, "classifier.pkl"))
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
    joblib.dump(best_model.classes_, os.path.join(MODEL_DIR, "classes.pkl"))
    print("[✓] Model artifacts saved to 'models/' directory successfully.")

if __name__ == "__main__":
    train_and_evaluate()
