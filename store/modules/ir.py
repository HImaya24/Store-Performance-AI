import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from pathlib import Path
from utils.config import DATA_PATH

class SimpleIR:
    def __init__(self):
        self.df = None
        self.vectorizer = None
        self.tfidf_matrix = None

    def load_data(self, path: str = None):
        p = path or DATA_PATH
        self.df = pd.read_csv(p)
        # Ensure a 'text' column exists; if not, fallback to concatenation
        if 'text' not in self.df.columns:
            # join relevant columns into a single text field
            self.df['text'] = self.df.fillna('').astype(str).agg(' '.join, axis=1)
        return self.df

    def build_index(self, max_features: int = 5000):
        texts = self.df['text'].fillna('')
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features)
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)

    def query(self, q: str, top_k: int = 5):
        q_vec = self.vectorizer.transform([q])
        cosine_similarities = linear_kernel(q_vec, self.tfidf_matrix).flatten()
        top_indices = cosine_similarities.argsort()[-top_k:][::-1]
        results = []
        for idx in top_indices:
            results.append({
                'index': int(idx),
                'score': float(cosine_similarities[idx]),
                'row': self.df.iloc[idx].to_dict()
            })
        return results
