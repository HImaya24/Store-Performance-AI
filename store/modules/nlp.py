import spacy
from typing import List, Dict

# Load spaCy model (user should `python -m spacy download en_core_web_sm`)
try:
    nlp = spacy.load('en_core_web_sm')
except Exception:
    nlp = None

def extract_entities(text: str) -> List[Dict]:
    if nlp is None:
        return []
    doc = nlp(text)
    return [{'text': ent.text, 'label': ent.label_} for ent in doc.ents]

# Simple summarizer fallback using transformers/gensim
try:
    from transformers import pipeline
    summarizer = pipeline('summarization')
except Exception:
    summarizer = None
    try:
        from gensim.summarization import summarize as gensim_summarize
    except Exception:
        gensim_summarize = None

def summarize_text(text: str, max_length: int = 120) -> str:
    if summarizer:
        out = summarizer(text, max_length=max_length, min_length=30, do_sample=False)
        return out[0]['summary_text']
    if 'gensim_summarize' in globals() and gensim_summarize:
        try:
            return gensim_summarize(text, word_count=60)
        except Exception:
            return text[:max_length]
    return text[:max_length]
