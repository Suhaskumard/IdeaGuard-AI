import numpy as np
import spacy
import yake
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load("en_core_web_sm")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
kw_extractor = yake.KeywordExtractor(top=10)

REFERENCE_IDEAS = [
    "Fake review detection using sentiment analysis",
    "Resume screening using NLP",
    "Hospital appointment no-show prediction",
    "AI chatbot for mental health",
    "Online fraud detection system"
]

ref_embeddings = embedder.encode(REFERENCE_IDEAS)

DOMAINS = {
    "health": ["hospital","patient","medical"],
    "ai": ["ai","machine learning","model"],
    "education": ["student","learning","exam"],
    "finance": ["fraud","transaction"],
    "psychology": ["behavior","stress","emotion"]
}

def originality_analysis(idea):
    emb = embedder.encode([idea])
    sim = cosine_similarity(emb, ref_embeddings).max()
    semantic = 1 - sim

    keywords = kw_extractor.extract_keywords(idea)
    novelty = min(len([k for k,s in keywords if s > 0.02]) / 5, 1)

    doc = nlp(idea)
    structure = min(len([t for t in doc if t.pos_=="VERB"]) / (len(list(doc.noun_chunks))+1), 1)

    fusion = sum(any(w in idea.lower() for w in words) for words in DOMAINS.values())
    fusion = min(fusion / 3, 1)

    redundancy = max(0, (len(idea.split()) - len(set(idea.split()))) / 5)

    score = (0.35*semantic + 0.25*novelty + 0.2*structure + 0.2*fusion) - 0.15*redundancy
    score = max(score, 0)

    explanation = []
    if semantic < 0.4: explanation.append("High similarity to existing ideas")
    if novelty < 0.4: explanation.append("Lacks rare or novel concepts")
    if fusion < 0.3: explanation.append("Single-domain idea")
    if redundancy > 0.2: explanation.append("Buzzword repetition detected")

    return round(score*100,2), explanation
