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

def preprocess(text):
    doc = nlp(text.lower())
    return " ".join([t.lemma_ for t in doc if not t.is_stop and t.is_alpha])

def domain_score(idea_emb):
    score = 0
    for words in DOMAINS.values():
        dom_emb = embedder.encode(words)
        if cosine_similarity(idea_emb, dom_emb).max() > 0.5:
            score += 1
    return min(score / 3, 1)

def originality_analysis(idea):
    clean_idea = preprocess(idea)
    
    emb = embedder.encode([clean_idea])
    sim = cosine_similarity(emb, ref_embeddings).max()
    semantic = 1 - sim

    keywords = kw_extractor.extract_keywords(clean_idea)
    novelty = min(len([k for k,s in keywords if s < 0.5]) / 5, 1)

    doc = nlp(clean_idea)
    verbs = len([t for t in doc if t.pos_=="VERB"])
    nouns = len(list(doc.noun_chunks))
    structure = min(verbs / (nouns + 1), 1)

    fusion = domain_score(emb)

    words = [w for w in clean_idea.split()]
    redundancy = max(0, (len(words) - len(set(words))) / 5)

    score = (0.35*semantic + 0.25*novelty + 0.2*structure + 0.2*fusion) - 0.15*redundancy
    score = max(score, 0)

    explanation = []
    if semantic < 0.4: explanation.append("High similarity to existing ideas")
    if novelty < 0.4: explanation.append("Lacks novel keywords")
    if fusion < 0.3: explanation.append("Limited domain diversity")
    if redundancy > 0.2: explanation.append("Repetition detected")

    return round(score*100,2), explanation
