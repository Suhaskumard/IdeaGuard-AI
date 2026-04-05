import numpy as np
import spacy
import yake
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load models once
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

# ✅ Precompute domain embeddings (optimization)
domain_embeddings = {
    d: embedder.encode(words) for d, words in DOMAINS.items()
}


# 🔹 Preprocess text
def preprocess(text):
    doc = nlp(text.lower())
    return " ".join([t.lemma_ for t in doc if not t.is_stop and t.is_alpha])


# 🔹 Domain fusion score (optimized)
def domain_score(idea_emb):
    score = 0
    for emb_list in domain_embeddings.values():
        if cosine_similarity(idea_emb, emb_list).max() > 0.5:
            score += 1
    return min(score / 3, 1)


# 🔹 Main function
def originality_analysis(idea):
    clean_idea = preprocess(idea)

    # Embedding
    emb = embedder.encode([clean_idea])

    # Semantic uniqueness
    sim = cosine_similarity(emb, ref_embeddings).max()
    semantic = 1 - sim

    # Keyword novelty (YAKE fix)
    keywords = kw_extractor.extract_keywords(clean_idea)
    novelty = min(len([k for k, s in keywords if s < 0.5]) / 5, 1)

    # Structural richness
    doc = nlp(clean_idea)
    verbs = sum(1 for t in doc if t.pos_ == "VERB")
    nouns = len(list(doc.noun_chunks))
    structure = min(verbs / (nouns + 1), 1)

    # Domain fusion
    fusion = domain_score(emb)

    # Redundancy (cleaner)
    words = clean_idea.split()
    redundancy = max(0, (len(words) - len(set(words))) / 5)

    # Final score
    score = (
        0.35 * semantic +
        0.25 * novelty +
        0.2 * structure +
        0.2 * fusion
    ) - (0.15 * redundancy)

    score = max(score, 0)

    # Explanation
    explanation = []
    if semantic < 0.4:
        explanation.append("High similarity to existing ideas")
    if novelty < 0.4:
        explanation.append("Lacks novel keywords")
    if fusion < 0.3:
        explanation.append("Limited domain diversity")
    if redundancy > 0.2:
        explanation.append("Repetition detected")

    return round(score * 100, 2), explanation
