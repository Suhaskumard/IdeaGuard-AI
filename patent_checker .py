import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load("en_core_web_sm")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

PATENT_LIKE_IDEAS = [
    "Automated appointment scheduling system",
    "AI system for document similarity analysis",
    "Predictive analytics for human behavior",
    "Automated academic plagiarism detection"
]

patent_embeddings = embedder.encode(PATENT_LIKE_IDEAS)


# ✅ Text preprocessing
def preprocess(text):
    doc = nlp(text.lower())
    return " ".join([t.lemma_ for t in doc if not t.is_stop and t.is_alpha])


# ✅ Improved patent similarity function
def patent_similarity(idea):
    clean_idea = preprocess(idea)

    emb = embedder.encode([clean_idea])
    similarities = cosine_similarity(emb, patent_embeddings)[0]

    max_sim = similarities.max()
    avg_sim = similarities.mean()

    # Combine max + avg for stability
    risk_score = (0.7 * max_sim + 0.3 * avg_sim)

    return round(risk_score * 100, 2)
