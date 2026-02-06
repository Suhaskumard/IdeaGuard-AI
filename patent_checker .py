
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

PATENT_LIKE_IDEAS = [
    "Automated appointment scheduling system",
    "AI system for document similarity analysis",
    "Predictive analytics for human behavior",
    "Automated academic plagiarism detection"
]

embedder = SentenceTransformer("all-MiniLM-L6-v2")
patent_embeddings = embedder.encode(PATENT_LIKE_IDEAS)

def patent_similarity(idea):
    emb = embedder.encode([idea])
    sim = cosine_similarity(emb, patent_embeddings).max()
    return round(sim * 100, 2)
