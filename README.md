# AI-Based Idea Originality Score 🚀

> An AI-powered platform that evaluates the originality, innovation potential, and patent similarity of research ideas, startup concepts, and academic projects using advanced NLP and semantic similarity analysis.

The system helps researchers, students, and innovators identify duplicate concepts, measure uniqueness, and generate research-oriented insights before project development or patent filing.

---

# ✨ Features

- ✅ AI-based originality scoring (0–100)
- ✅ Patent similarity risk estimation
- ✅ Semantic similarity detection using transformer embeddings
- ✅ Batch idea analysis using CSV datasets
- ✅ Explainable AI feedback for low originality scores
- ✅ Automatic research abstract generation
- ✅ Keyword extraction and concept analysis
- ✅ Duplicate and near-duplicate idea detection 
- ✅ Research-focused NLP pipeline
- ✅ Lightweight and notebook-friendly architecture

---
 
# 🧠 Core Functionalities

| Module | Purpose |
|---|---|
| Originality Scoring | Measures uniqueness of ideas |
| Patent Similarity Detection | Finds semantic overlap with existing concepts |
| Abstract Generation | Creates research-style abstracts automatically |
| Explainability Engine | Explains why originality is low |
| Batch Processing | Scores multiple ideas from CSV files |

---

# 🛠 Tech Stack

## Programming Language
- Python

## AI / NLP Libraries
- Sentence Transformers
- spaCy
- scikit-learn
- NLTK
- YAKE (Keyword Extraction)

## Data Processing
- Pandas
- NumPy

## NLP Techniques Used
- Semantic Similarity
- Sentence Embeddings
- Cosine Similarity
- Keyword Extraction
- Text Vectorization
- Research Abstract Generation

---

# 📂 Project Structure

```text
AI-Idea-Originality-Score/
│
├── abstract_generator.py      # Generates AI-based research abstracts
├── originality_model.py       # Originality scoring engine
├── patent_checker.py          # Patent similarity analysis
├── app.py                     # Main application entry
├── ideas.csv                  # Sample dataset for batch scoring
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

---

# ⚙️ Installation

## Clone Repository

```bash
git clone <repository-url>

cd AI-Idea-Originality-Score
```

---

## Install Dependencies

```bash
pip install sentence-transformers scikit-learn spacy yake pandas nltk
```

---

## Download spaCy Model

```bash
python -m spacy download en_core_web_sm
```

---

# ▶️ How to Run

## Option 1 — Jupyter Notebook / Google Colab

1. Open the project in:
   - Jupyter Notebook
   - Google Colab

2. Install dependencies

3. Run the Python scripts or notebook cells

---

## Option 2 — Run Using Python

```bash
python app.py
```

---

# 📊 Sample Workflow

```text
User Idea
   ↓
Text Preprocessing
   ↓
Embedding Generation
   ↓
Semantic Similarity Analysis
   ↓
Originality Score Calculation
   ↓
Patent Risk Estimation
   ↓
Explainable Feedback
   ↓
Research Abstract Generation
```

---

# 📈 Output

The system generates:

- 📌 Originality Score (%)
- 📌 Patent Similarity Risk (%)
- 📌 Semantic Similarity Metrics
- 📌 Explainable Feedback
- 📌 Auto-generated Research Abstract
- 📌 Extracted Keywords & Concepts

---

# 🧪 Example Use Cases

- Research project validation
- Startup idea uniqueness analysis
- Patent pre-screening
- Hackathon idea evaluation
- Academic innovation assessment
- Research paper proposal analysis

---

# 🔍 Future Enhancements

- Integration with real patent databases
- Deep learning-based novelty detection
- Research paper recommendation engine
- Web dashboard for visualization
- Multi-language idea analysis
- AI chatbot for research guidance

---

# 📦 Requirements

- Python 3.10+
- Internet connection (for model downloads)
- Jupyter Notebook or Google Colab (optional)

---

# 📄 License

This project is intended for academic, research, and educational purposes.

---
