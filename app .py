import streamlit as st
import pandas as pd
from originality_model import originality_analysis
from patent_checker import patent_similarity
from abstract_generator import generate_abstract

st.set_page_config(page_title="AI Originality Intelligence", layout="wide")

st.title("🧠 AI-Based Idea Originality Intelligence System")

idea = st.text_area("✍️ Enter Your Project / Research Idea")

if st.button("Analyze Originality"):
    score, reasons = originality_analysis(idea)
    patent_score = patent_similarity(idea)

    st.subheader(f"🎯 Originality Score: {score}%")
    st.subheader(f"📜 Patent Similarity Risk: {patent_score}%")

    if reasons:
        st.error("Why your score is low:")
        for r in reasons:
            st.write("•", r)
    else:
        st.success("Strong originality indicators detected")

    st.subheader("📄 Auto-Generated Research Abstract")
    st.write(generate_abstract(idea))

st.sidebar.header("📂 CSV Batch Idea Scoring")
file = st.sidebar.file_uploader("Upload CSV with column: idea", type="csv")

if file:
    df = pd.read_csv(file)
    df["Originality_Score"] = df["idea"].apply(lambda x: originality_analysis(x)[0])
    df["Patent_Risk"] = df["idea"].apply(patent_similarity)
    st.dataframe(df)
