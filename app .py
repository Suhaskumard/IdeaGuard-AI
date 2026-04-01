import pandas as pd
from originality_model import originality_analysis
from patent_checker import patent_similarity
from abstract_generator import generate_abstract

def analyze_single_idea():
    print("\n🧠 AI-Based Idea Originality Intelligence System")
    print("--------------------------------------------------")

    idea = input("\n✍️ Enter Your Project / Research Idea:\n")

    print("\n🔍 Analyzing...\n")

    score, reasons = originality_analysis(idea)
    patent_score = patent_similarity(idea)

    print(f"🎯 Originality Score: {score}%")
    print(f"📜 Patent Similarity Risk: {patent_score}%\n")

    if reasons:
        print("⚠️ Why your score is low:")
        for r in reasons:
            print(f"- {r}")
    else:
        print("✅ Strong originality indicators detected")

    print("\n📄 Auto-Generated Research Abstract:\n")
    print(generate_abstract(idea))


def analyze_csv():
    file_path = input("\n📂 Enter CSV file path (with column 'idea'): ")

    try:
        df = pd.read_csv(file_path)

        df["Originality_Score"] = df["idea"].apply(lambda x: originality_analysis(x)[0])
        df["Patent_Risk"] = df["idea"].apply(patent_similarity)

        print("\n✅ Analysis Complete:\n")
        print(df)

        save = input("\n💾 Save results to new CSV? (y/n): ")
        if save.lower() == 'y':
            output_path = "output_results.csv"
            df.to_csv(output_path, index=False)
            print(f"📁 Saved to {output_path}")

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    print("\n1. Analyze Single Idea")
    print("2. Analyze CSV File")

    choice = input("\nEnter choice (1/2): ")

    if choice == "1":
        analyze_single_idea()
    elif choice == "2":
        analyze_csv()
    else:
        print("❌ Invalid choice")
