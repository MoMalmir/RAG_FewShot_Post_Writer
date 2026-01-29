import pandas as pd
from openai import OpenAI
import os
from dotenv import load_dotenv
import time
from sklearn.model_selection import train_test_split

load_dotenv()

"""
I use this script to distill 138 training articles into dense, two-sentence 
financial summaries. By compressing long articles into their 'core thesis,' 
I remove linguistic noise and create high-quality, semantically rich data 
that makes the hybrid search much more accurate.
"""


# Initialize Client (Using a fast, cheap model for bulk summarization)
client = OpenAI(
    base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY")
)
MODEL = "anthropic/claude-3-haiku"


def generate_search_summary(text):
    """Generates a dense, technical summary focused on the investment thesis."""
    try:
        prompt = f"""Summarize the core financial thesis and key market insights of this article into 2-3 dense sentences. 
        Focus on the 'what' and 'why' so this can be used for topic matching.
        
        Article: {text[:5000]}"""

        response = client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": prompt}], temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error summarising: {e}")
        return None


def main():
    print("Loading processed training data...")
    df = pd.read_excel("data/processed_train.xlsx")

    print(f"Generating summaries for {len(df)} records...")
    summaries = []

    for idx, row in df.iterrows():
        print(f"[{idx+1}/{len(df)}] Summarising: {row['URL']}")
        summary = generate_search_summary(row["cleanedArticle"])
        summaries.append(summary)
        time.sleep(0.5)

    df["searchSummary"] = summaries

    # Split the data to train and golden evaluation and save the new version with summaries
    print("\nSplitting data into Index Set (85%) and Golden Set (15%)...")
    
    # random_state=42 ensures we get the same split every time you run it
    df_index, df_golden = train_test_split(df, test_size=0.15, random_state=42)

    df_index.to_excel("data/processed_train_with_summaries.xlsx", index=False)
    
    df_golden.to_excel("data/golden_evaluation_set.xlsx", index=False)

    print(f"SUCCESS:")
    print(f" - Index Set: {len(df_index)} records saved to 'data/processed_train_with_summaries.xlsx'")
    print(f" - Golden Set: {len(df_golden)} records saved to 'data/golden_evaluation_set.xlsx'")


if __name__ == "__main__":
    main()
