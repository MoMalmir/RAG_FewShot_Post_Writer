import os
import time
import pickle
import faiss
import numpy as np
import pandas as pd
import re
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

load_dotenv()


"""
I designed this final production pipeline to synthesize all previous steps 
into a robust content engine. It integrates real-time web extraction with 
my hybrid RRF and MMR retrieval system to provide the LLM with diverse, 
high-quality few-shot examples. The logic includes a custom sanitization 
layer to ensure every generated post strictly adheres to Capital Group’s 
professional brand voice and compliance standards.
"""


# --- CONFIGURATION ---
MODEL_NAME = "anthropic/claude-sonnet-4.5"
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
client = OpenAI(
    base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY")
)

# --- DATA PATHS ---
TRAIN_DATA = "data/processed_train_with_summaries.xlsx"
TEST_DATA = "data/test.xlsx"
FAISS_INDEX = "data/train_index.faiss"
BM25_INDEX = "data/bm25_index.pkl"
OUTPUT_FILE = "data/submission_long_format.xlsx"


# --- HELPER: ADVANCED CONTENT VALIDATOR ---
def get_test_article_context(url):
    """
    Extracts core article content using a targeted DOM selection strategy.
    Includes validation to skip redirected or expired links, consistent with training.
    """
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    )

    # Path configuration for WSL/Linux system binaries
    service = Service("/usr/bin/chromedriver")

    try:
        # Define binary_location for system-installed Chromium
        options.binary_location = "/usr/bin/chromium-browser"

        driver = webdriver.Chrome(service=service, options=options)
        driver.get(url)
        time.sleep(7)

        # Consistent Validation Logic
        final_url = driver.current_url
        base_valid_path = "https://www.capitalgroup.com/advisor/insights/articles/"

        # Check if redirected to a general page or a 404
        is_invalid_url = len(final_url) < len(base_valid_path) or "404" in final_url

        soup = BeautifulSoup(driver.page_source, "html.parser")
        content_blocks = soup.find_all("div", class_="cmp-text")

        article_paragraphs = []
        for block in content_blocks:
            paragraphs = block.find_all("p")
            for p in paragraphs:
                text_node = p.get_text(strip=True)
                if len(text_node) > 60:
                    article_paragraphs.append(text_node)

        content = " ".join(article_paragraphs)
        title = (
            soup.title.string.split("|")[0].strip()
            if soup.title
            else "Capital Group Insight"
        )
        driver.quit()

        # Consistent Quality Check: Redirected or insufficient content (<= 150 chars)
        if is_invalid_url or len(content) <= 150:
            # Trigger Fallback to ensure we still have a topic for the LLM
            inferred_topic = (
                url.split("/")[-1].replace("-", " ").replace(".html", "").title()
            )
            return {
                "type": "Metadata/Slug Fallback",
                "title": inferred_topic,
                "content": f"Topic: {inferred_topic}",
            }

        # If the article is valid and meets the quality bar
        return {"type": "Full Article", "title": title, "content": content[:3000]}

    except Exception as e:
        if "driver" in locals():
            driver.quit()
        print(f"Extraction failed for {url}: {str(e)}")
        # Final safety fallback based on URL slug
        inferred_topic = (
            url.split("/")[-1].replace("-", " ").replace(".html", "").title()
        )
        return {
            "type": "System Error Fallback",
            "title": inferred_topic,
            "content": f"Topic: {inferred_topic}",
        }


# --- HELPER: CLEANING TEXT ---
def clean_llm_chatter(raw_variations):
    """
    Removing conversational intros without harming valid content.
    """
    refined_variations = []

    # Pattern to match common LLM intro phrases at the START of a string
    intro_pattern = re.compile(
        r"^(here are|certainly|sure|i have|variation \d:)", re.IGNORECASE
    )

    for v in raw_variations:
        v = v.strip()

        # Check if the very beginning of the block matches an intro pattern
        if intro_pattern.match(v):
            # Split at the first newline to separate 'chat' from 'post'
            parts = v.split("\n", 1)
            if len(parts) > 1 and len(parts[1].strip()) > 30:
                v = parts[1].strip()  # Keep only the content after the intro
            else:
                continue  # Skip if it was purely a chatter block

        if len(v) > 40:
            refined_variations.append(v)

    return refined_variations


def get_hybrid_few_shots(
    query_text, df_train, faiss_idx, bm25_idx, n=3, lambda_param=0.5
):
    """
    Retrieves diverse examples using both semantic (FAISS) and keyword (BM25) matching.
    Implements Reciprocal Rank Fusion (RRF) and Maximum Marginal Relevance (MMR).
    n: number of examples to return
    lambda_param: balance between relevance (1) and diversity (0)
    """
    # 1. Get Top 10 from Semantic (FAISS)
    query_vec = EMBED_MODEL.encode([query_text]).astype("float32")
    faiss.normalize_L2(query_vec)
    _, faiss_indices = faiss_idx.search(query_vec, 10)

    # 2. Get Top 10 from Keyword (BM25)
    tokenized_query = query_text.lower().split()
    bm25_scores = bm25_idx.get_scores(tokenized_query)
    bm25_indices = np.argsort(bm25_scores)[-10:][::-1]

    # 3. Reciprocal Rank Fusion (RRF) - Combines both lists fairly
    all_candidates = list(set(faiss_indices[0]) | set(bm25_indices))
    rrf_scores = {}
    for idx in all_candidates:
        # RRF Score = 1 / (k + rank)
        f_rank = (
            np.where(faiss_indices[0] == idx)[0][0] if idx in faiss_indices[0] else 100
        )
        b_rank = np.where(bm25_indices == idx)[0][0] if idx in bm25_indices else 100
        rrf_scores[idx] = (1 / (60 + f_rank)) + (1 / (60 + b_rank))

    # Sort candidates by hybrid RRF score
    sorted_candidates = sorted(
        rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True
    )
    candidate_df = df_train.iloc[sorted_candidates]

    # 4. MMR (Maximum Marginal Relevance) for Diversity
    # We want 3 examples that are relevant but different from each other
    candidate_embeddings = EMBED_MODEL.encode(candidate_df["searchSummary"].tolist())
    selected_indices = [0]  # Start with the best hybrid match

    while len(selected_indices) < n:
        remaining_indices = [
            i for i in range(len(candidate_df)) if i not in selected_indices
        ]

        # Calculate similarity of remaining candidates to already selected ones
        similarities_to_selected = cosine_similarity(
            candidate_embeddings[remaining_indices],
            candidate_embeddings[selected_indices],
        )

        # MMR Formula: lambda * similarity_to_query - (1 - lambda) * max_similarity_to_selected
        # Note: Since we sorted by RRF, we use the RRF rank as a proxy for query relevance
        mmr_scores = []
        for i, idx in enumerate(remaining_indices):
            relevance = 1.0 / (i + 1)  # Position in sorted candidates
            redundancy = np.max(similarities_to_selected[i])
            score = lambda_param * relevance - (1 - lambda_param) * redundancy
            mmr_scores.append(score)

        selected_indices.append(remaining_indices[np.argmax(mmr_scores)])

    final_samples = candidate_df.iloc[selected_indices]
    raw_chunks = final_samples['postOriginal'].tolist()
    # Format for prompt
    examples_str = ""
    for i, (_, row) in enumerate(final_samples.iterrows()):
        examples_str += f"### EXAMPLE {i+1}\nTOPIC: {row['searchSummary']}\nPOST: {row['postOriginal']}\n\n"
    return examples_str, raw_chunks


# --- MAIN GENERATOR ---
def main():
    print("Initiating production pipeline...")
    df_train = pd.read_excel(TRAIN_DATA)
    df_test = pd.read_excel(TEST_DATA)
    faiss_idx = faiss.read_index(FAISS_INDEX)
    with open(BM25_INDEX, "rb") as f:
        bm25_idx = pickle.load(f)

    final_results = []

    for idx, row in df_test.iterrows():
        post_id = row["postId"]
        url = row["URL"]
        print(f"[{idx+1}/{len(df_test)}] Processing ID {post_id}...")

        context_data = get_test_article_context(url)
        few_shots = get_hybrid_few_shots(
            context_data["title"], df_train, faiss_idx, bm25_idx
        )

        # Updated system prompt with negative constraints to stop conversational filler
        system_prompt = (
            "You are a Senior Social Media Strategist at Capital Group. "
            "Replicate the firm's brand voice: authoritative, insightful, and sophisticated. "
            "STRICT RULES:\n"
            "1. Do NOT use hashtags (#).\n"
            "2. Do NOT use mathematical symbols like '+' or '=' as shorthand for words.\n"
            "3. Do NOT include introductory text or conversational filler.\n"
            "4. Do NOT include the 'Important disclosures' link in your response; it will be added automatically."
        )

        user_prompt = f"""
        INSTRUCTIONS:
        Analyze the approved posts for tone. Notice they use full words and professional punctuation.
        {few_shots}
        
        ---
        NEW ASSIGNMENT:
        Topic Title: {context_data['title']}
        Provided Context: {context_data['content']}
        
        TASK:
        Generate 4 distinct social media variations for this topic. 
        Focus on professional, clean English.
        Separate variations with '==='.
        """

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
            )

            raw_content = response.choices[0].message.content
            # Split by the delimiter
            initial_splits = [
                v.strip() for v in raw_content.split("===") if len(v.strip()) > 20
            ]

            # Apply the cleaner function to remove "Here are your posts..."
            actual_posts = clean_llm_chatter(initial_splits)

            # Process the final variations (Take top 4)
            for i, post_text in enumerate(actual_posts[:4]):
                # Remove hashtags and nonsensical symbols programmatically
                clean_text = post_text.replace("#", "").replace(" + ", " and ")

                # Append disclosure with standardized spacing
                final_post = f"{clean_text.strip()}\n\nImportant disclosures: https://bit.ly/2JzEDWl"

                final_results.append(
                    {
                        "postId": post_id,
                        "URL": url,
                        "variationNumber": i + 1,
                        "postText": final_post,
                        "extractionType": context_data["type"],
                    }
                )

        except Exception as e:
            print(f"Model Error on ID {post_id}: {str(e)}")

    # Save final results
    pd.DataFrame(final_results).to_excel(OUTPUT_FILE, index=False)
    print(f"Pipeline finished. Data saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
