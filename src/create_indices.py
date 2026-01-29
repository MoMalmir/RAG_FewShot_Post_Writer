import pandas as pd
import numpy as np
import faiss
import pickle
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

"""
I built this script to create a "Search Brain" that uses two different methods 
to find the best examples for few-shot learning: BM25 for exact keywords and FAISS for the overall 
meaning. I chose to index the summaries instead of the full articles to 
remove noise and make the matching more accurate for the AI.
"""


def main():
    # 1. Load data
    df = pd.read_excel("data/processed_train_with_summaries.xlsx")
    # We index the Summary because it's less noisy than the full article
    documents = df["searchSummary"].astype(str).tolist()

    print(f"Indexing {len(documents)} article summaries...")

    # 2. CREATE BM25 (Keyword Index)
    tokenized_corpus = [doc.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)

    # 3. CREATE FAISS (Semantic Index)
    # This model 'understands' financial concepts
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(documents, show_progress_bar=True).astype("float32")

    # Normalize for Cosine Similarity
    faiss.normalize_L2(embeddings)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    # 4. SAVE INDICES
    faiss.write_index(index, "data/train_index.faiss")
    with open("data/bm25_index.pkl", "wb") as f:
        pickle.dump(bm25, f)

    print("\nSUCCESS: Hybrid indices saved to 'data/'. ready for generation.")


if __name__ == "__main__":
    main()
