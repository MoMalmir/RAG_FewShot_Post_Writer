# RAG_FewShot_Post_Writer

This is a specialized **Retrieval-Augmented Generation (RAG)** framework designed to translate complex investment articles into sophisticated, institutional-grade social media content. By leveraging a hybrid retrieval strategy and persona-anchored few-shot learning, the system replicates the authoritative voice of a Senior Investment Writer at firms like Capital Group.

##  Key Features

* **Hybrid Retrieval Engine:** Combines **FAISS** (Dense/Semantic) and **BM25** (Sparse/Keyword) search to identify the most relevant stylistic benchmarks.
* **Reciprocal Rank Fusion (RRF):** Merges disparate retrieval lists to balance keyword precision with semantic depth.
* **MMR Diversity Logic:** Implements **Maximum Marginal Relevance** to select few-shot examples that are relevant yet stylistically diverse, preventing model redundancy.
* **Persona-Driven Generation:** A specialized prompt engineering architecture that enforces institutional sobriety, sophisticated vocabulary (e.g., *secular trends*, *valuation dispersion*), and strict negative constraints.
* **Comprehensive Evaluation Suite:** Moving beyond standard metrics, the project utilizes:
    * **Lexical:** ROUGE-1/2/L.
    * **Semantic:** BERTScore (F1).
    * **LLM-as-a-Judge:** Custom **DeepEval** metrics for "Financial Sophistication" and factual grounding.

##  Architecture


1.  **Data Preprocessing & Cleaning:** * **URL Validation:** Removed cases where URLs did not direct to a valid article to ensure data integrity.
    * **Efficient Scraping:** Scraped and extracted article content in a single batch after filtering to minimize overhead.
    * **Context Summarization:** Generated LLM-based summaries for each article to serve as the foundation for search indices, improving retrieval speed and relevance.
2.  **Retrieval:** When a new article is processed, the system searches the database of expert posts using a hybrid FAISS/BM25 approach.
3.  **Few-Shot Examples:** The top 3 most relevant and diverse examples are added to the prompt to show the LLM exactly how to write.
4.  **Generation:** The LLM (Claude 3.5/4.5) generates the post based on the new article and the provided examples.
5.  **Cleaning:** A post-processing script removes any "LLM chatter" (like "Sure, here are your posts") to make the output production-ready.


## 📊 Evaluation Results

The system was validated against a "Golden Evaluation Set" of expert-written posts.

| Metric | Score | Insight |
| :--- | :--- | :--- |
| **BERTScore (F1)** | **~0.82** | High semantic alignment with expert investment theses. |
| **GEval Tone** | **~0.81** | Successfully replicates authoritative, institutional brand voice. |
| **Hallucination** | **0.00** | 100% factual grounding in source article content. |
| **ROUGE-L** | **~0.23** | Demonstrates creative synthesis rather than verbatim copying. |

**Note:** For the full detailed evaluation of all test cases, please navigate to the `data/` folder and view the `evaluation_dashboard.xlsx` file.