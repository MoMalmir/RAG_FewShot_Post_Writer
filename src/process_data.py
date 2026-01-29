import pandas as pd
import time
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

"""
I built this script to automate the extraction of expert financial content 
from Capital Group's website. It includes a validation layer to automatically 
filter out broken links or redirects, ensuring that only high-quality, 
relevant article text is passed down the pipeline for summarization 
and indexing.
"""


def get_cleaned_article_text(url):
    """
    Extracts core article content using a targeted DOM selection strategy.
    Includes validation to skip redirected or expired links.
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
        # We define binary_location if the system doesn't find it automatically
        options.binary_location = "/usr/bin/chromium-browser"

        driver = webdriver.Chrome(service=service, options=options)
        driver.get(url)
        time.sleep(7)

        final_url = driver.current_url
        base_valid_path = "https://www.capitalgroup.com/advisor/insights/articles/"
        if len(final_url) < len(base_valid_path) or "404" in final_url:
            driver.quit()
            return None

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
        driver.quit()
        return content if len(content) > 150 else None

    except Exception as e:
        print(f"Extraction failed for {url}: {str(e)}")
        # If /usr/bin/chromedriver failed, try without the specific path once
        # (Some WSL setups prefer the default service)
        return None


def main():
    """
    Main execution loop for pre-processing the training dataset.
    Generates a localized cache of scraped content to optimize performance
    and data integrity for the generation phase.
    """
    TRAIN_PATH = "data/train.xlsx"
    if not os.path.exists(TRAIN_PATH):
        print(f"Error: Required source file '{TRAIN_PATH}' not found.")
        return

    print("Loading source data...")
    df_train = pd.read_excel(TRAIN_PATH, engine="openpyxl")
    processed_train = []

    print(f"Initiating content extraction for {len(df_train)} records...")

    for idx, row in df_train.iterrows():
        url = row["URL"]
        print(f"[{idx+1}/{len(df_train)}] Processing entry: {url}")

        content = get_cleaned_article_text(url)

        if content:
            processed_train.append(
                {
                    "postId": row.get("postId", idx),
                    "URL": url,
                    "postOriginal": row["postOriginal"],
                    "cleanedArticle": content,
                }
            )
        else:
            # Note: Link expiration or archived content is handled as a skip
            print(f"--- Warning: Content extraction returned null for {url} ---")

    # Persist the processed data for downstream indexing and model inference
    output_df = pd.DataFrame(processed_train)
    output_df.to_excel("data/processed_train.xlsx", index=False)
    print(f"\nPipeline Complete: {len(output_df)} records successfully cached.")


if __name__ == "__main__":
    main()
