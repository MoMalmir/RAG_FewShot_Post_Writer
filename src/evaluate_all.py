import os
import pandas as pd
import numpy as np
import pickle
import faiss
from openai import OpenAI
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from deepeval.metrics import GEval, HallucinationMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.models.base_model import DeepEvalBaseLLM  
# from ragas import evaluate
# from ragas.metrics import Faithfulness, ContextPrecision
# from ragas.llms import llm_factory
# from datasets import Dataset

# --- LangChain & Main Imports ---
from langchain_openai import ChatOpenAI 
from main import (
    get_test_article_context, get_hybrid_few_shots, 
    clean_llm_chatter, client, MODEL_NAME, 
    TRAIN_DATA, FAISS_INDEX, BM25_INDEX
)

# 1. WRAPPER CLASS FOR DEEPEVAL
class DeepEvalOpenRouterWrapper(DeepEvalBaseLLM):
    def __init__(self, model):
        self.model = model

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        return chat_model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        res = await chat_model.ainvoke(prompt)
        return res.content

    def get_model_name(self):
        return "OpenRouter Claude 4.5 Sonnet"

def run_comprehensive_evaluation():
    # Setup OpenRouter Judge
    langchain_llm = ChatOpenAI(
        model_name=MODEL_NAME, 
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
        default_headers={"HTTP-Referer": "http://localhost:3000", "X-Title": "FinScribe Eval"}
    )
    
    # Wrap the model for DeepEval
    deepeval_model = DeepEvalOpenRouterWrapper(langchain_llm)

    # Load Resources
    print("Loading resources and Golden Evaluation Set...")
    df_train = pd.read_excel(TRAIN_DATA)
    faiss_idx = faiss.read_index(FAISS_INDEX)
    with open(BM25_INDEX, "rb") as f:
        bm25_idx = pickle.load(f)
    
    df_golden = pd.read_excel("data/golden_evaluation_set.xlsx")
    results_for_df = []

    for idx, row in df_golden.iterrows():
        url = row["URL"]
        print(f"[{idx+1}/{len(df_golden)}] Generating: {url}")

        context_data = get_test_article_context(url)
        few_shots = get_hybrid_few_shots(context_data["title"], df_train, faiss_idx, bm25_idx)

        system_prompt = (
            "You are a Senior Investment Writer and Social Media Strategist at Capital Group. "
            "Your objective is to translate complex investment insights into sophisticated social commentary for institutional and high-net-worth audiences. "
            
            "\nBRAND VOICE PARAMETERS:"
            "\n- TONE: Authoritative, institutional, and steady. Avoid 'hype' or 'marketing speak'."
            "\n- VOCABULARY: Utilize industry-standard terminology such as 'secular trends', 'capital allocation', 'regime change', or 'valuation dispersion'."
            "\n- STRUCTURE: Use active voice and professional punctuation. Lead with the most significant macro or fundamental insight."

            "\nSTRICT NEGATIVE CONSTRAINTS:"
            "\n1. NO hashtags (#)."
            "\n2. NO mathematical shorthand (e.g., do not use '+' for 'and' or '=' for 'results in')."
            "\n3. NO introductory 'chatter' (e.g., 'Sure, here are the posts...'). Output variations only."
            "\n4. NO emojis or exclamation points. Maintain a sober, professional register."
            "\n5. DO NOT mention 'Important disclosures'; this is handled by the post-processing pipeline."
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
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                temperature=0.7,
            )
            raw_content = response.choices[0].message.content
            
            # 1. Split and clean to get a LIST of variations
            variations = [v.strip() for v in raw_content.split("===") if len(v.strip()) > 20]
            actual_posts = clean_llm_chatter(variations)
            
            if actual_posts:
                selected_post = actual_posts[0].strip()
                prediction = f"{selected_post}\n\nImportant disclosures: https://bit.ly/2JzEDWl"
            else:
                prediction = "Error: No valid posts generated."
        except Exception as e:
            prediction = f"Error: {e}"

        results_for_df.append({
            "URL": url,
            "title": context_data["title"],
            "reference": row["postOriginal"],
            "prediction": prediction,
            "context": context_data["content"]
        })

    eval_df = pd.DataFrame(results_for_df)

    # --- STEP B: METRICS ---
    print("Calculating Metrics...")
    
    # 1. ROUGE (1, 2, L)
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge1 = [scorer.score(r, p)['rouge1'].fmeasure for r, p in zip(eval_df['reference'], eval_df['prediction'])]
    rouge2 = [scorer.score(r, p)['rouge2'].fmeasure for r, p in zip(eval_df['reference'], eval_df['prediction'])]
    rougeL = [scorer.score(r, p)['rougeL'].fmeasure for r, p in zip(eval_df['reference'], eval_df['prediction'])]
    eval_df['rouge1'] = rouge1
    eval_df['rouge2'] = rouge2
    eval_df['rougeL'] = rougeL

    print('rouge1: ', rouge1)
    print('rouge2: ',rouge2 )
    print('rougeL: ', rougeL)

    # 2. BERTScore
    _, _, F1 = bert_score(
        eval_df['prediction'].tolist(), 
        eval_df['reference'].tolist(), 
        model_type="distilbert-base-uncased"
    )
    eval_df['bertscore_f1'] = F1.tolist()
    
    print(F1.tolist())

    # 3. DeepEval (Tone Check)
    
    tone_metric = GEval(
        name="Financial Sophistication",
        criteria="Score 1-10 on whether the tone is expert, authoritative, and professional.",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        model=deepeval_model  
    )
    # This metric checks if the post is grounded in the context
    hallucination_metric = HallucinationMetric(threshold=0.5, model=deepeval_model)

    tone_scores = []
    hallucination_scores = []

    # for pred in eval_df['prediction']:
    #     test_case = LLMTestCase(input="N/A", actual_output=pred)
    #     tone_metric.measure(test_case)
    #     tone_scores.append(tone_metric.score)
    # eval_df['deepeval_tone'] = tone_scores
    
    # print('GEval_tone_metric', tone_scores)

    for pred, ctx in zip(eval_df['prediction'], eval_df['context']):
        # Create a test case that includes the context for hallucination check
        test_case = LLMTestCase(
            input="N/A", 
            actual_output=pred,
            context=[ctx],
            retrieval_context=[ctx] # Context must be a list of strings
        )
        
        # Measure Tone
        tone_metric.measure(test_case)
        tone_scores.append(tone_metric.score)
        
        # Measure Hallucination
        hallucination_metric.measure(test_case)
        hallucination_scores.append(hallucination_metric.score)

    eval_df['deepeval_tone'] = tone_scores
    eval_df['deepeval_hallucination'] = hallucination_scores # 1.0 = No hallucination, 0.0 = Hallucinated
    
    print('deepeval_tone', tone_scores)
    print('deepeval_hallucination', hallucination_scores)

    # # 4. RAGAS (Faithfulness)
    # ragas_client = OpenAI(
    # api_key=os.getenv("OPENROUTER_API_KEY"),
    # base_url="https://openrouter.ai/api/v1"
    # )

    # # Use llm_factory to create InstructorLLM
    # ragas_llm = llm_factory(
    #     model=MODEL_NAME, 
    #     client=ragas_client
    # )

    # f_metric = Faithfulness(llm=ragas_llm)
    # cp_metric = ContextPrecision(llm=ragas_llm)

    # ragas_data = Dataset.from_dict({
    #     "question": [row['title'] for row in results_for_df],
    #     "answer": eval_df['prediction'].tolist(),
    #     "contexts": [[c] for c in eval_df['context'].tolist()],
    #     "ground_truth": eval_df['reference'].tolist() 
    # })
    
    # print("Running RAGAS Analysis (Faithfulness + Context Precision)...")
    # ragas_results = evaluate(
    #     ragas_data, 
    #     metrics=[f_metric, cp_metric],
    #     llm=ragas_llm 
    # )
    
    # # Store scores back into your Excel dashboard
    # ragas_df = ragas_results.to_pandas()
    # eval_df['ragas_faithfulness'] = ragas_df['faithfulness']
    # eval_df['ragas_context_precision'] = ragas_df['context_precision']
    
    # print('ragas_faithfulness: ', ragas_df['faithfulness'])
    # print('ragas_context_precision', ragas_df['context_precision'])

    # --- OUTPUT ---
    output_path = "data/evaluation_dashboard.xlsx"
    eval_df.to_excel(output_path, index=False)
    
    print("\n" + "="*50)
    print("           FINSCRIBE EVALUATION SUMMARY")
    print("="*50)
    
    # --- Lexical Metrics (ROUGE) ---
    print(f"ROUGE-1 (Unigrams):   {eval_df['rouge1'].mean():.4f}")
    print(f"ROUGE-2 (Bigrams):    {eval_df['rouge2'].mean():.4f}")
    print(f"ROUGE-L (Sequence):   {eval_df['rougeL'].mean():.4f}")
    print("-" * 30)

    # --- Semantic Metrics (BERTScore) ---
    print(f"BERTScore (F1):       {eval_df['bertscore_f1'].mean():.4f}")
    print("-" * 30)

    # # --- RAG Metrics (RAGAS) ---
    # # ragas_results is a dict-like object returned by evaluate()
    # print(f"RAGAS Faithfulness:   {eval_df['ragas_faithfulness'].mean():.4f}")
    # print(f"RAGAS Context Prec:   {eval_df['ragas_context_precision'].mean():.4f}")
    
    # --- LLM Judge Metrics (DeepEval) ---
    if 'deepeval_tone' in eval_df.columns:
        print("-" * 30)
        print(f"DeepEval Tone Score:  {eval_df['deepeval_tone'].mean():.4f}")

    if 'deepeval_hallucination' in eval_df.columns:
        print("-" * 30)
        print(f"DeepEval Hallucination:  {eval_df['deepeval_hallucination'].mean():.4f}")


    print("="*50)
    print(f"FULL REPORT SAVED TO: {output_path}")
    print("="*50)

if __name__ == "__main__":
    run_comprehensive_evaluation()