import pandas as pd
from deepeval.metrics import HallucinationMetric
from deepeval.test_case import LLMTestCase
from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_openai import ChatOpenAI
import os

# Use your existing wrapper to see what the Judge is thinking
class DebugJudgeWrapper(DeepEvalBaseLLM):
    def __init__(self, model):
        self.model = model
    def load_model(self):
        return self.model
    def generate(self, prompt: str) -> str:
        return self.load_model().invoke(prompt).content
    async def a_generate(self, prompt: str) -> str:
        res = await self.load_model().ainvoke(prompt)
        return res.content
    def get_model_name(self):
        return "Debug Claude Judge"

def debug_hallucination():
    # Setup the judge again
    langchain_llm = ChatOpenAI(
        model_name="anthropic/claude-3.5-sonnet", # Use your MODEL_NAME env
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1"
    )
    deepeval_model = DebugJudgeWrapper(langchain_llm)
    
    # Load your results
    df = pd.read_excel("data/evaluation_dashboard.xlsx")
    metric = HallucinationMetric(threshold=0.5, model=deepeval_model)

    for i, row in df.iterrows():
        print(f"\n--- Debugging Row {i+1} ---")
        test_case = LLMTestCase(
            input="N/A",
            actual_output=str(row['prediction']),
            context=[str(row['context'])] # Ensure it's a list of strings
        )
        
        metric.measure(test_case)
        print(f"Score: {metric.score}")
        print(f"Reasoning: {metric.reason}") # This will tell you WHY it's 0

if __name__ == "__main__":
    debug_hallucination()