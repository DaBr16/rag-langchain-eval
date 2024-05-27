from datasets import load_dataset
from ragas import evaluate
import os
from dotenv import load_dotenv 
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from ragas.metrics import (
    answer_relevancy,
    answer_correctness,
    context_recall,
    context_relevancy
)

load_dotenv()

os.environ.get('OPENAI_API_KEY')

eval_dataset = load_dataset("explodinggradients/fiqa", "ragas_eval", split="baseline[:]")




result = evaluate(dataset = eval_dataset, llm=ChatOpenAI(model_name='gpt-4o', temperature=0.0), embeddings=OpenAIEmbeddings(model='text-embedding-3-small'), metrics=[answer_relevancy, answer_correctness, context_recall, context_relevancy], raise_exceptions=False)

print(result)


