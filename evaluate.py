from datasets import load_dataset, Dataset
from rag import Rag
import pandas as pd
import numpy as np
import time
from ragas import evaluate
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
import os
from dotenv import load_dotenv
from datetime import datetime

from ragas.metrics import (
    answer_relevancy,
    answer_correctness,
    context_recall,
    context_relevancy
)

load_dotenv()
os.environ.get('OPENAI_API_KEY')

# initialize rag object
openAI_rag = Rag(chunkingstrat='fixed_size_2000', llm='GPT-3.5')

# load test data 
# create dataset and a question and ground_truth
# NOTE: are the question picked randomly?
fiqa_main=load_dataset("explodinggradients/fiqa", "main", split="train[:18]")

# create pandas dataframe
eval_dataset = fiqa_main.to_pandas()
eval_dataset['answer'] = np.nan
eval_dataset['contexts'] = np.nan

# set added columns to dtype cause of FutureWarning from pandas
eval_dataset['answer'] = eval_dataset['answer'].astype('object')
eval_dataset['contexts'] = eval_dataset['contexts'].astype('object')

# fill evaluation dataset
for index, row in eval_dataset.iterrows():
    question = row['question']
    eval_dataset.at[index, 'answer'] = openAI_rag.answer_question(question)
    eval_dataset.at[index, 'contexts'] = openAI_rag.collect_context(question)

now = datetime.now()
file_name = now.strftime("%H:%M %d.%m.%Y")

eval_dataset.to_parquet(f'eval_datasets/{file_name}.parquet')

dataset = Dataset.from_pandas(eval_dataset)

print("Stopping the process.")
time.sleep(60)  
print("One minute has passed, and the process resumes.")

metrics = [answer_relevancy, answer_correctness, context_recall, context_relevancy]
# evaluate dataset
# default temperature = 0.7
eval = evaluate(dataset=dataset, llm=ChatOpenAI(model_name='gpt-3.5-turbo-0125', temperature=0.0), embeddings=OpenAIEmbeddings(model='text-embedding-3-small'), metrics=metrics, raise_exceptions=False)
print(eval)


