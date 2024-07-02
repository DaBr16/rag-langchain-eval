from rag import Rag
from ragas.metrics import (
    answer_relevancy,
    answer_correctness,
    context_recall,
    context_relevancy
)
from datasets import load_dataset, Dataset
import numpy as np
import pandas as pd
from ragas import evaluate
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv 

class RagEvaluator:
    def __init__(self, rag: Rag, num_of_questions=18, num_of_runs=5, 
                 metrics=[answer_relevancy, answer_correctness, context_recall, context_relevancy]):
        self.rag = rag
        self.num_of_questions = num_of_questions
        self.num_of_runs = num_of_runs
        self.metrics = metrics
        self.golden_dataset_list = []
        self.eval_results = self.__evaluate()
        
    def __evaluate(self):
        load_dotenv()
        os.environ.get('OPENAI_API_KEY')
        eval_results = []
        num = 1
        while num <= self.num_of_runs:
            dataset = self.create_golden_dataset()
            result = evaluate(dataset = dataset, llm=ChatOpenAI(model_name='gpt-3.5-turbo-0125', temperature=0.0), 
                              embeddings=OpenAIEmbeddings(model='text-embedding-3-small'), metrics=self.metrics, raise_exceptions=False)
            eval_results.append(dict(result))
            num +=1

        return eval_results

    def create_golden_dataset(self):
        # Load FIQA dataset
        fiqa_main = load_dataset("explodinggradients/fiqa", "main", split=f"train[:{self.num_of_questions}]")
        eval_dataset = fiqa_main.to_pandas()

        # Initialize columns
        eval_dataset['answer'] = np.nan
        eval_dataset['contexts'] = np.nan

        # Set added columns to appropriate dtypes to avoid FutureWarnings
        eval_dataset['answer'] = eval_dataset['answer'].astype('object')
        eval_dataset['contexts'] = eval_dataset['contexts'].astype('object')

        # Fill the golden dataset
        for index, row in eval_dataset.iterrows():
            question = row['question']
            eval_dataset.at[index, 'answer'] = self.rag.answer_question(question)
            eval_dataset.at[index, 'contexts'] = self.rag.collect_context(question)

        # store golden dataset in list
        self.golden_dataset_list.append(eval_dataset)

        # Convert pandas dataframe to Dataset type
        golden_dataset = Dataset.from_pandas(eval_dataset)

        return golden_dataset

    # get mean result
    def get_mean_result(self):
        # create 4 list to store metrics
        answer_relevancies = []
        answer_correctness = []
        context_recalls = []
        context_relevancies = []
        # start looping through list of dicts
        for dic in self.eval_results:
        # enumerate through dict and add metrice to corresponding list
            for i, v in dic.items():
                match i:
                    case "answer_relevancy":
                        answer_relevancies.append(v)
                    case "answer_correctness":
                        answer_correctness.append(v)
                    case "context_recall":
                        context_recalls.append(v)
                    case "context_relevancy":
                        context_relevancies.append(v)

        # create dict with the mean of results
        mean_answer_relevancy = round(np.mean(answer_relevancies), 4)
        mean_answer_correctness = round(np.mean(answer_correctness), 4)
        mean_context_recall = round(np.mean(context_recalls), 4)
        mean_context_relevancy = round(np.mean(context_relevancies), 4)
        means = {'answer_relevancy': mean_answer_relevancy, 'answer_correctness': mean_answer_correctness, 
                 'context_recall': mean_context_recall, 'context_relevancy': mean_context_relevancy}

        return means

    # get specific result
    def get_result(self, run_number: int):
        num = run_number - 1
        return self.eval_results[num]
    
    # get golden dataset
    def get_golden_dataset(self, run_number: int):
        num = run_number - 1
        return self.golden_dataset_list[num]
    

    




