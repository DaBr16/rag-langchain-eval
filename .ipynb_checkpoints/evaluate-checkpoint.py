from datasets import load_dataset
from chain import process_question, collect_context
from ragas import evaluate
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings

# load test data 
# create dataset and a question and ground_truth
fiqa_main=load_dataset("explodinggradients/fiqa", "main", split="train[:24]")

# process question 
answer = []
contexts = []
for question in fiqa_main['question']:
    answer.append(process_question(question))
    # Not sure if the context collect seperatly is the same as the one in process_question
    contexts.append(collect_context(question))

# print("Answer", answer)
# print("Context", context)
# add answer and context to dataset
fiqa_main=fiqa_main.add_column('answer', answer)
fiqa_main=fiqa_main.add_column('contexts', contexts)

# evaluate dataset
eval = evaluate(dataset=fiqa_main, llm=Ollama(model="mistral", base_url='http://detst10avd01.testad.testo.de:8001'), embeddings=OllamaEmbeddings(model="mistral", base_url='http://detst10avd01.testad.testo.de:8001'))
print(eval)


