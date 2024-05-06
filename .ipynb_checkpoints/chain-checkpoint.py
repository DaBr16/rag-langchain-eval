from langchain_community.document_loaders import HuggingFaceDatasetLoader
from datasets import load_dataset
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain.output_parsers import StructuredOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser

# set datapath 
data_path= "data"

# load data
fiqa_corpus=load_dataset("explodinggradients/fiqa", "corpus")
# document_loader = HuggingFaceDatasetLoader(path="explodinggradients/fiqa", page_content_column="doc")
# documents = document_loader.load()

# set embedding_function
embedding_function=OllamaEmbeddings(model="mistral", base_url='http://detst10avd01.testad.testo.de:8001')

# create vectorstore
vectorstore = Chroma(persist_directory="data", embedding_function=embedding_function)

# corpus_len = len(fiqa_corpus['corpus'])

# for index, doc in enumerate(fiqa_corpus["corpus"]):
#     doc = Document(page_content=doc["doc"], metadata={"source": "local", "id": index})
#     vectorstore.add_documents([doc])

# set retriever
retriever = vectorstore.as_retriever()

# get context
def collect_context(query):
    context = []
    documents = retriever.get_relevant_documents(query)
    for doc in documents:
        context.append(doc.page_content)
    return context


# define prompt
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt_template = ChatPromptTemplate.from_template(template)

# define model
model=Ollama(model="mistral", base_url='http://detst10avd01.testad.testo.de:8001')

# define output parser

# execute the prompt and model inference
def answer_question(context, question):
    prompt = prompt_template.format_prompt(context=context, question=question)
    raw_response = model.invoke(prompt)
#   parsed_response = output_parser.parse(text=raw_response)
    return raw_response


# process the question input
def process_question(question):
    # Retrieve relevant context based on the question
    context = collect_context(question)
    
    # Generate an answer using the model
    answer = answer_question(context, question)
    
    return answer

result = process_question("What is considered a business expense on a business trip?")
print(result)







