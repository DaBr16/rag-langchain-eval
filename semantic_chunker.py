from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import os
import gc

openai_api_key = os.getenv("OPENAI_API_KEY")

with open("fiqa_dataset/corpus.txt") as f:
    corpus = f.read()

# Initialize the vector store
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma(
    embedding_function=embedding_model, persist_directory="vectorstores/percentile"
)

# Initialize the text splitter
text_splitter = SemanticChunker(embedding_model, breakpoint_threshold_type="percentile")

# Define the chunk size
cut = round(len(corpus) / 10000)

# Process the corpus in smaller chunks
i = 0
while i < len(corpus):
    chunk = corpus[i:i + cut]
    chunks = text_splitter.create_documents(chunk)
    db.add_documents(documents=chunks)

    # Clear the chunk and chunks variables to free up memory
    del chunk
    del chunks
    gc.collect()  # Force garbage collection to free memory

    i += cut

# Persist the database
db.persist()
