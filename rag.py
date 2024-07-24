from dotenv import load_dotenv
import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate


class Rag:
    def __init__(self, chunkingstrategy: str, llm: str, embedding: str):
        self.chunkingstrategy = chunkingstrategy
        self.llm = llm
        self.embedding = embedding

        load_dotenv()

        # Set embedding function and chat model
        if llm == "ollama":
            self.embedding_function = OllamaEmbeddings(
                model="mistral", base_url=os.environ.get("OLLAMA_SERVER")
            )
            self.chat_model = ChatOllama(
                model="mistral", base_url=os.environ.get("OLLAMA_SERVER")
            )
        else:
            self.embedding_function = OpenAIEmbeddings(
                model=self.embedding, api_key=os.environ.get("OPENAI_API_KEY")
            )
            self.chat_model = ChatOpenAI(
                model=self.llm, api_key=os.environ.get("OPENAI_API_KEY")
            )

        # Define prompt template
        template = """Answer the question based only on the following context:
        {context}

        Question: {question}
        """
        self.prompt_template = ChatPromptTemplate.from_template(template)

        # Set retriever (vectorstore as retriever)
        vectorstore_path = f"vectorstores/{chunkingstrategy}"
        self.vectorstore = Chroma(
            persist_directory=vectorstore_path,
            embedding_function=self.embedding_function,
        )
        self.retriever = self.vectorstore.as_retriever()

    def collect_context(self, question: str):
        context = []
        documents = self.retriever.get_relevant_documents(question)
        for doc in documents:
            context.append(doc.page_content)
        return context

    def answer_question(self, question: str):
        context = self.collect_context(question)
        prompt = self.prompt_template.format_prompt(context=context, question=question)
        answer = self.chat_model.invoke(prompt).content
        return answer
