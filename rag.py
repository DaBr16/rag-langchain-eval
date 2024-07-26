from dotenv import load_dotenv
import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate


class Rag:
    """
    Rag class represents a Retrieval-Augmented Generation (RAG) system.

    This class initializes a RAG system with specified chunking strategy, language model, and embedding function.
    It provides methods to collect context and answer questions based on the context.

    Attributes:
        chunkingstrategy (str): Strategy for chunking documents.
        llm (str): The language model used (either 'ollama' or OpenAI model).
        embedding (str): The embedding function used.
        embedding_function (Embeddings): The embedding function instance.
        chat_model (ChatModel): The chat model instance.
        prompt_template (ChatPromptTemplate): The template for generating prompts.
        vectorstore (Chroma): The vectorstore for document retrieval.
        retriever (Retriever): The retriever for fetching relevant documents.
    """

    def __init__(self, chunkingstrategy: str, llm: str, embedding: str):
        """
        Initialize the Rag object.

        Args:
            chunkingstrategy (str): Name of the vectorstore (see "\vectorstores").
            The vectorstore is named as the chunking strategy.
            llm (str): The language model to use ('ollama' or an OpenAI model).
            embedding (str): The embedding function to use.
        """
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
        """
        Collect relevant context for a given question.

        Args:
            question (str): The question to collect context for.

        Returns:
            list: A list of context strings relevant to the question.
        """
        context = []
        documents = self.retriever.get_relevant_documents(question)
        for doc in documents:
            context.append(doc.page_content)
        return context

    def answer_question(self, question: str):
        """
        Answer a question based on the collected context.

        Args:
            question (str): The question to be answered.

        Returns:
            str: The generated answer based on the context.
        """
        context = self.collect_context(question)
        prompt = self.prompt_template.format_prompt(context=context, question=question)
        answer = self.chat_model.invoke(prompt).content
        return answer
