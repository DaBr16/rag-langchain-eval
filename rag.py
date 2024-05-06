from dotenv import load_dotenv
import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate


class Rag:
    # initialize variables
    def __init__(self, chunkingstrat : str, llm : str):
        self.chunkingstrat = chunkingstrat
        self.llm = llm

        load_dotenv()

        # set embedding function and chat model
        if llm == 'ollama':
            self.embedding_function = OllamaEmbeddings(model='mistral', base_url=os.environ.get('OLLAMA_SERVER'))
            self.chat_model = ChatOllama(model='mistral', base_url=os.environ.get('OLLAMA_SERVER'))
        elif llm == 'GPT-3.5':
            self.embedding_function = OpenAIEmbeddings(model='text-embedding-3-small', api_key=os.environ.get('OPENAI_API_KEY'))
            # default temperature = 0.7
            self.chat_model = ChatOpenAI(model_name='gpt-3.5-turbo-0125', api_key=os.environ.get('OPENAI_API_KEY'))

        # set retriever (vectorstore as retriever) 
        vectorstore_path = f'vectorstores/{chunkingstrat}'
        self.vectorstore = Chroma(persist_directory = vectorstore_path, embedding_function=self.embedding_function)
        self.retriever = self.vectorstore.as_retriever()

        # define prompt template 
        template = """Answer the question based only on the following context:
        {context}

        Question: {question}
        """
        self.prompt_template = ChatPromptTemplate.from_template(template)

    
    # function to collect contexts retieved by retriever
    def collect_context(self, question):
        context = []
        documents = self.retriever.get_relevant_documents(question)
        for doc in documents:
            context.append(doc.page_content)
        return context
    
    # function to answer question
    def answer_question(self, question):
        context = self.collect_context(question)
        prompt = self.prompt_template.format_prompt(context=context, question=question)
        answer = self.chat_model.invoke(prompt).content
        return answer


    
    

            

    