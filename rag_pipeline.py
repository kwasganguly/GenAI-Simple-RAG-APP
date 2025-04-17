# rag_pipeline.py

from langchain.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import os

class RAGPipeline:
    def __init__(self, openai_api_key, model_name="gpt-3.5-turbo", embedding_model="text-embedding-ada-002"):
        os.environ["OPENAI_API_KEY"] = openai_api_key
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.vectorstore = None
        self.model_name = model_name

    def load_documents(self, file_path=None, url=None):
        if file_path:
            loader = PyPDFLoader(file_path)
        elif url:
            loader = WebBaseLoader(url)
        else:
            raise ValueError("Provide file_path or url.")
        return loader.load()

    def build_vectorstore(self, documents):
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)

    def query(self, question, top_n=3):
        if not self.vectorstore:
            raise ValueError("Vectorstore not initialized.")
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": top_n})
        qa = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(temperature=0, model_name=self.model_name),
            retriever=retriever,
            return_source_documents=True
        )
        return qa({"query": question})

