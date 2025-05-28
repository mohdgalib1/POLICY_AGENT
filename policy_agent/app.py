import os
import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint



def load_pdfs(pdf_dir="pdfs"):
    docs = []
    for file in os.listdir(pdf_dir):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(pdf_dir, file))
            docs.extend(loader.load())
    return docs

def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)


def create_vectorstore(docs):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(docs, embedding=embeddings, persist_directory="./chroma_db")
    return vectorstore

def build_qa_chain():
    docs = load_pdfs()
    split = split_docs(docs)
    vectorstore = create_vectorstore(split)
    retriever = vectorstore.as_retriever()
    

    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not hf_token:
        raise ValueError("Missing HUGGINGFACEHUB_API_TOKEN environment variable.")

    llm = HuggingFaceEndpoint(
        repo_id="google/flan-t5-base",
        huggingfacehub_api_token=hf_token,
        model_kwargs={"temperature": 0, "max_length": 512}
    )
    
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

qa_chain = build_qa_chain()


def ask_question(query):
    return qa_chain.run(query)

interface = gr.Interface(fn=ask_question, inputs="text", outputs="text", title="📄 PDF Chatbot Agent")

if __name__ == "__main__":
    interface.launch()
