from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from transformers import AutoTokenizer
from langchain.docstore.document import Document
from langchain_mistralai import MistralAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_mistralai import ChatMistralAI
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.vectorstores  import FAISS
from langchain.chains import RetrievalQA
from src.prompt import *
import time
import os
import pickle
from dotenv import load_dotenv

load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
os.environ["MISTRAL_API_KEY"] = MISTRAL_API_KEY

def file_processing(file_path):
    file_path = "/workspaces/codespaces-blank/data/SDG.pdf"
    loader=PyPDFLoader(file_path)
    data=loader.load()
    question_gen = ""
    for page in data:
        question_gen += page.page_content
#    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3",use_auth_token=HUGGINFACE_AUTH_TOKEN)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2500,   # Approximate token length
        chunk_overlap=200,  # Overlapping context to preserve meaning
    )
    chunks = splitter.split_text(question_gen)
    document_ques_gen = [Document(page_content=t) for t in chunks]
    document_splits = splitter.split_documents(document_ques_gen)

    return document_splits

def llm_pipeline(file_path):
    document_splits=file_processing(file_path)
    llm = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0.3
    )

    PROMPT_QUESTIONS = PromptTemplate(template=prompt_template, input_variables=["text"])

    REFINE_PROMPT_QUESTIONS = PromptTemplate(
        input_variables=["existing_answer","text"],
        template=refine_template
    )

    ques_gen_chain = load_summarize_chain(llm=llm,
                                      chain_type="refine",
                                      verbose=True,
                                      question_prompt=PROMPT_QUESTIONS,
                                      refine_prompt=REFINE_PROMPT_QUESTIONS)
    ques = ques_gen_chain.run(document_splits)
    # cache_folder="./vectorstore_cache"
    # cache_path = os.path.join(cache_folder, "vectorstore.pkl")    
    # if os.path.exists(cache_path):
    #     print(f"Loading vectorstore from cache: {cache_path}")
    #     with open(cache_path, 'rb') as f:
    #         vectorstore = pickle.load(f)
    try:
        vectorstore = FAISS.load_local("vectorstore", MistralAIEmbeddings(
        model="mistral-embed",
        ), allow_dangerous_deserialization=True)
    except:
        
        embeddings = MistralAIEmbeddings(
        model="mistral-embed",
        )
        vectorstore = FAISS.from_documents(documents=document_splits, embedding=embeddings)
        
        # print("System prompt and documents added successfully!")
        # os.makedirs(cache_folder, exist_ok=True)
        # with open(cache_path, 'wb') as f:
        #     pickle.dump(vectorstore, f)
        # print(f"Vectorstore saved to cache: {cache_path}")
        vectorstore.save_local("vectorstore")

    llm_answer_gen = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0.1
    )
    ques_list = ques.split("\n")
    answer_generation_chain = RetrievalQA.from_chain_type(
    llm=llm_answer_gen,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
    )
    return answer_generation_chain, ques_list