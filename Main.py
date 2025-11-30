#import dependencies
import os
import shutil
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.evaluation import load_evaluator, EvaluatorType
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.llms import HuggingFacePipeline

import os
from dotenv import load_dotenv

# Set LangSmith variables directly in code
os.environ["LANGCHAIN_API_KEY"] = "your actual API key"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "LangSmith Project Name"
load_dotenv()

# Step 1: Prepare your documents
#for Markdown files
def load_document():
   #glob module in Python is used to search for files and directories
   loader = DirectoryLoader("Data/",glob="*.txt")
   documents=loader.load()
   print("Number of Documents " ,len(documents))
   return documents


#Step 2 : Text Splitting
def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
    chunks=text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    chunk_example = chunks[3]
    print(chunk_example.page_content)
    print(chunk_example.metadata)
    return chunks

def save_to_faiss():
    #Step 3 : Load the embedding model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print("Embedding Model Name : sentence-transformers/all-MiniLM-L6-v2")
    #if we compare two vectors of similar words using Evaluator it will more close to 0
    apple_vector=embeddings.aembed_query("APPLE")
    print(f"As example embedding for APPLE is {apple_vector}" )
    # Step 4: Create the FAISS vector store and save it in Disk
    #or use chromaDB
    db=FAISS.from_documents(split_text(load_document()),embeddings)
    #clear folder if it already exists
    if os.path.exists("faiss_index/"):
        shutil.rmtree("faiss_index/")
    # Save to disk and remember to use same embedding model when you upload it
    db.save_local("faiss_index")
    #FAISS index in :index.faiss
    #metadata store in : index.pkl
    print("saved FAISS database in faiss_index")
    return db

def main():
   embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
   #First to save DB locally
   # #db=save_to_faiss()
   # Later to load FAISS
   db = FAISS.load_local("faiss_index/", embeddings, allow_dangerous_deserialization=True)
   #Step 5: Create retriever
   retriever = db.as_retriever()

   #LangChain
   #google-t5/t5-small German
   #google/gemma-2-2b-it ِArabic
   #mistralai/Mistral-7B-v0.1 /Good
   #BAAI/Infinity-Instruct
   #Step 6: Load LLM
   #when choosing LLM be aware of its number of parameters and required RAM
   model_id = "google/flan-t5-base"
   tokenizer = AutoTokenizer.from_pretrained(model_id)
   model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
   pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
   llm = HuggingFacePipeline(pipeline=pipe)

   #Step 7: Enter the query
   query = input('Enter your question: ')
   #print(f'The question is {query}')

   #Step 8: Load QA chain
   qa_chain = load_qa_chain(llm, chain_type="stuff")

   #Step 9: Ask a question
   relevant_docs = retriever.get_relevant_documents(query)
   #print(relevant_docs.page_content)
   #print(relevant_docs.metadata)
   response = qa_chain.run(input_documents=relevant_docs, question=query)

   #Print the Answer
   PROMPT_TEMPLATE = """
   The answer of the question : {question}
   is  {answer} 
   it got the content from page number {page_number}
   """

   #context_text ="\n\n ---- \n\n".join([doc.page_content for doc,_score in relevant_docs])
   prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
   prompt = prompt_template.format(question=query,answer=response,page_number=relevant_docs.page_content)
   print(prompt)

   #print(f'The answer is {response}')
   #sources = [doc.metdata.get("source", None) for doc, _score in relevant_docs]


if __name__ == '__main__':
    main()
