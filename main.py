import os
import textract
import pandas as pd
import matplotlib.pyplot as plt
import os.path
from os import path
from transformers import GPT2TokenizerFast
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from IPython.display import display
import ipywidgets as widgets
os.environ["OPENAI_API_KEY"] = "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"

loader = PyPDFLoader("./sample_data/faq.pdf")
pages = loader.load_and_split()

print(path.exists("./sample_data/faq.pdf"))

doc = textract.process('./faq.docx')
with open('faq.txt', 'w', encoding='UTF-8') as f:
    f.write(doc.decode('utf-8'))

with open('faq.txt', 'r', encoding='UTF-8') as f:
    text = f.read()

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

text_splitter = RecursiveCharacterTextSplitter(

    chunk_size = 512,
    chunk_overlap  = 24,
    length_function = count_tokens,
)

chunks = text_splitter.create_documents([text])

type(chunks[0]) 

token_counts = [count_tokens(chunk.page_content) for chunk in chunks]

df = pd.DataFrame({'Token Count': token_counts})

df.hist(bins=40, )



embeddings = OpenAIEmbeddings()

db = FAISS.from_documents(chunks, embeddings)


query = "Who created transformers?"
docs = db.similarity_search(query)

qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.1), db.as_retriever())

chat_history = []

def on_submit():
    query =input("Please enter your query ")
    
    
    if query.lower() == 'exit':
        print("Thank you for using the State of the Union chatbot!")
        return
    
    result = qa({"question": query, "chat_history": chat_history})
    chat_history.append((query, result['answer']))
    
    print (result["answer"])

print("Welcome to Swinburne Online chatbot! Type 'exit' to stop.")

on_submit()
