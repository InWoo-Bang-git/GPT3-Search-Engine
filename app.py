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
from flask import Flask, render_template, request
import openai
import os
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.environ["OPENAI_API_KEY"]

os.environ["OPENAI_API_KEY"] = "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"
loader = PyPDFLoader("./sample_data/faq.pdf")
pages = loader.load_and_split()

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

qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.1), db.as_retriever())

chat_history = []

def on_submit(input):
    query =input
    
    if query.lower() == 'exit':
        print("Thank you for using the State of the Union chatbot!")
        return
    
 # check if the query is in the PDF
    found_in_pdf = False
    for chunk in chunks:
        if query in chunk.page_content:
            found_in_pdf = True
            break

    if not found_in_pdf:
        return "I don't know"

    result = qa({"question": query, "chat_history": chat_history})
    chat_history.append((query, result['answer']))
    
    return (result["answer"])

#print("Welcome to Swinburne Online chatbot! Type 'exit' to stop.")

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chatbot.html", methods=["POST"])
def chatbot():
    #pass
    
    user_input = request.form["message"]
    #user_input = request.form["msg"]
    
    prompt = f"User: {user_input}\nChatbot: "
    chat_history = []
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        temperature=0.5,
        max_tokens=60,
        top_p=1,
        frequency_penalty=0,
        stop=["\nUser: ", "\nChatbot: "]
    )

    bot_response = on_submit(user_input)

    chat_history.append(f"User: {user_input}\nChatbot: {bot_response}")

    return render_template(
        "/chatbot.html",
        user_input=user_input,
        bot_response=bot_response,
    )

if __name__ == "__main__":
    app.run(debug=True)