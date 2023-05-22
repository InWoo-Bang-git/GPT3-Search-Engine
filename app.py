import os
from flask import Flask, render_template, request
import openai
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
from transformers import GPT2TokenizerFast

# Load environment variables
load_dotenv()

# Set OpenAI API Key
openai.api_key = os.getenv("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")

# Load and process FAQ document
loader = PyPDFLoader("./sample_data/faq.pdf")

pages = loader.load_and_split()

# Initialize GPT2 tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# Function to count tokens in text
def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 512,
    chunk_overlap  = 24,
    length_function = count_tokens,
)

chunks = text_splitter.create_documents([page.page_content for page in pages])

# Initialize OpenAI Embeddings
embeddings = OpenAIEmbeddings()

# Initialize FAISS vector store from chunks and embeddings
db = FAISS.from_documents(chunks, embeddings)

# Create question answering chain
qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.1), db.as_retriever())

# Initialize chat history
chat_history = []

# Function to handle submitted queries
def on_submit(input):
    query = input
    if query.lower() == 'exit':
        print("Thank you for using Swinburne Online chatbot!")
        return

    result = qa({"question": query, "chat_history": chat_history})
    chat_history.append((query, result['answer']))
    
    return (result["answer"])


# Flask application
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chatbot.html", methods=["POST"])
def chatbot():
    user_input = request.form["message"]
    bot_response = on_submit(user_input)
    return render_template("/chatbot.html", user_input=user_input, bot_response=bot_response)

if __name__ == "__main__":
    app.run(debug=True)
