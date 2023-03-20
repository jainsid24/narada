import os
import logging
from flask import Flask, request, jsonify, render_template
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import DirectoryLoader
from langchain.llms import OpenAIChat
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import WebBaseLoader
import yaml

from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import LLMChain

import nltk

nltk.download("punkt")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration from YAML file
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

os.environ["OPENAI_API_KEY"] = config["openai_api_key"]

template_dir = os.path.abspath("templates")
app = Flask(__name__, template_folder=template_dir, static_folder="static")

prompt_turbo=PromptTemplate(
    input_variables=["chat_history", "human_input", "tone", "persona"],
    template="""You are a chatbot who acts like {persona}, having a conversation with a human.
    On each turn, when a user says "narayan narayan", you give an interesting story from ancient mythology and draw its parallelism to the events happening around the world when the story was written in a bulleted list.
    When asked a followup question, give some more interesting facts from the story.
    Given the following question, Create a final answer in the tone {tone}. 
    {chat_history}
    Human: {human_input}
    Chatbot:""",
)

tone = config.get("tone", "default")
persona = config.get("persona", "default")

# Initialize the QA chain
logger.info("Initializing QA chain...")
chain = LLMChain(llm=OpenAIChat(), prompt=prompt_turbo, memory=ConversationBufferMemory(memory_key="chat_history", input_key="human_input"),)



@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        # Get the question from the request
        question = request.json["question"]

        # Get the bot's response
        question =  request.json["question"]
        response = chain(
            { 
                "human_input": question,
                "tone": tone,
                "persona": persona,  
            }
        )["text"]
        
        # Increment message counter
        session_counter = request.cookies.get('session_counter')
        if session_counter is None:
            session_counter = 0
        else:
            session_counter = int(session_counter) + 1

        # Check if it's time to flush memory
        if session_counter % 10 == 0:
            chain.memory.clear()

        # Set the session counter cookie
        resp = jsonify({"response": response})
        resp.set_cookie('session_counter', str(session_counter))

        # Return the response as JSON with the session counter cookie
        return resp

    except Exception as e:
        # Log the error and return an error response
        logger.error(f"Error while processing request: {e}")
        return jsonify({"error": "Unable to process the request."}), 500

if __name__ == "__main__":
    app.run(debug=True)
