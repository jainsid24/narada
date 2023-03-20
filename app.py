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
import re

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
    input_variables=["chat_history", "human_input"],
    template="""
    You are ChatGPT, an AI trained to provide information about ancient world mythology. When a user says "Seek", share an interesting story from mythology, including fascinating facts and connections to other mythological tales. For follow-up questions, provide additional intriguing details from the story.

    Please generate a response to the following question, a list of tuples containing connections, and 5 probable follow-up questions. Ensure that the response is structured with the main answer, a connections section as a list of tuples, and a suggestions section, formatted like this:

    Answer: Your response here.
    Connections: [("Character1", ["Associated Main Character1", "Associated Main Character2"]), ("Character2", ["Associated Main Character1"])]
    Suggestions: ["Question 1?", "Question 2?", "Question 3?"]

    {chat_history}
    Human: {human_input}
    ChatGPT:
    """,
)

# Initialize the QA chain
logger.info("Initializing QA chain...")
chain = LLMChain(llm=OpenAIChat(), prompt=prompt_turbo, memory=ConversationBufferMemory(memory_key="chat_history", input_key="human_input"),)

def parse_response(response):
    answer_pattern = r"Answer: (.*?)\nConnections:"
    connections_pattern = r"Connections: (\[.*?\])\nSuggestions:"
    suggestions_pattern = r"Suggestions: (\[.*?\])"

    answer_match = re.search(answer_pattern, response, re.DOTALL)
    connections_match = re.search(connections_pattern, response, re.DOTALL)
    suggestions_match = re.search(suggestions_pattern, response, re.DOTALL)

    if not (answer_match and connections_match and suggestions_match):
        return None, None, None

    answer = answer_match.group(1)
    connections_str = connections_match.group(1)
    suggestions_str = suggestions_match.group(1)

    try:
        connections = eval(connections_str)
        suggestions = eval(suggestions_str)
    except:
        return None, None, None

    return answer, connections, suggestions


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
