import os
import logging
from flask import Flask, request, jsonify, render_template
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import DirectoryLoader
from langchain.llms import OpenAIChat
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import yaml
import json
import re
import time
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

prompt_turbo = PromptTemplate(
    input_variables=["chat_history", "human_input"],
    template="""
    You are ChatGPT, an AI trained to provide information about ancient world mythology. Your goal is to provide users with interesting and informative stories from mythology, including fascinating facts and connections to other mythological tales.
    When a user says "Seek", generate a response that provides information related to their input. Your response should include the main answer, a list of connections to other mythological tales or characters, and a list of probable follow-up questions to encourage further engagement with the topic.
    Please generate a JSON-formatted response with the following keys:
    "Answer": a list containing your response to the user's input
    "Connections": a list of tuples where each tuple contains a character and a list of associated main characters from the same story.
    "Suggestions": a list of five probable follow-up questions for the user to engage further with the topic
    Format your response as shown below:
    {{
        "Answer": ["Your response here"],
        "Connections": [["Character1", ["Associated Main Character1", "Associated Main Character2"]], ["Character2", ["Associated Main Character1"]]],
        "Suggestions": ["Question 1?", "Question 2?", "Question 3?"]
    }}
    {chat_history}
    Human: {human_input}
    ChatGPT:
    """,
)


# Initialize the QA chain
logger.info("Initializing QA chain...")
chain = LLMChain(llm=OpenAIChat(temperature=0.5), prompt=prompt_turbo, memory=ConversationBufferMemory(memory_key="chat_history", input_key="human_input"),)

def parse_response(response):
    print("Response: {}".format(response))
    try:
        response_data = json.loads(response)

        answer = response_data.get("Answer")[0]
        connections = response_data.get("Connections")
        suggestions = response_data.get("Suggestions")

        nodes = []
        links = []
        
        # Create a node for each character
        for connection in connections:
            character = connection[0]
            associated_characters = connection[1]
            nodes.append({"id": character})
            
            # Create a link for each associated character
            for associated_character in associated_characters:
                links.append({"source": character, "target": associated_character})
        
        return answer, nodes, links, suggestions
    except Exception as e:
        logger.error(f"Error while parsing response: {e}")
        return None, None, None, None
    
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        # Get the question from the request
        question = request.json["question"]

        response = None
        answer, connections, suggestions = None, None, None

        for _ in range(3):  # Retry mechanism
            response = chain(
                {
                    "human_input": question,
                }
            )["text"]

            answer, nodes, links , suggestions = parse_response(response)
            if answer and nodes and links and suggestions:
                break
            time.sleep(5)

        if not answer:
            return jsonify({"error": "Unable to process the request."}), 500
        print("Answer: {}".format(answer))
        if nodes:
            print("Nodes:", nodes)
        if links:
            print("Links:", links)

        if suggestions:
            print("Suggestions:", suggestions)

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
        resp = jsonify({"response": answer, "suggestions": suggestions, "nodes": nodes, "links": links})
        resp.set_cookie('session_counter', str(session_counter))

        # Return the response as JSON with the session counter cookie
        return resp

    except Exception as e:
        # Log the error and return an error response
        logger.error(f"Error while processing request: {e}")
        return jsonify({"error": "Unable to process the request."}), 500

if __name__ == "__main__":
    app.run(debug=True)
