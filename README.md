# Narada 

![Python version](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9-blue?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)
![Commit Activity](https://img.shields.io/github/last-commit/jainsid24/neural-network-simulation?style=flat-square)
![Repo Size](https://img.shields.io/github/repo-size/jainsid24/neural-network-simulation?style=flat-square)
![OpenAI API key](https://img.shields.io/badge/OpenAI%20API%20key-required-red?style=flat-square)
![Docker](https://img.shields.io/badge/docker-available-blue?style=flat-square)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black?style=flat-square)

TBD

## Getting Started
To use this utility:
1. Clone the repository
```
git clone https://github.com/jainsid24/narada
```
2. Build the Docker image by running the following command in the terminal:
```
docker build -t narada:latest .
```
3. Once the image is built, run the Docker container using the following command:
```
docker run -p 5001:5001 narada
```
4. Use curl/postman for API call
```
curl --header "Content-Type: application/json" \
     --request POST \
     --data '{"question": "Narayan Narayan?"}' \
     http://<pods-ip-address>:5001/api/chat
```

## Configuration

Before you can use the utility, you need to set up the configuration file. The configuration file is a YAML file that contains the following options:

* openai_api_key: Your OpenAI API key.

## Usage

To start the chatbot, run:

```
python app.py
```

This will start the chatbot on port 5000.

To use the chatbot, send a POST request to http://localhost:5000/api/chat with a JSON payload containing the question to ask, like this:

```
curl -X POST \
  http://localhost:5001/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"question": "What is the capital of France?"}'
```

This will return a JSON response containing the chatbot's answer to the question:

```
{"response": "The capital of France is Paris."}
```

## Contributing

If you find a bug or have an idea for a new feature, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
