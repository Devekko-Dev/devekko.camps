https://github.com/ollama/ollama/blob/main/docs/modelfile.md

https://search.brave.com/search?q=docker+compose+create+your+own+ollama+model&source=web&summary=1&conversation=78bc476cc9fa359bd547b0

https://www.gpu-mart.com/blog/custom-llm-models-with-ollama-modelfile

https://unmesh.dev/post/ollama_custom_model/

https://stackoverflow.com/questions/78438394/how-to-create-an-ollama-model-using-docker-compose


https://chriskirby.net/run-a-free-ai-coding-assistant-locally-with-vs-code/

Prepare the Docker Compose File: Create a docker-compose.yml file that specifies the Ollama service and mounts the necessary directories for your model files. Here is an example configuration:
version: '3.8'
services:
  ollama:
    image: ollama/ollama:latest
    container_name: ollama
    ports:
      - "11435:11434"
    volumes:
      - ./model_files:/model_files
    entrypoint: ["/bin/sh", "/model_files/run_ollama.sh"]
    tty: true
    restart: unless-stopped

Prepare the Model Files: Ensure you have your Modelfile and the GGUF file (e.g., file_name.gguf) located in a directory named model_files.
Create the Run Script: Create a shell script named run_ollama.sh in the model_files directory with the following content:
#!/bin/bash
echo "Starting Ollama server..."
ollama serve &
echo "Ollama is ready, creating the model..."
ollama create mymodel -f model_files/Modelfile
ollama run mymodel

Build and Run the Docker Compose: Execute the following commands in your terminal to build and run the Docker Compose services:
docker-compose -f compose.yaml up --build

docker compose up camps_ollama
docker compose down camps_ollama -v

docker exec -it camps_ollama bash

ollama run deepseek-r1:70b
ollama run mistral-small


/etc/sysconfig/docker-storage