# UVA Engineering RAG Chatbot by @omairionz - Inspired by pixegami

## Install dependencies

1. Do the following before installing the dependencies found in `requirements.txt` file because of current challenges installing `onnxruntime` through `pip install onnxruntime`. 

     - For Windows users, follow the guide [here](https://github.com/bycloudai/InstallVSBuildToolsWindows?tab=readme-ov-file) to install the Microsoft C++ Build Tools. Be sure to follow through to the last step to set the enviroment variable path.

2. Now run this command to install dependenies in the `requirements.txt` file. 

```python
pip install -r requirements.txt
```

3. Install markdown depenendies with: 

```python
pip install "unstructured[md]"
```

# How to run this project

## 1. Create database

Create the Chroma DB.

```python
python create_database.py
```

## 2. Query the database

Query the Chroma DB.

```python
python query_data.py "What is Computer Science?"
```

# Secondary Method to Run Project via StreamLit

## 1. Head over to app.py

Activate StreamLit by running the following in terminal:

```python
streamlit run app.py
```

Make sure you've already ran `create_database.py` before doing this in order to create proper context embeddings.

> You'll also need to set up an OpenAI account (and set the OpenAI key in your environment variable) for this to work. You can do that [here](https://platform.openai.com/api-keys).


Here is a step-by-step tutorial video I used to create this project: [RAG+Langchain Python Project: Easy AI/Chat For Your Docs](https://www.youtube.com/watch?v=tcqEUSNCn8I&ab_channel=pixegami).
