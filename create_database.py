# Import for data to "document" conversion
from langchain_community.document_loaders import DirectoryLoader # STAGE 1

#Import document class
from langchain_core.documents import Document # Stage 1/2

# Import for recursive text splitting (to split documents into chunks)
from langchain_text_splitters import RecursiveCharacterTextSplitter # STAGE 2

# Import for CHROMA to turn chunks into vector-embeddings database
from langchain_chroma import Chroma # STAGE 3

# Import OpenAIEmbeddings model for chunks --> embeddings conversion
from langchain_openai import OpenAIEmbeddings # Stage 3

# Import os and shutil to check database existance
import os # Stage 3
import shutil # Stage 3

# import openai
import openai

# Imports everything from .env into this file
from dotenv import load_dotenv

# ------------------------------------------------------------------------ #

# Loads everything from .env file into conputer memory
load_dotenv()
# Goes to .env file and captures value under variable name "OPENAI_API_KEY"
# Enviorment variables live on Operating System and is captured using os.
openai.api_key = os.environ['OPENAI_API_KEY']

# defines FOLDER, not file, of data location
DATA_PATH = "data"

# To be able to query each chunk, we need to turn this into a database
# Chroma is a special kind of database that uses vector embeddings 
# This variable instantiation is used later to create a database folder we can access
CHROMA_PATH = "chroma-embeddings"

# runs defined methods to create database
def main():
    generate_data_store()

def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)

# ******STAGE 1******

# This piece of code uses langchain's DirectoryLoader to take data and turn them into "documents".
# Each file in DATA_PATH is converted to documents
# Data is first loaded up into "documents" before split into many chunks of text later on (also contains meta data)
# Only works for markdown files
def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    documents = loader.load()
    return documents

# ******STAGE 2******

# This is a recursive text splitter. It takes the data from each document and splits it into smaller chunks
# Takes in a LIST of documents for splitting. Good if you have multiple data files
def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter( # Recursive used to intelligently split text. No sentence/word breaks.
        chunk_size = 1000, # How long each chunk is in characters
        chunk_overlap = 500, # How much overlap between each chunk
        length_function = len, # Measures chunk size using Python's len() function (basically confirms correct char count)
        add_start_index = True, # Each chunk stores meta data from origin
    )

    # Takes list of document objects and splits them into smaller document chunks & perserves meta data 
    # Each chunk is now ready to be embedded
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.") # displays how many documents were used to create how many chunks

    return chunks

# ******STAGE 3******

# This piece of code is VERY important. It turns each chunk into embeddings (numerica vectors that represent MEANING)
# Then, embeddings + meta data + origional text are stored in Chroma database
# When chatbot is prompted: 
# 1. Question is converted into embedding
# 2. Chroma finds stored vectors from chunks that are mathematically closest to question vectors
# 3. Closeest chunks are returned to generate an answer
# NOTE:
# 1. OpenAI handles chunk --> embeddings conversion
# 2. Chroma stores and searches the embeddings

def save_to_chroma(chunks: list[Document]):

    # Clears out database if it already exists. Useful if you need to clear out database before running script to create a new one
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    db = Chroma.from_documents(
        chunks, # each chunks contains text & meta data. Chroma uses embeddings model to embed each chunk, stores its vector numbers, & keeps metadata
        OpenAIEmbeddings(), # Tells Chroma to USE OPENAI to convert chunks into embeddings
        persist_directory=CHROMA_PATH # allows chroma saveto save everything to specified disk (folder) instead of memory so we don't have to reembed everything
    )   # Saves everything in chroma memory to disk (a folder)
    
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.") # tells you how many chunks were saved to chroma-path


if __name__ == "__main__":
    main()



