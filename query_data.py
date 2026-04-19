# used to take input from user
import argparse

# Chroma import to store and search embeddings
from langchain_chroma import Chroma

# Used to convert input into enbeddings
from langchain_openai import OpenAIEmbeddings

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv

# for a more polished look
from rich.console import Console
from rich.panel import Panel

# delay for more realistic chatbot experience
from time import sleep

from datetime import datetime

load_dotenv()

# chroma path declaration for query embeddings
CHROMA_PATH = "chroma-embeddings"

# Prompt template
PROMPT_TEMPLATE = """
You are a helpful UVA Engineering Academic Advisor. 
Answer clearly and concisely in plain language.
If information is missing, say so honestly.
Do not fabricate requirements.
Have a warm, welcoming, helpful personality.

Answer the question based only on the following context and chat history:

{context}

{history}

---

Answer the question based on the above context and chat history: {question}

"""

def main():
     
     # ******STAGE 4******
     ''' 
     # Allowes user input to be taken from command-line interface (CLI)
     parser = argparse.ArgumentParser() # Creates parser object to prepare script to read inputs from terminal
     parser.add_argument("query_text", type=str, help="The query text.") # input is stored as "query_text", type=str means treat input as a string, if someone runs --help - "The query text." will show along with other help text
     args = parser.parse_args() # reads what's been typed into CLI and is converted to object and is also stored
     query_text = args.query_text # takes stored input and puts it into query_text variable to be easily accessable
     '''
     
     embedding_function = OpenAIEmbeddings() # creates an embedding function, same for chunks, used to convert user query into embeddings
     db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function) # Opens document embeddings and prepares to check against query embeddings

     # console object created for cleaner CLI user experience
     console = Console()

     # conversation history list
     conversation_history = []

     print("\n🤖 Welcome to the UVA Engineering ChatBot! Type 'exit' to quit.\n")
     while True:
          query_text = console.input(f"[bold yellow]You: [/bold yellow]")
          if query_text.lower() in ["exit", "quit", "bye", "clear"]:
               break
          
          console.print("[dim]UVA Chatbot is typing...[/dim]\n", style="blink")

          # adds user input to conversation history     
          conversation_history.append({"role": "user", "content": query_text})     

          # query_text is converted to embedding vector, Chroma compares vector to each stored document vectors, calculates cosine similarity, and returns top k most similar chunks
          results = db.similarity_search_with_relevance_scores(query_text, k=4)
          # Used to check how similar or different each result is to query_text
          # In chroma: 1 = very similar; 0 = not similar at all
          # results are already sorted by Chroma in order of highest to lowest relevence score so only first one needs to be checked
          # Best matching chunk being checked against query embeddings
          # this is a confidence check, all k chunks are used as context_text for output generation later. But if best match is weak, program stops
          if len(results) == 0 or results[0][1] < 0.7:
               print("Unable to find matching results")
               continue
     
          # ******STAGE 5******
          
          # Loops through each item in results, capturing document page content, ignoring score, and joining that in str context_text
          context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
          # Loads prompt template above and prepares it to take in context_text and query_text
          # Prompt contains place holders {context} & {question}
          prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
          # gets history text for each item in conversation_history -> -4 represents how many items in conversation_history included in history_text
          # example: 
          # user: question
          # assistant: answer
          history_text = "\n".join([f"{item['role']}: {item['content']}" for item in conversation_history[-4:]])

          # Context, query, and history get inserted into prompt template and is ready to be sent to LLM
          prompt = prompt_template.format(
               context=context_text, 
               question=query_text,
               history=history_text
               )

          model = ChatOpenAI() # Selects model which will take in prompt and generateg response
          response_text = model.invoke(prompt) # Generates response using selected model
          conversation_history.append({"role": "assistant", "content": response_text.content}) # adds ai output to conversation history 

          # This is used to retrieve sources of where response comes from.
          # Each item in result contains metadata, which includes source information and is captured for each item in result
          # If source doesn't exists, None is returned
          sources = [doc.metadata.get("source", None) for doc, _score in results]

          FOLLOWUP_PROMPT_TEMPLATE = """
               Based on this question and answer:

               Question: {query_text}
               Answer: {response_text}

               Generate 3 short, natural follow-up question ideas the user might want to know next related to what they just asked.
               Remember to act as a UVA Engineering Academic Advisor
               Only return the questions as a numbered list.
          """
          followup_prompt = FOLLOWUP_PROMPT_TEMPLATE.format(query_text=query_text, response_text=response_text.content)

          followup_response = model.invoke(followup_prompt)

          suggestions = followup_response.content.split(f"\n")

          formatted_suggestions = "\n".join([s.strip() for s in suggestions if s.strip()])

           # Wrap question and response in a colored panel
          console.print(Panel(f"{query_text}", title="You"))
          console.print(Panel(f"[bold green]{response_text.content}[/bold green]\n\n[bold green]You might also ask:[/bold green]\n[bold green]{formatted_suggestions}[/bold green]", title="🤖 UVA Engineering Chatbot"))
          
          # sets timestamp format
          timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")

          # append query_text and response into log.txt
          with open(file="log.txt", mode="a", encoding="utf-8") as f:
               f.write(f"{timestamp} User: {query_text}\n\n")
               f.write(f"{timestamp} Assistent: {response_text.content}\n{formatted_suggestions}\n\n")
               f.write("-" * 60 + "\n\n")

          '''
          # Cleanly formats response and prints it to terminal
          formatted_response = f"\n{response_text.content}\n"
          #\n\n{"-"*50}\nSources: {sources}\n\n" # extra code if you want sources in response
          print(formatted_response)
          '''

# Only run main() if query_data is being run directly, not imported
if __name__ == "__main__":
    main()
