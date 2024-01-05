import os
import sys
from dotenv import load_dotenv
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate


#load Azure OpenAI API type, key, and proxy (if used) from .envazure file
load_dotenv('.env.azure')

#Create a template for a string prompt. Supposed to give the chatbot a particular conversational style, in this case a multilingual chatbot for Fair Entry
template = """You are an AI assistant for answering questions about the Fair Entry program provided by the City of Calgary and any of the programs from the 
City of Calgary that are subsidized by Fair Entry.
You are given the following extracted parts from various documents and a question. Provide a concise answer.
If you don't know the answer, just say "I am not sure, sorry" Don't try to make up an answer.
If the question is not about the Fair Entry program or any of the programs from the City of Calgary
that are subsidized by Fair Entry, politely inform them that you are trained to only answer questions about those topics.
Lastly, you are a multilingual chatbot, so answer in whatever language the user asks the question in, adjusting to a new language if 
the user changes the language they ask their question in.
Question: {question}
=========
{context}
=========
Answer in Markdown:"""
CUSTOM_PROMPT = PromptTemplate(template=template, input_variables=[
                           "question", "context"])

#Create a list of pdf and html documents from the files in the ./docs folder
raw_data = []

"""Loops through available files, appends file name onto current path depending on file format,
stores path in a variable and loads unstructured data using respective loaders"""
for file in os.listdir("documents"):

    if file.endswith(".pdf"):

        pdf_path = "./documents/" + file

        loader = UnstructuredPDFLoader(pdf_path)

        raw_data.extend(loader.load())

    elif file.endswith('.html') or file.endswith('.htm'):

        html_path = "./documents/" + file

        loader = UnstructuredHTMLLoader(html_path)

        raw_data.extend(loader.load())

#Split the documents into smaller chunks for embeddings and then vector storage
text_splitter = CharacterTextSplitter(chunk_size=2500, chunk_overlap=10)

split_data = text_splitter.split_documents(raw_data)

#Convert chunks to embeddings (vector representations of text) and then store them in a vector database
#Store this database info in the ./data directory
embeddings = OpenAIEmbeddings(
    deployment="embeddings-deployment-name",

    model="embeddings-model-name",

    openai_api_base="https://embeddings-endpoint.openai.azure.com/",

    openai_api_version="yyyy-mm-dd"
)

vectordb = Chroma.from_documents(split_data, embeddings, persist_directory="./data")

vectordb.persist()

#Create Q&A chain
#Chains in LangChain are end-to-end wrappers around multiple components, combining them together to create an application
llm = AzureChatOpenAI(
    deployment_name="chat-deployment-name",

    openai_api_base="https://chat-endpoint.openai.azure.com/",

    openai_api_version="yyyy-mm-dd",

    temperature=0
)

retriever = VectorStoreRetriever(vectorstore=vectordb)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

doc_qa = ConversationalRetrievalChain.from_llm(

    llm=llm,

    retriever=retriever,

    memory=memory,

    combine_docs_chain_kwargs={"prompt": CUSTOM_PROMPT}

    )

#Create a CLI for user to interact with chatbot
yellow = "\033[0;33m"

green = "\033[0;32m"

white = "\033[0;39m"

chat_history = []

print(f"{yellow}Welcome to the Fair Entry program chatbot. Ask your questions in any language you would like!")
print('---------------------------------------------------------------------------------')

while True:
    query = input(f"{green}Prompt: ")

    if query == "exit" or query == "quit" or query == "q" or query == "leave" or query == "done":

        print('Exiting')

        sys.exit()

    if query == '':

        continue

    result = doc_qa(
        {"question": query, "chat_history": chat_history})
    
    print(f"{white}Answer: " + result["answer"])
    
    chat_history.append((query, result["answer"]))