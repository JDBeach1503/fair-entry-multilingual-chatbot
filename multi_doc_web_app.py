import os
from dotenv import load_dotenv
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate
from typing import Optional, Tuple
from threading import Lock
import gradio as gr

#load OpenAI API key from .env file
load_dotenv('.env')

os.environ['OCR_AGENT'] = 'tesseract'

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
embeddings = OpenAIEmbeddings()

vectordb = Chroma.from_documents(split_data, embeddings, persist_directory="./data")

vectordb.persist()

#Create Q&A chain
#Chains in LangChain are end-to-end wrappers around multiple components, combining them together to create an application
llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo')

retriever = VectorStoreRetriever(vectorstore=vectordb)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

doc_qa = ConversationalRetrievalChain.from_llm(

    llm=llm,

    retriever=retriever,

    memory=memory,

    combine_docs_chain_kwargs={"prompt": CUSTOM_PROMPT}

    )

#define a function to set the api key and return the chain (which was created earlier in the code)
def set_openai_api_key(api_key: str):

    if api_key:

        os.environ["OPENAI_API_KEY"] = api_key

        chain = doc_qa

        os.environ["OPENAI_API_KEY"] = ""

        return chain


class ChatWrapper:

    def __init__(self):

        self.lock = Lock()

    def __call__(
        self, api_key: str, inp: str, history: Optional[Tuple[str, str]], chain
    ):
 
        self.lock.acquire()

        try:

            history = history or []

            #If chain is None, that is because no api key was provided previously which causes set_openai_api_key() to return None.
            if chain is None:

                history.append((inp, "Please paste your OpenAI key to use"))

                return history, history
            
            # Set OpenAI key
            import openai

            openai.api_key = api_key

            # Run chain and append the input.
            output = chain({"question": inp})["answer"]

            history.append((inp, output))

        except Exception as e:

            raise e
        
        finally:

            self.lock.release()

        return history, history

#Create web UI using Gradio
chat = ChatWrapper()

#gr.theme sets overall appearance of UI. In this case, used the prebuilt theme Soft and edited a few of the CSS variable values
theme = gr.themes.Soft(

    primary_hue="red"

).set(

    body_text_color='*neutral_950',

    background_fill_primary='*neutral_300',

    block_background_fill='*primary_100',

    border_color_primary='*neutral_700',

    body_text_color_subdued='*neutral_700'
)

with gr.Blocks(theme=theme) as block:
    with gr.Row():

        #Create a Markdown output to act as a title for the chatbot
        gr.Markdown(
            "<h3><center>City of Calgary Fair Entry Program Chatbot</center></h3>")
        
        #Create a textbox for user to paste their OpenAI key into
        openai_api_key_textbox = gr.Textbox(

            placeholder="Paste your OpenAI key",

            show_label=False,

            lines=1,

            type="password",
        )

    chatbot = gr.Chatbot()

    with gr.Row():

        #Create a textbox for user to enter their question for the chatbot
        message = gr.Textbox(

            label="Please ask your question",

            placeholder="Ask questions about Fair Entry or the programs it subsidizes",

            lines=1,
        )

        #Submit button
        submit = gr.Button(value="Submit", variant="secondary").style(
            full_width=False)
    
    #Sample questions that, when clicked on, will autopopulate the question textbox. User still has to click Submit themselves though
    gr.Examples(

        examples=[
            "What is the Fair Entry program?",
            "Where can I submit my Fair Entry application?",
            "How do I speak to a human agent?",
            "What is the fax number for Fair Entry?",
            "Are there fees for applying to Fair Entry?"
        ],
        inputs=message,

    )

    gr.HTML("Demo application of LangChain, OpenAI, ChromaDB, and Gradio.")

    gr.HTML(
        "<center>Powered by <a href='https://github.com/hwchase17/langchain'>LangChain 🦜️🔗</a></center>"
    )

    #gr.State is a hidden component that stores session state across application runs by the user
    #The value of the State variable is cleared when the page is refreshed
    state = gr.State()
    agent_state = gr.State()

    #User can submit the question by either the Submit button (submit.click()) or pressing the Enter key (message.submit())
    submit.click(chat, inputs=[openai_api_key_textbox, message,
                 state, agent_state], outputs=[chatbot, state])
    message.submit(chat, inputs=[openai_api_key_textbox, message, state, agent_state], outputs=[chatbot, state])

    openai_api_key_textbox.change(
        set_openai_api_key,
        inputs=[openai_api_key_textbox],
        outputs=[agent_state],
    )

#runs the UI on a local URL
block.launch()