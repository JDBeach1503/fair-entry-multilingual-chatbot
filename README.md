# Fair Entry Multilingual Chatbot, built with LangChain, (Azure) OpenAI, and ChromaDB

## Summary

This repo contains the files needed to interact with a PoC of a multilingual chatbot instructed to only provide data concerning the Fair Entry program at the City of Calgary

The documents that contain the data the chatbot uses can be found in the folder named `documents`

There are two ways of interacting with the chatbot, the first being via a Command-Line Interface (CLI) accessible from a terminal, and a web User Interface (UI) that will run on a local URL

The CLI can be utilized by running `multi_doc_cli.py`, and the UI can be accessed using `multi_doc_web_app.py`. Note that included in this repo are 2 more Python files (`azure_multi_doc_cli.py` and `azure_multi_doc_web_app.py`) that are in essence the same as their non-Azure counterparts, but include additonal code to use Azure OpenAI endpoints and API keys to access the Embeddings and Chat functionality, rather that just the regular OpenAI
service

Lastly, below is an inclusion of the resources that greatly assisted in providing me a jumping off point to develop a functioning Fair Entry chatbot:

[Tutorial: ChatGPT Over Your Data](https://blog.langchain.dev/tutorial-chatgpt-over-your-data/)
[chat-your-data](https://github.com/hwchase17/chat-your-data?ref=blog.langchain.dev)
[QA over Documents](https://python.langchain.com/docs/use_cases/question_answering/)
[Building a Multi-Document Reader and Chatbot With LangChain and ChatGPT](https://betterprogramming.pub/building-a-multi-document-reader-and-chatbot-with-langchain-and-chatgpt-d1864d47e339)
[multi-doc-chatbot](https://github.com/smaameri/multi-doc-chatbot)
[Gradio Theming Guide](https://www.gradio.app/guides/theming-guide)


## Overview of how it works

The folder `documents` is iterated through, and if the file format type is either .pdf or .html (or .htm, alternatively), then the file is loaded using the respective LangChain Document Loader as a Document (a Document in the context of LangChain is a piece of text with associated metadata). The Document Loaders that are used are `UnstructuredPDFLoader` and `UnstructuredHTMLLoader`, which leverage the `unstructured` Python library to ingest and pre-process text and images

The Documents are split into smaller pieces (called "chunks") by a LangChain Text Splitter for embedding and then vector storage. The Text Splitter that is used is `CharacterTextSplitter`, which defines how long a "chunk" will be based on the number of characters

The contents of each Document are embedded (converted from words to numerical vector representations). The embeddings are used to index the Document. `OpenAIEmbeddings` is used as the embedding model

The embeddings are stored in a vector store, and when the data is queried by the user, the vector store performs the vector search for the user. `Chroma` is used as the vector store integration. Data is stored in-memory, but `Chroma` has options for on-disk storage and configuring a client/server model

A Q&A chain is created (chains are end-to-end wrappers around multiple components, combining them together to create an LLM application)

The chain used is the `ConversationalRetrievalChain`, which allows the user to have a conversation with an LLM about information in retrieved documents

The API Reference documentation for `ConversationalRetrievalChain` describes the algorithm used in this chain as follows:

1. Use the chat history and the new question to create a “standalone question”. This is done so that this question can be passed into the retrieval step to fetch relevant documents. If only the new question was passed in, then relevant context may be lacking. If the whole conversation was passed into retrieval, there may be unnecessary information there that would distract from retrieval.

2. This new question is passed to the retriever and relevant documents are returned.

3. The retrieved documents are passed to an LLM along with either the new question (default behavior) or the original question and chat history to generate a final response.

(link to API Reference: [ConversationalRetrievalChain](https://api.python.langchain.com/en/latest/chains/langchain.chains.conversational_retrieval.base.ConversationalRetrievalChain.html))

The LLM used in the chain is the `gpt-3.5-turbo` model from OpenAI 


## Getting started

To get started, clone this repo and install the required packages

```
git clone https://github.com/USERNAME/REPOSITORY
cd fair-entry-multilingual-chatbot
pip install -r requirements.txt
```

## Setting your OpenAI or Azure OpenAI credentials

There are two copies of the `.env` file, one for use with the regular OpenAI service and one for use with the Azure OpenAI service.

The OpenAI `.env`file just contains

`OPENAI_API_KEY=`

as that was all that was needed to use both the Embeddings and Chat functionality.

The Azure OpenAI `.env.azure` file contains

`OPENAI_API_TYPE=`

`OPENAI_API_KEY=`

`OPENAI_API_PROXY=`

Type should be set to `azure`, and if a corporate proxy is being used then set the URL as well.

There are two more environment variables that could be set

`OPENAI_API_BASE=`

`OPENAI_API_VERSION=`

but since Azure OpenAI is being used for both the Embeddings and Chat functionality, the models will have different endpoints (`OPENAI_API_BASE`) and could have different versions (`OPENAI_API_VERSION`). Therefore, instead of setting these values as environment variables, they will be passed as named parameters in the constructors of their respective classes.

For example:

```python
embeddings = OpenAIEmbeddings(
    deployment="embeddings-deployment-name",

    model="embeddings-model-name",

    openai_api_base="https://embeddings-endpoint.openai.azure.com/",

    openai_api_version="yyyy-mm-dd"
)
```

```python
llm = AzureChatOpenAI(
    deployment_name="chat-deployment-name",

    openai_api_base="https://chat-endpoint.openai.azure.com/",

    openai_api_version="yyyy-mm-dd"
)
```


## Using the bot

Put the files you want to query into the `documents` folder. Run the below command from a terminal

```python
python3 multi_doc_cli.py
```

or use the "Run" option in your preferred editor to run `multi_doc_cli.py` to access the bot via the CLI

While using the CLI, you can query the data in your language of choice, and the bot should adapt to the language of the question without prior prompting to switch to that language. Enter `exit`, `quit`, `q`, `leave`, `done` to stop running the file

If you want to access the bot using the Web UI, then run the following command:

```python
python3 multi_doc_web_app.py
```
or use the "Run" option in your preferred editor to run `multi_doc_web_app.py` to access the UI

Like the CLI, you can query the data in your language of choice, and the bot should adapt to the language of the question without prior prompting to switch to that language. When done, you can close the window the UI is running in and you can use `Ctrl+C` in the terminal to stop running the file

If you want to use the AzureOpenAI service instead, follow the above steps with the exception of using the respective Azure files for the CLI and Web UI

NOTE: Please delete the `data` folder after you finish interacting with the chatbot, so that it can be recreated on the next run. If the code is altered and `data` is not deleted, the responses returned can be strange.


## License
LangChain is licensed under the [MIT](https://choosealicense.com/licenses/mit/) License, and in following with the conditions outlined the license and copyright notice are included in this repo
