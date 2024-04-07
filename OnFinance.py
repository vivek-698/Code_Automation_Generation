import os
from datasets import load_dataset
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_transformers import Html2TextTransformer
from langchain.document_loaders import AsyncChromiumLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA

from datasets import load_dataset

dataset = load_dataset("neulab/conala")

print(dataset)

for row in dataset['train']:
    df=(row['intent'], row['snippet'])
    print(df)

intent_snippet_list = []

for row in dataset['train']:
  intent_snippet_list.append([row['intent'], row['snippet']])

print(intent_snippet_list)

loader = TextLoader('/Users/rapakavivek/Downloads/intent_snippets.txt')
documents = loader.load()
print(documents)

text_splitter = CharacterTextSplitter(chunk_size=100,
                                        chunk_overlap=0)

chunked_documents = text_splitter.split_documents(documents)
print(chunked_documents)


from langchain_community.embeddings import OllamaEmbeddings
db = FAISS.from_documents(chunked_documents,
                            OllamaEmbeddings(model="nomic-embed-text"))

retriever = db.as_retriever()

## Prompt Engineering ##

prompt_template = """
### [INST] Instruction: You are a code snippet generator. Given a natural language query as input, provide the corresponding code snippet.
# Here is the context so that you can refer to the intent and its respective snippet:


# {context}

# ### QUESTION:
# {question} [/INST]
#  """

# Create prompt from prompt template
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)


import ollama


def rag_chain(question):

    retrieved_docs = retriever.invoke(question)
    
    response = ollama.chat(model='mistral:instruct', messages=[{'role': 'user', 'content': prompt_template.format(context=retrieved_docs, question=question)}])

    return response['message']['content']

import gradio as gr

iface = gr.Interface(
    fn=rag_chain,
    inputs=["text"],
    outputs="text",
    title="RAG Pipeline ",
    description="Natural Language Automation Code Generation"
)

# Launch the app
iface.launch(share=True)

