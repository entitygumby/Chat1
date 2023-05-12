# Use LoadQAWithSources for source, no memory
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.llms import OpenAI

import config
import logging
import streamlit as st
import os

#Creating the chatbot interface
st.set_page_config(
    layout="wide"
)

# Initialize logging with the specified configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(config.LOGS_FILE),
        logging.StreamHandler(),
    ],
)
LOGGER = logging.getLogger(__name__)

# Define answer generation function
def answer(prompt: str) -> str:

    # Log a message indicating that the function has started
    LOGGER.info(f"Start answering based on prompt: {prompt}.")

    # load persisted database from disk, and use it as normal
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPEN_API_KEY"])
    db = Chroma(persist_directory=config.PERSIST_DIR, embedding_function=embeddings)

    # Create a prompt template using a template from the config module and input variables
    # representing the context and question.
    prompt_template = PromptTemplate(template=config.prompt_template, input_variables=["summaries", "question"])

    # Initiate retriever
    docs = db.similarity_search(prompt)

    # used to combine QA_WITH_SOURCES functionality with Retrieval Sources Chain
    qa_chain = load_qa_with_sources_chain(
        llm=OpenAI(
            openai_api_key=st.secrets["OPEN_API_KEY"],
            model_name="gpt-3.5-turbo",
            temperature=0,
            max_tokens=300,
        ),
        chain_type="stuff",
        prompt=prompt_template,
    )


    # Log a message indicating the number of chunks to be considered when answering the user's query.
#    LOGGER.info(f"The top {config.k} chunks are considered to answer the user's query.")

    # Call the QA object to generate an answer to the prompt.
    result = qa_chain({"input_documents": docs, "question": prompt}, return_only_outputs=True)
    answer = result['output_text']

    # Log a message indicating the answer that was generated
    LOGGER.info(f"The returned answer is: {answer}")

    # Log a message indicating that the function has finished and return the answer.
    LOGGER.info(f"Answering module over.")
    return answer
