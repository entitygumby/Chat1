from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import DirectoryLoader
import config


# Load documents from the specified directory using a DirectoryLoader object
loader = DirectoryLoader(config.FILE_DIR, glob='*.pdf')
documents = loader.load()

# split the text to chuncks of of size 1000
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# Split the documents into chunks of size 1000 using a CharacterTextSplitter object
texts = text_splitter.split_documents(documents)

# Create a vector store from the chunks using an OpenAIEmbeddings object and a Chroma object
# Supplying a persist_directory will store the embeddings on disk
persist_directory = config.PERSIST_DIR

embeddings = OpenAIEmbeddings(openai_api_key=config.OPENAI_API_KEY)
docsearch = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory)
docsearch.persist()
docsearch = None
