from langchain_ollama import ChatOllama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import pdfplumber
from llama_index.core import Settings
from llama_index.core import Document
from langchain_core.messages import HumanMessage, SystemMessage



llm = ChatOllama(model="llama3.1:8b", temperature= 0)
#llm_json_mode = ChatOllama(model="llama3.1:8b", temperature=0, format="json")

input_file_path = "/home/eathanasakis/Thesis/RAG_Query/Resources/Thesis_Resources/PDFs/TEST_PDF.pdf"


embed_model = HuggingFaceEmbedding('paraphrase-multilingual-MiniLM-L12-v2')
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size = 200,
    chunk_overlap = 0
)

with pdfplumber.open(input_file_path) as pdf:
        full_text = ""
        for page in pdf.pages:
            full_text += page.extract_text()  # Extracts text page-by-page
            
documents = []
# Add document to the list
documents.append(Document(text=full_text))

Settings.llm = llm
Settings.embed_model = embed_model
Settings.context_window = 2048

vector_store_index = VectorStoreIndex.from_documents(documents,
                                                    splitter=text_splitter, 
                                                    #embed_model = embed_model,
                                                    #llm = llm,
                                                    show_progress=True)   


query_engine = vector_store_index.as_query_engine() # this retrives the most relevants docs and then combines 
                                                    # them with the input query plus the LLM in order to provide 
                                         

# Prompt
router_instructions = """You are an expert at routing a user question to a vectorstore or web search.

The vectorstore contains documents related to a marine company, including a bill of lading.

Use the vectorstore for questions on these topics. For all else, and especially for current events, use web-search.

Return JSON with single key, datasource, that is 'websearch' or 'vectorstore' depending on the question."""
