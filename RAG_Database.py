from llama_index.core import Document
from llama_index.core import VectorStoreIndex
from langchain_ollama import OllamaLLM
from llama_index.core import Settings
from langchain.llms import BaseLLM
from PyPDF2 import PdfReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import pathlib
import pdfplumber
import ocrmypdf
import chromadb

# Function to detect if the PDF is scanned or not
def is_pdf_scanned(file_path):
    reader = PdfReader(file_path)
    num_pages = len(reader.pages)
    
    # Try to extract text from each page
    for page_num in range(num_pages):
        page = reader.pages[page_num]
        text = page.extract_text()
        if text and text.strip():  # If there's text, it's not scanned
            return False   
        
    return True  # No text found, likely a scanned PDF


# The line BaseLLM.predict = patched_predict overrides the deprecated predict method and uses invoke instead.
# This should ensure that anywhere the predict method is called within llama_index, it uses invoke
def patched_predict(self, prompt, **kwargs):
    return self.invoke(prompt, **kwargs)


# load a specific prompt from a given file
def load_prompt(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return file.read()

# Initialize ChromaDB
def init_chroma_db():
    client = chromadb.Client()
    collection = client.get_or_create_collection(name="DataBase")
    return collection


# input, output files
input_files_dir = "C:\Users\vagga\Desktop\RAG\RAG_Query\Resources\Soil_Analysis_Resources\ΕΔΑΦΟΣ ΤΟΠΟΘ ΚΑΣΑΠΑΚΗΣ"
response_file = ("C:\Users\vagga\Desktop\RAG\RAG_Query\outputs\SOIL_ANALYSIS_RES.txt")
text_output_file = ("C:\Users\vagga\Desktop\RAG\RAG_Query\outputs\SOIL_ANALYSIS_TEXT.txt")
prompt = load_prompt("C:\Users\vagga\Desktop\RAG\RAG_Query\Prompts\Soil_Analysis_JSON_prompt.txt")


## main
BaseLLM.predict = patched_predict


# Initialize ChromaDB
chroma_collection = init_chroma_db()

# Initialize Vector Store
vector_store = ChromaVectorStore(chroma_collection)

# load the LLM that we are going to use
llm = OllamaLLM(model="llama3.1:8b", temperature = 0.1)

# embedding model
embed_model = HuggingFaceEmbedding('paraphrase-multilingual-MiniLM-L12-v2')


# The Settings class in llama_index (formerly known as GPT Index) is used to configure
# global parameters that influence how the library interacts with language models (LLMs),
# embedding models, and other system components.
Settings.llm = llm
Settings.embed_model = embed_model
Settings.context_window = 2048


# Process and Add PDFs
documents = []
input_files = pathlib.Path(input_files_dir).rglob("*.pdf")  # Find all PDFs in the directory


for input_file_path in input_files:
    # Detect if the PDF is scanned
    if is_pdf_scanned(input_file_path):
        print(f"\nPerforming OCR on: {input_file_path}\n")
        ocrmypdf.ocr(str(input_file_path), str(input_file_path), image_dpi=600)

    # Extract text from the PDF
    with pdfplumber.open(input_file_path) as pdf:
        full_text = ""
        for page in pdf.pages:
            full_text += page.extract_text()  # Extracts text page-by-page

    # Save extracted text for reference
    pathlib.Path(text_output_file).write_bytes(full_text.encode())

    # Add document to the list
    documents.append(Document(text=full_text))


########################################################
# Try the VectorStoreIndex 

# Specify the splitter
#splitter = SentenceSplitter(chunk_size=700)

#  This is a class from the llama_index library that represents a vector store index for efficient retrieval
#  of documents based on their semantic similarity.
vector_store_index = VectorStoreIndex.from_documents(
    documents=documents,
    vector_store = vector_store)   

# Build the query engine from the vector store index
query_engine_vector_index = vector_store_index.as_query_engine()

# Create the response based on the input file and the prompt
response = query_engine_vector_index.query(prompt) 

# Store the response into a text file 
with open(response_file, "w", encoding='utf-8') as file:
    file.write("Using VectorStoreIndex\n\n")
    file.write(str(response))

