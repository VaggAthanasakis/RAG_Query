from llama_index.core import SimpleDirectoryReader
from llama_index.core import Document
from llama_index.core import VectorStoreIndex, DocumentSummaryIndex
from llama_index.core.node_parser import SentenceSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from llama_index.core import Settings
from langchain.llms import BaseLLM
import pytesseract
from pdf2image import convert_from_path
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaEmbeddings
import pymupdf4llm
import pathlib

# Function to extract text from a scanned PDF using OCR
# Converts the pdf pages to images and then performs OCR on each page
# in order to extract the text
def extract_text_from_scanned_pdf(file_path):
    pages = convert_from_path(file_path, dpi=600)  # Convert PDF pages to images
    extracted_text = ""
    
    for page in pages:
        page_text = pytesseract.image_to_string(page)  # Perform OCR on each page
        extracted_text += f"{page_text}\n"             # Add the text to the output
    
    return extracted_text


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


## main
BaseLLM.predict = patched_predict

# input, output files
input_file_path = ("Resources/SOIL_ANALYSIS.pdf")
response_file = ("outputs/SOIL_ANALYSIS_RES.txt")
text_output_file = ("outputs/SOIL_ANALYSIS_TEXT.txt")


# Detect if the PDF is scanned
if is_pdf_scanned(input_file_path):
    # If the PDF is scanned, use OCR to extract the text

    # Use Pytesseract
    extracted_text = extract_text_from_scanned_pdf(input_file_path)
    with open(text_output_file, "w", encoding='utf-8') as file:
        file.write(str(extracted_text))

    documents = [Document(text=extracted_text)]
else:
    # If it's not scanned, load the document normally
    documents = SimpleDirectoryReader(
        input_files=[input_file_path]
    ).load_data()
    # Open the file in write mode
    with open(text_output_file, "w", encoding="utf-8") as file:
        for doc in documents:
        # Assuming each doc has a 'text' attribute containing the document's content
            file.write(doc.text + "\n\n")  # Write each document's content to the file


# load the LLM that we are going to use
llm = Ollama(model="llama3.1:8b", temperature = 0.1)

# https://docs.llamaindex.ai/en/stable/module_guides/supporting_modules/service_context_migration/
#embed_model = "local:BAAI/bge-small-en-v1.5"
#embed_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
#embed_model = SentenceTransformer('all-MiniLM-L6-v2')
embed_model = OllamaEmbeddings(model="llama3.1:8b")


# The Settings class in llama_index (formerly known as GPT Index) is used to configure
# global parameters that influence how the library interacts with language models (LLMs),
# embedding models, and other system components.
Settings.llm = llm
Settings.embed_model = embed_model
Settings.context_window = 2048


prompt = load_prompt("Prompts/Soil_Analysis_prompt.txt")


########################################################
# Try the VectorStoreIndex 

# Specify the splitter
splitter = SentenceSplitter(chunk_size=1000)
# splitter = RecursiveCharacterTextSplitter(
#     chunk_size=512,   # Size of each chunk
#     chunk_overlap=50  # Overlap of 50 tokens between chunks
# )


#  This is a class from the llama_index library that represents a vector store index for efficient retrieval
#  of documents based on their semantic similarity.
vector_store_index = VectorStoreIndex.from_documents(documents, splitter=splitter)   

# Build the query engine from the vector store index
query_engine_vector_index = vector_store_index.as_query_engine()

# Create the response based on the input file and the prompt
response = query_engine_vector_index.query(prompt)

# Store the response into a text file 
with open(response_file, "w", encoding='utf-8') as file:
    file.write("Using VectorStoreIndex\n\n")
    file.write(str(response))


########################################################
# Try DocumentSummaryIndex

document_summary_index = DocumentSummaryIndex.from_documents(documents,splitter=splitter)
query_engine_summary_index = document_summary_index.as_query_engine()

# You can use the same prompt with the query_engine_summary_index
response_summary = query_engine_summary_index.query(prompt)

with open(response_file, "a", encoding='utf-8') as file:
    file.write("\n\nUsing DocumentSummaryIndex\n\n")
    file.write(str(response_summary))
