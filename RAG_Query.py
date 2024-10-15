from llama_index.core import SimpleDirectoryReader
from llama_index.core import Document
from llama_index.core import VectorStoreIndex, DocumentSummaryIndex
from llama_index.core.node_parser import SentenceSplitter
from langchain_community.llms import Ollama
from llama_index.core import Settings
from langchain.llms import BaseLLM
import pytesseract
from pdf2image import convert_from_path
from PyPDF2 import PdfReader


# Function to extract text from a scanned PDF using OCR
# Converts the pdf pages to images and then performs OCR on each page
# in order to extract the text
def extract_text_from_scanned_pdf(file_path):
    pages = convert_from_path(file_path)  # Convert PDF pages to images
    extracted_text = ""
    
    for page in pages:
        text = pytesseract.image_to_string(page)  # Perform OCR on each page
        extracted_text += text + "\n\n"           # Add the text to the output
    
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



## main
BaseLLM.predict = patched_predict

# input, output files
input_file_path = ("Resources\\TEST_PDF_3.pdf")
response_file = ("outputs\\extracted_pdf.txt")

# Detect if the PDF is scanned
if is_pdf_scanned(input_file_path):
    # If the PDF is scanned, use OCR to extract the text
    extracted_text = extract_text_from_scanned_pdf(input_file_path)
    documents = [Document(text=extracted_text)]
else:
    # If it's not scanned, load the document normally
    documents = SimpleDirectoryReader(
        input_files=[input_file_path]
    ).load_data()



# load the LLM that we are going to use
llm = Ollama(model="llama3.1:8b", temperature = 0.5)

# https://docs.llamaindex.ai/en/stable/module_guides/supporting_modules/service_context_migration/
embed_model = "local:BAAI/bge-small-en-v1.5"


# The Settings class in llama_index (formerly known as GPT Index) is used to configure
# global parameters that influence how the library interacts with language models (LLMs),
# embedding models, and other system components.
Settings.llm = llm
Settings.embed_model = embed_model
Settings.context_window = 2000



# Create the prompt
prompt = f"""
Please extract the following details from the provided document:

Shipper: (Extract name after keyword "Shipper" or "Shipper name" or "Exporter")
CONSIGNEE: (Extract name after keyword "Consignee" or "Consignee name")
Document number: (Extract number (if exists) after keyword "Document number", "Doc No", or any reference to document number)
B/L Number: (Extract number (if exists) after "B/L Number" or "Bill of Lading Number")
Type of Cargo: (Extract type of cargo, such as containers, boxes, etc.)
Total Weight: (Extract weight, look for keywords like "Total weight" or "Gross weight")

Ensure each piece of information is extracted and presented as:

Shipper: [Extracted Shipper]
CONSIGNEE: [Extracted Consignee]
Document number: [Extracted Document Number]
B/L Number: [Extracted B/L Number]
Type of Cargo: [Extracted Cargo Type]
Total Weight: [Extracted Total Weight]

Instructions:
Do not rewrite the question.
Do not make an intro or an outro.
"""

# prompt = f"""
# Extract all terms and their definitions from the provided document. Focus on the glossary or sections that explicitly list terms. Include any abbreviations, acronyms, and their corresponding explanations.
# If a definition isn't given, provide a brief summary based on the context.

# **Output Format:**

# 1. Term: Definition

# **Examples:**

# 1. EXW: Ex Works
# 2. FOB: Free On Board
# 3. B/L: Bill of Lading
# 4. Terminal: The port or depot at which containers are loaded or unloaded onto or from container vessels, railways or trucks

# **Instructions:**

# - Include all terms in CAPITALS as well as commonly used acronyms.
# - Look for any bold or highlighted terms from the document.
# - Provide concise definitions, even if not explicitly mentioned in the text.
# - Do not add any additional comments, just the extracted terms and definitions.
# """


########################################################
# Try the VectorStoreIndex 

# Specify the splitter
splitter = SentenceSplitter(chunk_size=500)

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
