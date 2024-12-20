from llama_index.core import SimpleDirectoryReader
from llama_index.core import Document
from llama_index.core import VectorStoreIndex, DocumentSummaryIndex
from llama_index.core.node_parser import SentenceSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM
from llama_index.core import Settings
from langchain.llms import BaseLLM
import pytesseract
from pdf2image import convert_from_path
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaEmbeddings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import pathlib
import pymupdf4llm
import pdfplumber
import ocrmypdf
from deep_translator import GoogleTranslator
from langdetect import detect
import json
#import fitz
#from pdfminer.high_level import extract_text
#from tika import parser



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


# function to split text into segments of max_length length
def split_text(text, max_length=2000):
    segments = []
    while len(text) > max_length:
        # Find the last space within the 5000 character limit to avoid splitting words
        split_index = text[:max_length].rfind(' ')
        segments.append(text[:split_index])
        text = text[split_index + 1:]
    segments.append(text)  # Add any remaining text as the last segment
    return segments

# Function that translates each chunk from greek to english
def translate_text_in_chunks(text, source_lang='auto', target_lang='en'):
    segments = split_text(text)
    translated_segments = []

    for segment in segments:
        translated_text = GoogleTranslator(source=source_lang, target=target_lang).translate(segment)
        translated_segments.append(translated_text)

    # Join all translated segments into a single text
    return ' '.join(translated_segments)


# function to translate the input of the LLM in English In order to achieve betterc accuracy
def text_translator(greek_text):
    translated_text = translate_text_in_chunks(greek_text)
   
    return translated_text

## main
BaseLLM.predict = patched_predict

# input, output files
# input_file_path = ("/home/eathanasakis/Thesis/RAG_Query/Resources/Soil_Analysis_Resources/Soilanalysis-38-Zannias/Zanias/108/240438-zannias-kephales-bio.pdf")
# response_file = ("/home/eathanasakis/Thesis/RAG_Query/outputs/SOIL_ANALYSIS_RES.txt")
# text_output_file = ("/home/eathanasakis/Thesis/RAG_Query/outputs/SOIL_ANALYSIS_TEXT.txt")

# input_file_path = ("RAG_Query/OCR/OCR_outputs/TEST_PDF_3_OCRmyPDF_def.pdf") 
# response_file = ("RAG_Query/outputs/TEST_PDF_3_RES.txt")
# text_output_file = ("RAG_Query/outputs/TEST_PDF_3_TEXT.txt")

input_file_path = "/home/eathanasakis/Thesis/RAG_Query/Resources/Thesis_Resources/PDFs/TEST_PDF_2.pdf"
response_file = "/home/eathanasakis/Thesis/RAG_Query/outputs/TEST_PDF_RES.txt"
text_output_file = ("/home/eathanasakis/Thesis/RAG_Query/outputs/TEST_PDF_TEXT.txt")


# Detect if the PDF is scanned
if is_pdf_scanned(input_file_path):
    # If the PDF is scanned, use OCR to extract the text

    # Use Pytesseract
    # extracted_text = extract_text_from_scanned_pdf(input_file_path)
    # with open(text_output_file, "w", encoding='utf-8') as file:
    #     file.write(str(extracted_text))
    
    # if the input file is scanned, convert it to native pdf
    # using ocr and then process it normally
    print("\nPerforming OCR...\n")
    ocrmypdf.ocr(input_file_path, input_file_path, image_dpi=600)    
    
# If it's not scanned, load the document normally
# Use PDFPlumber
#text_output_file = "outputs/SOIL_ANALYSIS_TEXT_PDFPlumber.txt"
# with pdfplumber.open(input_file_path) as pdf:
#     full_text = ""
#     for page in pdf.pages:
#         full_text += page.extract_text()  # Extracts text page-by-page


# # # Check if we have to translate the text
# # if (detect(full_text) != 'en'):
# #     print("\nTranslating the text..")
# #     full_text = text_translator(full_text)

# pathlib.Path(text_output_file).write_bytes(full_text.encode())

# documents = [Document(text=full_text)]
    
    # with open("RAG_Query/outputs/translated.txt", encoding='utf-8') as file:
    #     translated_text = file.read()
    # documents = [Document(text=translated_text)]

    # # Use SimpleDirectoryReader
documents = SimpleDirectoryReader(
    input_files=[input_file_path]).load_data()

    # # Open the file in write mode
with open(text_output_file, "w", encoding="utf-8") as file: 
    for doc in documents:
      # Assuming each doc has a 'text' attribute containing the document's content
      file.write(doc.text + "\n\n")  # Write each document's content to the file

    # # Use pymupdf4llm 
    # text_output_file = "outputs/SOIL_ANALYSIS_TEXT_FOR_LLM.txt"
    # full_text = pymupdf4llm.to_markdown(input_file_path)
    # pathlib.Path(text_output_file).write_bytes(full_text.encode())
    

    # # Use fitz
    # text_output_file = "outputs/SOIL_ANALYSIS_TEXT_fitz.txt"
    # doc = fitz.open(input_file_path)
    # full_text = ""
    # for page_num in range(doc.page_count):
    #     page = doc[page_num]
    #     full_text += page.get_text()  # 'text' parameter extracts p
    # pathlib.Path(text_output_file).write_bytes(full_text.encode())

    # # Use PDFMiner
    # text_output_file = "outputs/SOIL_ANALYSIS_TEXT_PDFMiner.txt"
    # full_text = extract_text(input_file_path)
    # pathlib.Path(text_output_file).write_bytes(full_text.encode())

    # # Use Tika
    # text_output_file = "outputs/SOIL_ANALYSIS_TEXT_Tika.txt"
    # parsed = parser.from_file(input_file_path)
    # full_text = parsed["content"]
    # pathlib.Path(text_output_file).write_bytes(full_text.encode())

    # # Use PyPDF2
    # text_output_file = "outputs/SOIL_ANALYSIS_TEXT_PyPDF2.txt"
    # reader = PdfReader(input_file_path)
    # full_text = ""
    # for page in reader.pages:
    #     full_text += page.extract_text()
    # pathlib.Path(text_output_file).write_bytes(full_text.encode())




# load the LLM that we are going to use
llm = OllamaLLM(model="llama3.1:8b", temperature = 0)

# https://docs.llamaindex.ai/en/stable/module_guides/supporting_modules/service_context_migration/
#embed_model = "local:BAAI/bge-small-en-v1.5"
#embed_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
#embed_model = SentenceTransformer('all-MiniLM-L6-v2')
#embed_model = OllamaEmbeddings(model="llama3.1:8b")
#embed_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
#embed_model = HuggingFaceEmbedding(model_name='all-MiniLM-L6-v2')
embed_model = HuggingFaceEmbedding('paraphrase-multilingual-MiniLM-L12-v2')
#embed_model = "local:BAAI/bge-small-en-v1.5"



# The Settings class in llama_index (formerly known as GPT Index) is used to configure
# global parameters that influence how the library interacts with language models (LLMs),
# embedding models, and other system components.
Settings.llm = llm
Settings.embed_model = embed_model
Settings.context_window = 2048


prompt = load_prompt("/home/eathanasakis/Thesis/RAG_Query/Prompts/Info_extraction_prompt.txt")


########################################################
# Try the VectorStoreIndex 

# Specify the splitter
splitter = SentenceSplitter(chunk_size=700)
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

# Convert the output into a json string
# dict_string = json.loads(str(response))
# json_string = json.dumps(dict_string)
# print("\n",json_string)

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
