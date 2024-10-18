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
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from transformers import BitsAndBytesConfig
from langchain.llms.base import LLM

# Function to extract text from a scanned PDF using OCR
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

class CustomHFLLM(LLM):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def predict(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_length=512)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)



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

# Setup CUDA for transformer-based LLM usage
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# If you want to use Ollama, ensure that it is configured for CUDA (depends on Ollama support).
# Here, I’ll show how to use Hugging Face's LLaMA model with GPU

# Configure quantization (8-bit or 4-bit)
quantization_config = BitsAndBytesConfig(load_in_4bit=True, load_in_8bit_fp32_cpu_offload=True)

# Define a custom device map, keeping parts of the model on the CPU to save GPU memory
# device_map = {
#     "transformer.wte": "cpu",   # Embeddings to CPU
#     "transformer.wpe": "cpu",   # Positional encodings to CPU
#     "transformer.ln_f": "cpu",  # Final LayerNorm to CPU
#     "transformer.h": "cpu",  # Main transformer layers on GPU
#     "lm_head": "cpu"            # Output layer to CPU
# }
device_map = "cpu"



# Load the LLaMA model from Hugging Face and move to GPU
model_name = "C:\\Users\\vagga\\Desktop\\Μαθήματα\\Μαθήματα\\Μαθήματα - TUC\Διπλωματική\\Llama-3.1-8B-Instruct" # Example model, replace with the actual model you're using
tokenizer = AutoTokenizer.from_pretrained(model_name)
llm = AutoModelForCausalLM.from_pretrained(model_name,
                                           quantization_config=quantization_config,
                                           torch_dtype=torch.bfloat16,
                                           device_map=device_map)#.to(device)


# Load the embedding model
embed_model_name = "local:BAAI/bge-small-en-v1.5"  # Replace with your embedding model
# embed_tokenizer = AutoTokenizer.from_pretrained(embed_model_name)
# embed_model = AutoModel.from_pretrained(embed_model_name)#.to(device)

# Initialize the custom wrapper
custom_llm = CustomHFLLM(model=llm, tokenizer=tokenizer)


Settings.llm = custom_llm
Settings.embed_model = embed_model_name
Settings.context_window = 1000

# Prepare document texts for embedding
# document_texts = [doc.text for doc in documents]
# embedding_inputs = embed_tokenizer(document_texts, padding=True, truncation=True, return_tensors="pt").to(device)

# # Get embeddings from the model
# with torch.no_grad():
#     embeddings = embed_model(**embedding_inputs).last_hidden_state.mean(dim=1)  # Average pooling


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

# Tokenize the input and pass it through the model
# inputs = tokenizer(prompt, return_tensors="pt").to(device)
# outputs = llm.generate(**inputs, max_length=512)

# # Decode the generated text from tokens to a human-readable form
# response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# # Store the response into a text file
# with open(response_file, "w", encoding='utf-8') as file:
#     file.write("Using Transformer Model\n\n")
#     file.write(response_text)


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
