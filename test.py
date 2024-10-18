import torch
from transformers import LlamaForCausalLM, LlamaTokenizer,PreTrainedTokenizerFast
import PyPDF2
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import os

# Load LLaMA model and tokenizer (adjust paths as necessary)
model_path = "C:\\Users\\vagga\\Desktop\\Μαθήματα\\Μαθήματα\\Μαθήματα - TUC\\Διπλωματική\\Llama-3.1-8B-Instruct"

offload_folder = "offload"
os.makedirs(offload_folder, exist_ok=True)


tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
# Load the model and dispatch with safetensors support if applicable
try:
    model = LlamaForCausalLM.from_pretrained(model_path, use_safetensors=True)
except Exception as e:
    print(f"Error loading with safetensors: {e}")
    # Fallback to regular loading
    model = LlamaForCausalLM.from_pretrained(model_path)

# Automatically distribute the model across CPU and GPU
model = load_checkpoint_and_dispatch(model, model_path, device_map="auto", offload_folder=offload_folder)


# Check if CUDA is available and move the model to GPU if it is
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model.to(device)

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text

# Function to generate query response using LLaMA
def query_pdf(pdf_text, query, max_length=500):
    # Prepare input prompt for the model
    input_text = f"Document: {pdf_text}\n\nQuery: {query}\n\nAnswer:"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=2048)

    # Move the inputs to the proper device (accelerate will handle it)
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    # Generate response using LLaMA model
    output = model.generate(
        inputs.input_ids,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        early_stopping=True
    )

    # Decode and return the generated answer
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return answer

# Example usage
pdf_path = "Resources//TEST_PDF.pdf"
query = "What is the shipping policy mentioned in the document?"

# Extract text from the PDF
pdf_text = extract_text_from_pdf(pdf_path)

# Perform query on the PDF
response = query_pdf(pdf_text, query)

# Print the result
print("Query Response:")
print(response)
