from langchain.prompts import ChatPromptTemplate
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


# The line BaseLLM.predict = patched_predict overrides the deprecated predict method and uses invoke instead.
# This should ensure that anywhere the predict method is called within llama_index, it uses invoke
def patched_predict(self, prompt, **kwargs):
    return self.invoke(prompt, **kwargs)

## main
BaseLLM.predict = patched_predict

# load the LLM that we are going to use
llm = OllamaLLM(model="llama3.1:70b", temperature = 0.1)


template_string = """
You are provided with a document that contains a soil analysis. Domument: {doc}

Extract the following information:

1. Full name
2. Type of cultivation 
3. Place
4. Μechanical soil structure (Μηχανική Σύσταση in greek): List elements (Άμμος (Sand), Ιλύς (Silt), Άργιλος (Clay)) in the basic soil analysis along with its measured value and unit. 
5. Physicochemical properties (Φυσικοχημικές Ιδιότητες): List each element (pH, Ηλεκτ. Αγωγιμότητα, Οργανική Ουσία), including its measured value and unit in the format: Element: Value (Unit).
6. Available nutritional forms (Διαθέσιμες μορφές Θρεπτικών): Identify and list each available nutritional form, along with its value and unit in the format: Element: Value (Unit). 
7. Evaluation (Αξιολόγηση), from the basic soil analysis, extract the Evaluation of the soil. This should be one word ("Αργιλώδες" or "Αμμώδες" or "Ασβεστώδες" or "Ιλυώδες" or "Πηλώδες" or something else).


Instructions:
You must not create an intro or outro, just give the above info ONLY.
Do not provide any additional text, just list the 7 nodes described above.

"""


with open("RAG_Query/outputs/SOIL_ANALYSIS_TEXT.txt") as file:
    document_text = file.read()

prompt_template = ChatPromptTemplate.from_template(template_string)
llm_input = prompt_template.format_messages(doc = document_text)

response = llm.invoke(llm_input)
print(response)