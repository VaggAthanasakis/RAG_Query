from langchain_ollama import ChatOllama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from llama_index.core import Document
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import pdfplumber
from llama_index.core import Settings
from langchain_core.tools import tool
import warnings
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from langchain.schema import AIMessage


# Ignore all warnings
warnings.filterwarnings("ignore")

# load a specific prompt from a given file
def load_prompt(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return file.read()


def RAG():

    input_file_path = r"C:\Users\vagga\Desktop\RAG\RAG_Query\Resources\Thesis_Resources\PDFs\TEST_PDF.pdf"

    llm = ChatOllama(model="llama3.1:8b", temperature= 0)
    # load the LLM that we are going to use

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

    return vector_store_index

# retriever = vector_store_index.as_retriever()       # this retrieves the raw k=2 docs from the documents
# query_engine = vector_store_index.as_query_engine() # this retrives the most relevants docs and then combines 
#                                                     # them with the input query plus the LLM in order to provide 
#                                                     # the proper response

@tool                                       
def retrieve_documents(query: str) -> str:
      """Retrieve documents from the vector store based on the query."""
      print("\nCalling retrieve_documents Tool")
      retriever = RAG().as_query_engine()    # this retrieves the raw k=2 docs from the documents
      return str(retriever.retrieve(query))

@tool
def response_by_docs(query: str) -> str:
      """Provide an appropriate response to the input based on the vector store"""
      print("\nCalling response_by_docs Tool")
      query_engine = RAG().as_query_engine()
      return str(query_engine.query(query))



tools = [retrieve_documents, response_by_docs]

llm = ChatOllama(model="llama3.1:8b", temperature= 0)


# prompt = ChatPromptTemplate.from_messages([
#      ("system", "Υou are responsible to retrieve the appropriate information from a given document."),
#      ("placeholder", "{messages}"),
#      ("user", "Give the response to a json format"),
# ])


# create the agent
agent_executor = create_react_agent(llm,
                                    tools=tools)


# response = agent_executor.invoke({"arg1": "Shipper", "arg2": "Destination of cargoo"})

prompt = load_prompt(r"C:\Users\vagga\Desktop\RAG\RAG_Query\Prompts\Info_extraction_prompt.txt")

response = agent_executor.invoke({"messages": [HumanMessage(content=prompt)]})["messages"]

final_message = next(
    (message for message in response if isinstance(message, AIMessage) and message.content),
    None
).content

print(final_message)




# template ="""
# Give me the most relevant doc regarding the {topic}

# """

# template ="""
# Who is the {topic}

# """

# prompt = ChatPromptTemplate.from_template(template=template)





# chain = prompt | llm

# #response = chain.invoke({"topic" : "SHIPPER"})
# result = chain.invoke({"topic":"Shipper"})

