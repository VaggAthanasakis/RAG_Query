import extract_msg
from llama_index.core import SimpleDirectoryReader
from llama_index.core import Document
from llama_index.core import VectorStoreIndex 
from langchain_community.llms import Ollama
from llama_index.core import Settings
from langchain.llms import BaseLLM
from llama_index.core.node_parser import SentenceSplitter


# Monkey patch the predict method of BaseLLM to use invoke instead
def patched_predict(self, prompt, **kwargs):
    return self.invoke(prompt, **kwargs)

BaseLLM.predict = patched_predict


# Load the .msg file
file_path = ("Resources\\TEST_EMAIL_2.msg")
msg = extract_msg.Message(file_path)

# Extract content (subject, body, etc.)
email_subject = msg.subject
email_body = msg.body
email_sender = msg.sender
email_receiver = msg.to
email_cc = msg.cc

# extract the attached files

attachments = msg.attachments
# Save the attachments to a specified folder
for attachment in attachments:
    attachment_filename = attachment.longFilename
    with open(f"outputs\\{attachment_filename}", "wb") as f:
        f.write(attachment.data)
    print(f"Attachment {attachment_filename} saved successfully.")


# Combine subject and body for summarization
email_content = f"Subject: {email_subject}\n\n{email_body}"

# Write the content of the email into a file
email_content_file = 'outputs//email_output.txt'
with open(email_content_file, 'w', encoding='utf-8') as f:
    f.write(email_content)     # or whatever text you're writing

documents = SimpleDirectoryReader(
    input_files=[email_content_file]
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
Settings.context_window = 1000

# Specify the splitter
splitter = SentenceSplitter(chunk_size=200)

#  This is a class from the llama_index library that represents a vector store index for efficient retrieval
#  of documents based on their semantic similarity.

vector_store_index = VectorStoreIndex.from_documents(documents, splitter=splitter) 
#vector_store_index = VectorStoreIndex.from_documents(documents)  
query_engine_vector_index = vector_store_index.as_query_engine()


#############################
# One_Shot Prompt
# Combine the one-shot example and the actual input for the final prompt
one_shot_prompt = f"""
Give the context of each sender for all the responses
INSTRUCTIONS:
Do not rewrite the question.
Do not give any other output. No intro or outro

Response format: Sender: {email_sender}\nReveiver: {email_receiver}\nCc: {email_cc}\nSubject: {email_subject}\n
"""

response_prompt = query_engine_vector_index.query(one_shot_prompt)
response_file_prompt = ("outputs//prompt_response.txt")
with open(response_file_prompt, "w", encoding='utf-8') as file:
    file.write(str(response_prompt))