RAG_Query.py
This python code is responsible for creating a Rag System using llama 3.1 8B
-> It takes as input a pdf file which is a Bill of Lading document.
-> As an output, it extracts info about shipper, consignee etc.
-> In the code, a comparison of the methods DocumentSummaryIndex, VectorStoreIndex is been      
   held to find out which one of the two is the better way of extracting the relevant info.
-> The same code has been used, with a different prompt, in order to extract terminology from 2 different documents.

* The document can either be a native pdf file or a scanned document

email_extractor.py
This python code is responsible for creating a Rag System using llama 3.1 8B
-> It takes as input a .msg file
-> It extracts all the attached files and stores them at the folder "outputs"
-> VectorStoreIndex in used in order to perform queries in the input
-> The response contains the sender, the receivers and Ccs and the email subject plus the context of
   each sender.