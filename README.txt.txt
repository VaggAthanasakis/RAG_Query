This python code is responsible for creating a Rag System using llama 3.1 8B
-> It takes as input a pdf file which is a Bill of Lading document.
-> As an output, it extracts info about shipper, consignee etc.
-> In the code, a comparison of the methods DocumentSummaryIndex, VectorStoreIndex is been      
   held to find out which one of the two is the better way of extracting the relevant info.

* The document can either be a native pdf file or a scanned document