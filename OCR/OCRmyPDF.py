import ocrmypdf
from llama_index.core import SimpleDirectoryReader


input_file = "RAG_Query/Resources/TEST_PDF_2.pdf"
output_file = "RAG_Query/OCR/OCR_outputs/TEST_PDF_2_OCRmyPDF_def.pdf"
text_output_file = "RAG_Query/OCR/OCR_outputs/TEST_PDF_2_OCRmyPDF.txt"


ocrmypdf.ocr(input_file, input_file, image_dpi=600)

# Use SimpleDirectoryReader
documents = SimpleDirectoryReader(
        input_files=[input_file]
    ).load_data()

 # Open the file in write mode
with open(text_output_file, "w", encoding="utf-8") as file:
    for doc in documents:
        # Assuming each doc has a 'text' attribute containing the document's content
            file.write(doc.text + "\n\n")  # Write each document's content to the file
