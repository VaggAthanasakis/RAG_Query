import pymupdf4llm
import pathlib

input_file = "Resources//TEST_PDF.pdf"
output_file = "outputs//TEST_PDF_TEXT.txt"


text = pymupdf4llm.to_markdown(input_file)
pathlib.Path(output_file).write_bytes(text.encode())