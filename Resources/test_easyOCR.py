import PIL.Image
import easyocr
import pymupdf4llm
import pathlib
import pytesseract
from pdf2image import convert_from_path
import cv2
import numpy as np
import PIL
from PIL import ImageDraw

input_file = "Resources/TEST_PDF_3.pdf"
output_file = "outputs/TEST_PDF_3_EASYOCR.txt"

reader = easyocr.Reader(['en'], gpu=True,verbose=False)

# Convert PDF to list of images
images = convert_from_path(input_file, dpi=600)  # Optional: Set a high DPI for better OCR accuracy

# Loop through each image (i.e., each page)
for i, image in enumerate(images):
  #image = PIL.Image.open() 
    image = np.array(image)
    bounds = reader.readtext(image)
    #print(bounds)
    for i in bounds:
        with open(output_file, 'a', encoding='utf-8') as f:
            for bound in bounds:
                f.write(bound[1] + '\n')  # Write each detected text and add a newline