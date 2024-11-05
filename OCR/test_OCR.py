import pymupdf4llm
import pathlib
import pytesseract
from pdf2image import convert_from_path
import cv2
import numpy as np


# Input and output file paths
input_file = "Resources/TEST_PDF_3.pdf"
output_file = "outputs/TEST_PDF_3_PLAΙN_TEXT.txt"

# Convert PDF to list of images
images = convert_from_path(input_file, dpi=600)  # Optional: Set a high DPI for better OCR accuracy

# Initialize an empty string to store the extracted text
extracted_text = ""

# custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,-"'
# Loop through each image (i.e., each page)
for i, image in enumerate(images):
    
    image = np.array(image)

    # Noise Removal
    # This step removes the small dots/patches which have high intensity compared
    # to the rest of the image for smoothening of the image.
    # OpenCV’s fast Nl Means Denoising Coloured function can do that easily.
    cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 15)
    
    # Normalization
    norm_img = np.zeros((image.shape[0], image.shape[1]))
    image = cv2.normalize(image, norm_img, 0, 255, cv2.NORM_MINMAX)
    
 
    #image = np.array(image)

    # Apply binary thresholding
    #_, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Convert to Grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # denoise the image
    image = cv2.GaussianBlur(image, (5,5), 0)

    # Set the PSM mode to, for example, single line (7)
    custom_config = r'--psm 4'

    page_text = pytesseract.image_to_string(image,config=custom_config)
    # Append the page's text to the full extracted text
    extracted_text += f"{page_text}\n"

#extracted_text = pytesseract.image_to_string(images, config=custom_config)
pathlib.Path(output_file).write_bytes(extracted_text.encode('utf-8'))



# text = pymupdf4llm.to_markdown(input_file)
# pathlib.Path(output_file).write_bytes(text.encode())