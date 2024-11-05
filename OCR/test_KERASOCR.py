import pathlib
import numpy as np
import keras_ocr
from pdf2image import convert_from_path
import tensorflow as tf
import matplotlib.pyplot as plt


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))



input_file = "Resources/TEST_PDF_3.pdf"
output_file = "outputs/TEST_PDF_3_KERASOCR.txt"

# Initialize the keras-ocr pipeline
pipeline = keras_ocr.pipeline.Pipeline()

# Convert PDF to list of images
images = convert_from_path(input_file, dpi=600)  # Optional: Set a high DPI for better OCR accuracy

prediction_groups = []

# Loop through each image (i.e., each page)
for i, image in enumerate(images):
    # Convert PIL image to numpy array
    image = np.array(image)
    
    # Use keras-ocr to detect text
    bounds = pipeline.recognize([image])[0]
    prediction_groups.append(bounds)  # Store predictions for this image

    # fig, axs = plt.subplots(nrows=len(images), figsize=(10, 20))
    # keras_ocr.tools.drawAnnotations(image=image, 
    #                                 predictions=prediction_groups, 
    #                                 ax=axs)

    # Append each detected text to the output file
    with open(output_file, 'a', encoding='utf-8') as f:
        for bound in bounds:
            f.write(bound[0] + '\n')  # Write each detected text and add a newline


# Create subplots for displaying images and their predictions
fig, axs = plt.subplots(nrows=len(images), figsize=(10, 20))

# Create subplots for displaying images and their predictions
# Make sure axs is a 1D array
if len(images) == 1:
    axs = [axs]  # Convert single Axes to list

for ax, image, predictions in zip(axs, images, prediction_groups):
    keras_ocr.tools.drawAnnotations(image=np.array(image), 
                                     predictions=predictions, 
                                     ax=ax)
    ax.axis('off')  # Turn off axis

plt.tight_layout()
plt.show()