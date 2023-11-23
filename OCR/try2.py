import keras_ocr
import matplotlib.pyplot as plt
plt.ioff()
# Create a Keras-OCR pipeline
pipeline = keras_ocr.pipeline.Pipeline()

# Define a list of image paths to process
image_paths = [
    r"C:\Users\kuksh\Downloads\XGOCAELS_avatar_medium_square.jpg",
    r"C:\Users\kuksh\Downloads\x1080.jpg"
]

# Read the images
images = [keras_ocr.tools.read(img) for img in image_paths]

plt.figure(figsize=(10,20))
plt.imshow(images[0])
plt.show()
plt.figure(figsize=(10,20))
plt.imshow(images[1])
plt.show()

# Recognize text in the images
prediction_groups = pipeline.recognize(images)

# Loop through the recognized text and print it
for i, predictions in enumerate(prediction_groups):
    print(f"Predictions for image {i + 1}:")
    for prediction in predictions:
        print(prediction[0])

# Note: prediction[0] contains the recognized text, and prediction[1] contains the confidence score.
