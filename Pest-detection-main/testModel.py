import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os 

# Load the trained model
model = load_model('plant_disease_model.h5')  # Load your trained model file

# Define image size (must be the same size used during training)
img_size = (224, 224)

# Define the path to your test dataset directory
test_dataset_directory = "D:/Titan/Dataset/TrainNDtest/test"  # Replace with your test dataset path

# Create a function to predict the class of an image
def predict_class(image_path):
    img = image.load_img(image_path, target_size=img_size)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize the pixel values
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    return predicted_class

# Loop through the images in the test dataset and make predictions
for class_name in os.listdir(test_dataset_directory):
    class_path = os.path.join(test_dataset_directory, class_name)
    if os.path.isdir(class_path):
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            predicted_class = predict_class(image_path)
            print(f"Image: {image_name}, Predicted Class: {predicted_class}, Predicted Class Name: {class_name}")
