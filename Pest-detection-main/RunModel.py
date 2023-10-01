import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os 
import numpy as np

model = load_model('plant_disease_model.h5')

img_size = (224, 224)

def predict_class(image_path):
    img = image.load_img(image_path, target_size=img_size)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    return predicted_class

image_path = input("Enter the path of the image: ")

if not os.path.isfile(image_path):
    print("Error: File not found.")
else:
    predicted_class = predict_class(image_path)
    print(f"Predicted Class Index: {predicted_class}")