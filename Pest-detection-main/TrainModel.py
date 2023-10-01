import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import os


# Define the path to your dataset directory
dataset_directory = "D:/Josika/Metaverse/Dataset" # Replace with your dataset path

# Define image size and batch size
img_size = (224, 224)
batch_size = 32

# Define the number of classes (plant diseases)
num_classes = len(os.listdir(dataset_directory))

# Create data generators for training and testing
data_generator = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # Split the dataset into 80% training and 20% validation
)

train_generator = data_generator.flow_from_directory(
    dataset_directory,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,
    subset='training'  # Use the training split
)

validation_generator = data_generator.flow_from_directory(
    dataset_directory,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False,
    subset='validation'  # Use the validation split
)

# Load MobileNetV2 model with pre-trained weights (include_top=False for custom output layer)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom layers for classification
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(1024, activation='relu')(x)
output = Dense(num_classes, activation='softmax')(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=output)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss=CategoricalCrossentropy(),
              metrics=[CategoricalAccuracy()])

# Train the model
num_epochs = 10  # Adjust as needed
history = model.fit(
    train_generator,
    epochs=num_epochs,
    validation_data=validation_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_steps=validation_generator.samples // batch_size
)

# Save the trained model for future use
model.save('plant_disease_model.h5')

# Evaluate the model on the test set
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

test_generator = test_datagen.flow_from_directory(
    "D:/Titan/Dataset/TrainNDtest/test",  # Replace with the path to your test dataset
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False  # No need to shuffle for evaluation
)

# Perform model inference on the test dataset
y_true = test_generator.classes  # True labels
y_pred = model.predict(test_generator)  # Predicted labels

# Compute the confusion matrix
confusion = confusion_matrix(y_true, y_pred.argmax(axis=1))

# Display the confusion matrix
print("Confusion Matrix:")
print(confusion)

# Generate a classification report (includes precision, recall, F1-score, and more)
report = classification_report(y_true, y_pred.argmax(axis=1))
print("\nClassification Report:")
print(report)