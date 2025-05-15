import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras import layers, models
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.densenet import preprocess_input

# Set dataset paths
train_data_dir = r"/content/drive/MyDrive/embryo_data_tvt/Train"
validation_data_dir = r"/content/drive/MyDrive/embryo_data_tvt/Val"

# Improved Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],  # Brightness Adjustment
    channel_shift_range=30.0,  # Color Augmentation
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

# Load Dataset from Directory
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Load Pre-trained DenseNet201 Model
base_model = DenseNet201(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze the base model for now

#  Custom Classifier on Top
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),  # Helps stabilize training
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.3),  # Increased Dropout for Regularization
    layers.Dense(len(train_generator.class_indices), activation='softmax')
])

#  Compile the Model
model.compile(
    optimizer=Adam(learning_rate=0.0005),  # Slightly Increased LR for Better Convergence
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

#  Train the Model (Initial Training - Feature Extraction)
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=20,  # Start with 20 epochs
    validation_data=validation_generator,
    validation_steps=len(validation_generator)
)

#  Unfreeze Some Layers for Fine-Tuning
base_model.trainable = True  # Unfreeze the whole model
for layer in base_model.layers[:350]:  # Keep first 350 layers frozen
    layer.trainable = False

#  Recompile with Lower Learning Rate for Fine-Tuning
model.compile(
    optimizer=Adam(learning_rate=0.0001),  # Lower LR for fine-tuning
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Fine-Tune the Model
history_finetune = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=20,  # Another 20 epochs for fine-tuning
    validation_data=validation_generator,
    validation_steps=len(validation_generator)
)

# Final Model Evaluation
loss, acc = model.evaluate(validation_generator)
print(f"\nFinal Validation Accuracy: {acc * 100:.2f}%")
print(f"Final Validation Loss: {loss:.4f}")

# Save the Final Model
model.save('embryo_classification_model_finetuned.h5')

#  **Image Prediction Function**
def load_and_preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)
    return img_array

# **Prediction Function**
def load_and_predict_image(image_path):
    img_array = load_and_preprocess_image(image_path)
    predictions = model.predict(img_array)

    predicted_class_index = np.argmax(predictions, axis=1)[0]

    # Adjust class labels based on your dataset
    class_labels = ['8Cell_A', '8Cell_B','8Cell_C', 'Blastocyst_A', 'Blastocyst_B', 'Blastocyst_C', 'Morula_A', 'Morula_B', 'Morula_C', ]

    predicted_class_name = class_labels[predicted_class_index]
    class_name, subcategory = predicted_class_name.split('_')

    confidence = predictions[0][predicted_class_index]

    return class_name, subcategory, confidence

# **Example Usage**
image_path = r"/content/drive/MyDrive/embryo_data_tvt/Test/Blastocyst_A/aug_0_9246.jpeg"
class_name, subcategory, confidence = load_and_predict_image(image_path)

print(f"\nPredicted Class: {class_name}")
print(f"Subcategory: {subcategory}")
print(f"Confidence Score: {confidence * 100:.2f}%")

image_path = r"/content/drive/MyDrive/embryo_data_tvt/Test/Morula_A/aug_0_9157.jpeg"
class_name, subcategory, confidence = load_and_predict_image(image_path)
print(f"\nPredicted Class: {class_name}")
print(f"Subcategory: {subcategory}")
print(f"Confidence Score: {confidence * 100:.2f}%")