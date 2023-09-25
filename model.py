import tensorflow as tf
from keras.api._v2.keras import layers, models
from keras.api._v2.keras.preprocessing.image import ImageDataGenerator
import os

# noqa: E501

# Define the path to folders
pos_folder = "training_data/positive"
neg_folder = "training_data/negative"
models_folder = "models"
model_path = os.path.join(models_folder, "character_recognition_model")

# Check if the folders exist
pos_exists = os.path.exists(pos_folder)
neg_exists = os.path.exists(neg_folder)
models_exists = os.path.exists(models_folder)

# Create the positive and negative folders if they don't exist
os.makedirs(pos_folder, exist_ok=True)
os.makedirs(neg_folder, exist_ok=True)
os.makedirs(models_folder, exist_ok=True)

# If either folder was created, print a message and exit the program
if not pos_exists or not neg_exists or not models_exists:
    print(
        "The folders have been created. "
        "Please add positive and negative data "
        "to the folders appropriately."
    )
    exit()

parent_dir = "training_data"


# Load and preprocess labeled data
train_datagen = ImageDataGenerator(validation_split=0.2, rescale=1.0 / 255)
train_generator = train_datagen.flow_from_directory(
    parent_dir,
    target_size=(224, 224),
    batch_size=2,
    class_mode="binary",
    subset="training",
)
validation_generator = train_datagen.flow_from_directory(
    parent_dir,
    target_size=(224, 224),
    batch_size=2,
    class_mode="binary",
    subset="validation",
)

# Define the model
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3), include_top=False, weights="imagenet"
)
base_model.trainable = False

pooling_layer = layers.GlobalAveragePooling2D()
dense_layer = layers.Dense(1, activation="sigmoid")
model = models.Sequential([base_model, pooling_layer, dense_layer])

# Compile the model
model.compile(
    optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
)  # noqa

# Train the model
model.fit(train_generator, validation_data=validation_generator, epochs=15)

model.save(model_path)
