import tensorflow as tf
from keras.api._v2.keras import layers, models
from keras.api._v2.keras.preprocessing.image import ImageDataGenerator
from keras.api._v2.keras.optimizers import Adam
from keras.api._v2.keras.regularizers import l2
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
train_datagen = ImageDataGenerator(
    validation_split=0.2,
    rescale=1.0 / 255,
    rotation_range=10,  # Reduced rotation range
    width_shift_range=0.1,  # Reduced shift range
    height_shift_range=0.1,  # Reduced shift range
    shear_range=0.1,  # Reduced shear range
    zoom_range=0.1,  # Reduced zoom range
    horizontal_flip=True,  # Kept horizontal flip
    fill_mode="nearest",
)
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
dense_layer = layers.Dense(1, activation="sigmoid", kernel_regularizer=l2(0.01))  # noqa
model = models.Sequential([base_model, pooling_layer, dense_layer])

# Set custom learning rate
custom_learning_rate = 0.0001

# Create an instance of the Adam optimizer with the custom learning rate
optimizer = Adam(learning_rate=custom_learning_rate)

# Compile the model
model.compile(
    optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"]
)  # noqa

# Train the model
model.fit(train_generator, validation_data=validation_generator, epochs=15)

model.save(model_path)
