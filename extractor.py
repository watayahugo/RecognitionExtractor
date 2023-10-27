import cv2
import numpy as np
import os
import tensorflow as tf
import configparser

# Read configuration file
config = configparser.ConfigParser()
config.read("config.cfg")

# Retrieve parameters from configuration file
frame_interval = int(config["EXTRACT"]["FrameInterval"])
prediction_leniency = float(config["EXTRACT"]["PredictionLeniency"])
video_input = str(config["VIDEO"]["VideoName"])


def preprocess_frame(frame):
    # Resize the frame to the input size that the model expects
    resized_frame = cv2.resize(frame, (224, 224))

    # Convert the frame to RGB format
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

    # Normalize pixel values
    normalized_frame = rgb_frame / 255.0

    # Expand the dimensions to add the batch size
    batch_frame = np.expand_dims(normalized_frame, axis=0)

    return batch_frame


def predict_frame(frame, model):
    # Preprocess the input frame
    preprocessed_frame = preprocess_frame(frame)

    # Make a prediction using the model
    prediction = model.predict(preprocessed_frame)

    print(f"The current prediction ratio is {prediction}")

    return prediction


def extract_frames(video_path, output_folder, model):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Predict the class of the frame
        predicted_class = predict_frame(frame, model)

        # If the predicted class indicates the character is present,
        # save the frame
        if predicted_class > prediction_leniency:
            frame_path = os.path.join(output_folder, f"frame_{frame_count}.png")  # noqa
            cv2.imwrite(frame_path, frame)

        frame_count += frame_interval
    cap.release()


# Load the trained model
model_path = "models/character_recognition_model"
model = tf.keras.models.load_model(model_path)

# Call the extract_frames function with the video path,
# output folder, and the loaded model
video_path = video_input
output_folder = "frames"
extract_frames(video_path, output_folder, model)
