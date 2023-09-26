# RecognitionExtractor

## About

Specialized implementation of computer vision and deep learning techniques aimed at extracting specific frames from a video where a designated character / person appears. Utilizing TensorFlow, this project employs a fine-tuned neural network model to analyze each frame of the input video and determine the presence of the target character. The model is trained using a pre-trained MobileNetV2 as the base model, with additional pooling and dense layers. The training process involves data augmentation, normalization, and binary cross-entropy loss. The extractor script analyzes each frame of the input video, predicts the presence of the character, and saves frames where the character is detected.

## Installation

1. Clone the repository

```bash
git clone https://github.com/watayahugo/RecognitionExtractor.git
```

2. Create the Conda environment

```bash
conda env create -f environment.yml
```

> **_NOTE:_** Ensure that Anaconda is installed and set up.

3. Check if environment was successfully created

```bash
conda env list
```

4. Activate the new environment

```bash
conda activate vidextract
```

## Usage

1. **Prepare Data:**

   - Place images of the character in the `positive` folder.
   - Place images without the character in the `negative` folder.

2. **Train Model:**

   ```bash
   python model.py
   ```

````

3. **Extract Frames:**
   ```bash
   python extractor.py
   ```
````
