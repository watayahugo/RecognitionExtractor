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

> **_NOTE:_** Depending on the training data available to the model, it may be beneficial to change a few of the parameters located in the `config.cfg` file **prior** to training the model and running the extraction script. (Learning rate, Prediction leniency, Validation split, Epochs, etc.) See **Training / Extraction Parameters** below for more details.

1. **Prepare Data:**

   - Place images of the character in the `training_data/positive` folder.
   - Place images without the character in the `training_data/negative` folder.
   - Place video (mkv, mp4, mov) in same directory as both `model.py` and `extractor.py` **and name it `vid.mkv`**
     > Name of video can be changed in the `config.cfg` file



2. **Train Model:**

   ```bash
   python model.py
   ```

   A character recognition model will then be saved in the models/ directory. This is what is used upon execution of `extractor.py`

4. **Extract Frames:**
   ```bash
   python extractor.py
   ```


> **_CAUTION:_** As the extractor script saves **uncompressed** images directly from the video file, depending on the length and quality of the video, file sizes may get **very large**.


## Training / Extraction Parameters

The `config.cfg` file contains a few easily changeable parameters to deal with differing data sets:

| Parameter           | Default Value | Description                                                                                             | Note                                                                                                                                                  |
|---------------------|---------------|---------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|
| BatchSize           | 2             | Determines how many samples (images in positive/negative data) are trained at once.                     | Significant degradation of the model is likely with larger batch sizes. If experiencing performance issues, lower this value.                         |
| LearningRate        | 0.0001        | Regulation of the frequency at which the model's weights are changed during the training.               | Setting learning rate too low can cause loss function to not improve and setting it too high can cause it to diverge. When lowering/raising this setting, make sure it is done in small increments (~±0.0015) |
| Epochs              | 20            | The total number of times the learning algorithm will work through the entire training dataset.         | Too many epochs can lead to overfitting of the training dataset, while too few may result in an underfit model.                                      |
| ValidationSplit     | 0.2           | The proportion of the training data to set aside for validating the model during training.              | Setting this value too high may result in using too little data for training. Setting it too low may result in overfitting.                         |
| FrameInterval       | 2             | How many frames you want the extractor to increment by.                                                | A default value of 2 means the extractor is going to analyze every other frame in the video. A value of 1 means the extractor will analyze every single frame. |
| PredictionLeniency  | 0.5           | The threshold for classifying a prediction as positive or negative.                                    | The lower this setting is, the more lenient the threshold is during frame analyzation. Setting this value too low (i.e. 0.2) may result in too many false positives, while setting it too high (i.e. 0.8) may result in too many false negatives.                |
