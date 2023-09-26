# RecognitionExtractor

## Installation
```
git clone https://github.com/watayahugo/RecognitionExtractor.git
```

## Usage
1. **Prepare Data:**
   - Place images of the character in the `positive` folder.
   - Place images without the character in the `negative` folder.

2. **Train Model:**
   ```bash
   python train_model.py
   ```

## Model Training
The model is trained using a pre-trained MobileNetV2 as the base model, with additional pooling and dense layers. The training process involves data augmentation, normalization, and binary cross-entropy loss.

## Frame Analysis
The script analyzes each frame of the input video, predicts the presence of the character, and saves frames where the character is detected. It efficiently skips similar frames to reduce redundancy.
