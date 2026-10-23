# AgriCare AI: Plant Disease Detection

A deep learning project that looks at a photo of a plant leaf and predicts the plant type and whether it is healthy or affected by one of 25 diseases (38 classes in total). Built with TensorFlow/Keras, trained on Google Colab, and served through a Streamlit web app.

> **Live demo:** [https://agricare-ai-plant-disease-detection-app.streamlit.app/]


## Problem

Plant diseases reduce crop yield, and farmers often cannot reach an expert quickly. This project builds an image classifier that gives a first estimate of the plant and its health status from a single leaf photo, together with a confidence score.

**Task:** multiclass image classification (38 classes).

## Dataset

- **PlantVillage** (color images), downloaded from Kaggle: `abdallahalidev/plantvillage-dataset`
- About 54,305 images in 38 classes (14 plant species, healthy and diseased leaves). Class folders are named `Plant___Disease`.
- The TensorFlow Datasets (TFDS) version could not be used because its download returned HTTP 403 from the source host, so the Kaggle copy was used instead.
- The classes are imbalanced, so accuracy alone is not enough. Per-class precision, recall and F1-score were checked as well.

The dataset is not included in this repository because of its size.

## Method

1. **Split:** 80% train, 10% validation, 10% test, using `train_test_split` twice with `random_state=42` and `stratify` on the label.
2. **Input pipeline (`tf.data`):** read file, decode, resize to 128x128, batch size 32, shuffle (train only), prefetch. Normalization is done inside the model.
3. **CNN from scratch:** three Conv2D + MaxPooling blocks (32, 64, 128 filters), Flatten, Dense 128, Softmax 38.
4. **Reducing overfitting:** data augmentation (random flip, rotation, zoom, contrast), Dropout (0.5) and early stopping.
5. **Transfer learning:** MobileNetV2 pre-trained on ImageNet.
   - Step 1: frozen base with a new head (Global Average Pooling, Dropout 0.3, Dense 38).
   - Step 2: fine-tuning of the last 30 layers with a learning rate of 1e-5.
6. **Evaluation:** validation set for model selection and error analysis, test set used once for the final result.

## Results

| Model | Validation accuracy | Validation loss |
|---|---|---|
| CNN from scratch (5 epochs) | 92.85% | 0.251 |
| CNN + augmentation + dropout (15 epochs) | 96.17% | 0.114 |
| MobileNetV2, frozen base | 91.93% | 0.255 |
| **MobileNetV2, fine-tuned (final model)** | **96.59%** | **0.108** |

**Final test set result (used once):** accuracy **96.93%**, loss **0.098**, macro F1 [add from classification report].

Notes:
- With a frozen base, MobileNetV2 was weaker than the CNN trained from scratch. Fine-tuning the last 30 layers improved it by about 2.9 points and made it slightly better than the CNN.
- The gap between the two best models is small (about 23 validation images), so the choice is not clear-cut. MobileNetV2 was selected because it is also smaller (about 2.3M parameters versus about 4.3M) and easier to deploy.

## Error analysis

The model was analysed on the validation set with a classification report, a confusion matrix and the most common confusions.

- Weakest classes: [add the 3 lowest-F1 classes and their F1]
- Most common confusions: [add 2 or 3 examples, e.g. class A predicted as class B]

## Limitations

- **Laboratory-style data.** PlantVillage images show a single leaf on a plain background. Performance on real field photos is not guaranteed.
- **Failure case.** A real close-up photo of a green leaf with a blurred background was classified as "Strawberry: Leaf scorch" with 80.8% confidence (see screenshot). Softmax confidence is not a reliable measure of correctness for images that look different from the training data.
- **Closed set.** The model only knows the 38 classes. It will return one of them for any image, including non-leaf images.
- **Possible optimistic scores.** PlantVillage often has several photos of the same leaf. A random split may place some of them in both training and test sets, so the test accuracy may be higher than real-world accuracy.
- **Not a diagnosis.** The output is an estimate and not a replacement for an agricultural expert.

## Project structure

```
agricare-ai/
├── .streamlit/
│   └── config.toml
├── data/
│   └── README.md            # dataset is not included
├── model/
│   ├── final_model.keras    # fine-tuned MobileNetV2
│   └── class_names.json
├── notebook/
│   └── AgriCare_AI.ipynb    # full training workflow
├── screenshots/
├── app.py                   # Streamlit app
├── requirements.txt
├── README.md
└── .gitignore
```

## How to run

```bash
# 1. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # Mac/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the app
streamlit run app.py
```

The app opens at `http://localhost:8501`. Upload a leaf photo (or use the camera) to see the predicted class, confidence and the top 3 predictions. Results below 60% confidence are shown as uncertain.

> **Important:** use the same TensorFlow and Keras versions as in `requirements.txt` (the versions used for training). A different version can fail to load the saved model. On Windows, keep the project path short, because TensorFlow has very long file names that can exceed the Windows path limit during installation.

To retrain, open `notebook/AgriCare_AI.ipynb` in Google Colab with a GPU runtime and download the dataset from Kaggle.

## Tech stack

Python, TensorFlow/Keras, MobileNetV2, NumPy, pandas, scikit-learn, matplotlib, Pillow, Streamlit, Google Colab

## Future work

- Train with field-style images (for example PlantDoc or self-collected photos) and background augmentation.
- Add out-of-distribution detection so the app can reject non-leaf or unfamiliar images.
- Use a group-aware split to remove the possible same-leaf leakage.
- Add treatment suggestions reviewed by an agricultural expert.