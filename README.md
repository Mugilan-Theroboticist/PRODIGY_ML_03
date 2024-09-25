# Cat and Dog Classification using SVM
## Project Overview
This project aims to classify images of cats and dogs using a Support Vector Machine (SVM). The SVM algorithm is a powerful supervised machine learning technique used for classification tasks. In this project, the dataset consists of images from the Kaggle Cats and Dogs Dataset, and the goal is to correctly classify the images as either "cat" or "dog."

# Dataset
The dataset contains two categories:

Cats: Images of cats
Dogs: Images of dogs
### The dataset used in this project is sourced from Kaggle, containing both training and testing sets. The images are resized and preprocessed before being used for training the model.
## Project Structure
├── dataset/
│   ├── cats/
│   ├── dogs/
├── models/
│   ├── svm_model.pkl
├── notebooks/
│   └── svm_classification.ipynb
├── README.md
└── requirements.txt
#### dataset/: Contains the image dataset, split into categories for training and testing.
#### models/: Stores the trained SVM model after fitting.
#### notebooks/: Jupyter notebook for the implementation and exploration of the SVM classification.
#### requirements.txt: List of dependencies required to run the project.

## Preprocessing
Before training the model, the following preprocessing steps are performed:

Resizing Images: Each image is resized to a fixed dimension of (150, 150, 3) to ensure uniformity.
Flattening: Images are flattened into a 1D array before being fed into the SVM.
Label Encoding: Labels (cat or dog) are encoded as 0 for cats and 1 for dogs.

## Model Training
The SVM classifier is trained using the following process:

Feature Extraction: The images are transformed into a flat feature vector.
SVM Model: A linear SVM classifier is applied using the Scikit-learn SVC class.
Hyperparameter Tuning: Parameters such as C and kernel can be adjusted for optimal performance.

## Evaluation
After training, the model is evaluated using:

Accuracy Score: Measures the percentage of correct predictions on the test set.
Confusion Matrix: Visual representation of the classifier performance in terms of true positives, true negatives, false positives, and false negatives.
# Conclusion
This project demonstrates the use of SVM for binary classification tasks, specifically for classifying images of cats and dogs. The SVM model shows good performance with accurate classification results.
