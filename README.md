# Email Spam Classification Project

## Overview
This project aims to classify emails as spam or not-spam using two machine learning methods: k-Nearest Neighbors (k-NN) and Multi-Layer Perceptron (MLP). The dataset used contains 4601 examples with 57 attributes each.

## Methods
1. **k-NN Classifier**:
   - **k**: 3
   - **Distance Metric**: Euclidean distance

2. **MLP Classifier**:
   - **Hidden Layers**: Two layers (first layer with 10 neurons, second layer with 5 neurons)
   - **Activation Function**: Logistic (sigmoid)

## How to Run
1. Ensure you have Python and scikit-learn installed.
2. Run the `main.py` script to execute the classifiers and generate results.

## Results
- **Accuracy, Precision, Recall, and F1-Score** for both classifiers will be reported in the output.
- Confusion matrices will be generated and included in the report.

## Code Structure
- **NN Class**: Implements the k-NN classifier.
- **load_data**: Loads data from a CSV file.
- **preprocess**: Normalizes the features of the dataset.
- **train_mlp_model**: Trains the MLP classifier using scikit-learn.
- **evaluate**: Computes evaluation metrics.
- **main**: Main function to run the classifiers and print results.


