import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)



class NN:
    def __init__(self, trainingFeatures, trainingLabels):
        self.trainingFeatures = trainingFeatures
        self.trainingLabels = trainingLabels

    def predict(self, features, k):
        predictions = []
        for test_point in features:
            distances = []
            for train_point in self.trainingFeatures:
                distance = np.linalg.norm(test_point - train_point)  # Euclidean distance
                distances.append(distance)
            k_nearest_labels = [self.trainingLabels[i] for i in np.argsort(distances)[:k]]
            prediction = max(set(k_nearest_labels), key=k_nearest_labels.count)
            predictions.append(prediction)
        return predictions


def load_data(file_path):
    data = []
    labels = []
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            data.append(row[:-1])
            labels.append(int(row[-1]))
    return np.array(data, dtype=float), np.array(labels)


def preprocess(features):
    means = np.mean(features, axis=0)
    stds = np.std(features, axis=0)
    normalized_features = (features - means) / stds
    return normalized_features


def train_mlp_model(features, labels):
    model = MLPClassifier(hidden_layer_sizes=(10, 5), activation='logistic')
    model.fit(features, labels)
    return model


def evaluate(labels, predictions):
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    return accuracy, precision, recall, f1


def main():

    # Load data from CSV file and split into train and test sets
    features, labels = load_data('spambase.csv')
    print("Processing ... ")
    features = preprocess(features)
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.3, random_state=42)

    # Train a k-NN model and make predictions
    model_nn = NN(X_train, y_train)
    predictions_nn = model_nn.predict(X_test, k=3)
    accuracy_nn, precision_nn, recall_nn, f1_nn = evaluate(y_test, predictions_nn)

    # Print k-NN results
    print("**** k-NN Results ****")
    print("Accuracy: ", accuracy_nn)
    print("Precision: ", precision_nn)
    print("Recall: ", recall_nn)
    print("F1: ", f1_nn)

    # Train an MLP model and make predictions
    model_mlp = train_mlp_model(X_train, y_train)
    predictions_mlp = model_mlp.predict(X_test)
    accuracy_mlp, precision_mlp, recall_mlp, f1_mlp = evaluate(y_test, predictions_mlp)
    print("*************************");
    # Print MLP results
    print("**** MLP Results ****")
    print("Accuracy: ", accuracy_mlp)
    print("Precision: ", precision_mlp)
    print("Recall: ", recall_mlp)
    print("F1: ", f1_mlp)


if __name__ == "__main__":
    main()
