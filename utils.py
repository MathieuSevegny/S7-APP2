import os
import numpy as np

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from helpers import analysis, dataset
from helpers.classifier import Classifier
from helpers.classifier import NeuralNetworkClassifier
from helpers.classifier import BayesClassifier
def one_hot_for_labels(labels:np.ndarray, unique_labels:list) -> np.ndarray:
    number_of_classes = len(unique_labels)
    labels_one_hot = np.zeros((labels.data.shape[0], number_of_classes))
    label_name_to_index = {label: index for index, label in enumerate(unique_labels)}

    for i, label in enumerate(labels):
        label_index = label_name_to_index[label]
        labels_one_hot[i, label_index] = 1
    
    return labels_one_hot



def get_impact_each_features_pred(classifier: Classifier, features:np.ndarray, labels:np.ndarray, unique_labels:list, feature_names: list):
    baseline_predictions = classifier.predict(features)
    if isinstance(classifier, NeuralNetworkClassifier) or isinstance(classifier, BayesClassifier):
        # For neural networks, we need to convert numeric labels to string label
        baseline_predictions = np.array([unique_labels[pred] for pred in baseline_predictions])
    error_rate, _ = analysis.compute_error_rate(labels, baseline_predictions)
    
    for i in range(features.shape[-1]):
        features_copy = features.copy()
        # shuffle the values of the feature
        np.random.shuffle(features_copy[:, i])
        predictions = classifier.predict(features_copy)
        if isinstance(classifier, NeuralNetworkClassifier) or isinstance(classifier, BayesClassifier):
            # For neural networks, we need to convert numeric labels to string label
            predictions = np.array([unique_labels[pred] for pred in predictions])
        error_rate_i, _ = analysis.compute_error_rate(labels, predictions)
        impact = error_rate_i - error_rate  # Calculate the increase in error rate when the feature is removed
        print(f"Impact of feature {i} ({feature_names[i]}): {impact:.4f}")