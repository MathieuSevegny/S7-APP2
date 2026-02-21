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



def get_impact_each_features_pred(classifier: Classifier, features:np.ndarray, labels:np.ndarray, unique_labels:list, feature_names: list) -> np.ndarray:
    impact = np.zeros(features.shape[-1])
    nom_caracteristique = []
    baseline_predictions = classifier.predict(features)
    if isinstance(classifier, NeuralNetworkClassifier) or isinstance(classifier, BayesClassifier):
        # For neural networks, we need to convert numeric labels to string label
        baseline_predictions = np.array([unique_labels[pred] for pred in baseline_predictions])
    error_rate, _ = analysis.compute_error_rate(labels, baseline_predictions)
    
    for i in range(features.shape[-1]):
        features_copy = features.copy()
        features_copy[:, i] = 0  # Set the i-th feature to zero
        predictions = classifier.predict(features_copy)
        if isinstance(classifier, NeuralNetworkClassifier) or isinstance(classifier, BayesClassifier):
            # For neural networks, we need to convert numeric labels to string label
            predictions = np.array([unique_labels[pred] for pred in predictions])
        error_rate_i, _ = analysis.compute_error_rate(labels, predictions)
        print(f"Error rate with feature {i} removed: {error_rate_i * 100:.2f}%")
        impact[i] = error_rate_i - error_rate  # Calculate the increase in error rate when the feature is removed
        nom_caracteristique.append(feature_names[i] if i < len(feature_names) else f"Caractéristique_{i}")

    print(f"Impact de chaque caractéristique sur la performance :")
    for nom, val in zip(nom_caracteristique, impact):
        print(f"  - {nom} : {val:.4f}")
    
    print(f"Taux d'erreur de base : {error_rate * 100:.2f}%")
    
    # On retourne le tableau COMPLET et la liste des noms
    return impact, nom_caracteristique