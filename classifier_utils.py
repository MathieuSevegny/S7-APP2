import copy
import os

# Must be call before any other TensorFlow/Keras import
# Suppress oneDNN custom operations info
# Suppress INFO and WARNING messages from TF (0=all, 1=no INFO, 2=no INFO/WARN, 3=no error)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
from sklearn.model_selection import train_test_split

from helpers import analysis, classifier, dataset


def get_impact_each_features_pred(cls: classifier.Classifier, features:np.ndarray, labels:np.ndarray, unique_labels:list, feature_names: list):
    baseline_predictions = cls.predict(features)
    if isinstance(cls, classifier.NeuralNetworkClassifier) or isinstance(cls, classifier.BayesClassifier):
        # For neural networks, we need to convert numeric labels to string label
        baseline_predictions = np.array([unique_labels[pred] for pred in baseline_predictions])
    error_rate, _ = analysis.compute_error_rate(labels, baseline_predictions)
    
    for i in range(features.shape[-1]):
        features_copy = features.copy()
        # shuffle the values of the feature
        np.random.shuffle(features_copy[:, i])
        predictions = cls.predict(features_copy)
        if isinstance(cls, classifier.NeuralNetworkClassifier) or isinstance(cls, classifier.BayesClassifier):
            # For neural networks, we need to convert numeric labels to string label
            predictions = np.array([unique_labels[pred] for pred in predictions])
        error_rate_i, _ = analysis.compute_error_rate(labels, predictions)
        impact = error_rate_i - error_rate  # Calculate the increase in error rate when the feature is removed
        print(f"Impact of feature {i} ({feature_names[i]}): {impact:.4f}")

def get_best_parameters_knn(representation: dataset.Representation) -> dict:
    X_train, X_test, y_train, y_test = train_test_split(
        representation.data, representation.labels, test_size=0.2, random_state=42
    )
    
    rep_train = copy.copy(representation)
    rep_train.data = X_train
    rep_train.labels = y_train

    valeurs_k = [1, 3, 5, 7, 9, 11]
    valeurs_representants = [None, 5, 10, 15, 20] 
    
    meilleure_erreur = float('inf')
    meilleurs_params = {}

    print("Évaluation des combinaisons en cours...")
    for k in valeurs_k:
        for rep in valeurs_representants:
            use_km = rep is not None
            n_rep = rep if use_km else 1
    
            if use_km and n_rep < k:
                continue 
            knn = classifier.KNNClassifier(n_neighbors=k, use_kmeans=use_km, n_representatives=n_rep)
            knn.fit(rep_train)
            predictions = knn.predict(X_test)
            erreur, _ = analysis.compute_error_rate(y_test, predictions)
            
         
            if use_km:
                print(f"Test : K={k}, K-Means=Oui, Représentants={n_rep} --> Erreur : {erreur * 100:.2f}%")
            else:
                print(f"Test : K={k}, K-Means=Non Erreur : {erreur * 100:.2f}%")
            
            # Mise à jour du champion
            if erreur < meilleure_erreur:
                meilleure_erreur = erreur
                meilleurs_params = {'k': k, 'use_kmeans': use_km, 'n_representatives': n_rep}

    print(f"\nMeilleurs paramètres trouvés : K={meilleurs_params['k']}, K-Means={meilleurs_params['use_kmeans']}, Représentants={meilleurs_params['n_representatives']}")
    print(f"   Avec un taux d'erreur de validation de : {meilleure_erreur * 100:.2f}%\n")
    
    return meilleurs_params