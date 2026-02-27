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
    valeurs_representants = [None,2,3,4,5,6,7,8,9,10] 
    
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





def recherche_hyperparametres_rna(representation, liste_couches, liste_neurones, liste_activations):
    """
    Fonction indépendante qui teste plusieurs architectures de RNA (Grid Search).
    Retourne un dictionnaire avec les meilleurs hyperparamètres et sauvegarde l'historique en Excel.
    """
    # couches = [4]
    # neurones = [4,5,6,7, 8, 16]
    # activations = ["relu", "tanh", "sigmoid"]
    # best_params = classifier_utils.recherche_hyperparametres_rna(representation, couches, neurones, activations)
    import itertools
    import pandas as pd

    meilleur_taux_erreur = float('inf')
    meilleurs_params = {}
    
    historique_resultats = []
    
    for n_hidden, n_neurons, activation in itertools.product(liste_couches, liste_neurones, liste_activations):
        print(f"Test -> Couches: {n_hidden} | Neurones: {n_neurons} | Activation: '{activation}'")

        nn_test = classifier.NeuralNetworkClassifier(
            input_dim=representation.dim,
            output_dim=len(representation.unique_labels),
            n_hidden=n_hidden,
            n_neurons=n_neurons,
            activation=activation,
            lr=0.001,
            n_epochs=30, 
            batch_size=16
        )
        
        nn_test.fit(representation)
        data_preprocessed = nn_test.preprocess_data(representation.data)
        predictions = nn_test.predict(data_preprocessed)
        predictions_labels = np.array([representation.unique_labels[i] for i in predictions])
        
        error_rate, _ = analysis.compute_error_rate(representation.labels, predictions_labels)
        print(f"   => Taux d'erreur : {error_rate * 100:.2f}%\n")
        
        historique_resultats.append({
            "Nb de couches cachées": n_hidden,
            "Neurones par couche": n_neurons,
            "Fonction d'activation": activation,
            "Taux d'erreur (%)": round(error_rate * 100, 2)
        })
        
        if error_rate < meilleur_taux_erreur:
            meilleur_taux_erreur = error_rate
            meilleurs_params = {
                "n_hidden": n_hidden,
                "n_neurons": n_neurons,
                "activation": activation
            }


    df_resultats = pd.DataFrame(historique_resultats)
    df_resultats = df_resultats.sort_values(by="Taux d'erreur (%)") 
    
    nom_fichier = "historique_grid_search_rna.xlsx"
    df_resultats.to_excel(nom_fichier, index=False)
    
    print(f"Meilleurs paramètres trouvés : {meilleurs_params} avec {meilleur_taux_erreur * 100:.2f}% d'erreur")
    
    return meilleurs_params