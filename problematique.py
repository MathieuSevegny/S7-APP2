
import os
import pathlib

import numpy

import utils


# Must be call before any other TensorFlow/Keras import
# Suppress oneDNN custom operations info
# Suppress INFO and WARNING messages from TF (0=all, 1=no INFO, 2=no INFO/WARN, 3=no error)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from matplotlib import pyplot as plt
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from helpers import analysis, classifier, viz
import helpers.dataset as dataset
from features import *
from sklearn.model_selection import train_test_split
import copy


def etape1_representation(images: dataset.ImageDataset, show_plots: bool = True) -> np.ndarray:
    print("--- Étape 1 : Représentation ---")
    noise_feature = calculate_noise(images).reshape(-1, 1)
    ratio_vertical_horizontal = calculate_ratio_vertical_horizontal(images).reshape(-1, 1)
    number_lab_peaks = calculate_lab_peaks(images).reshape(-1, 3)
    ecart_type = calculate_std_dev(images).reshape(-1, 3)
    contrast = calculate_contrast(images).reshape(-1, 1)
    most_common_color_top_left = calculate_most_common_color_in_top_left_corner(images).reshape(-1, 3)
    ratio_high_low_freq = calculate_ratio_high_low_frequency(images).reshape(-1, 1)
    ratio_symmetry = calculate_ratio_symmetry(images).reshape(-1, 1)
    
    
    features = np.hstack((noise_feature, number_lab_peaks[:,[2]], ecart_type[:,[0]], ratio_vertical_horizontal))
    feature_names = [
            "Bruit",
            "Pics Lab(b)",
            "Écart R",
            "Ratio Vert/Horiz",
        ]
    
    assert len(feature_names) == features.shape[1], f"Le nombre de noms de features doit correspondre au nombre de features extraites. Actuellement, {len(feature_names)} noms pour {features.shape[1]} features."
    
    print(f"Features extraites. Shape: {features.shape}\n")
    
    if show_plots:

        spickes_representation = dataset.Representation(data=number_lab_peaks, labels=images.labels)
        viz.plot_features_distribution(spickes_representation, 
                                   title="Distribution du nombre de pics dans les canaux Lab", 
                                   xlabel="Nombre de pics dans le canal Lab", 
                                   ylabel="Nombre d'images",
                                   n_bins=32,
                                   features_names=["Nombre de pics dans le canal L Lab", "Nombre de pics dans le canal A Lab", "Nombre de pics dans le canal B Lab"])
        ecart_representation = dataset.Representation(data=ecart_type, labels=images.labels)
        viz.plot_features_distribution(ecart_representation, 
                                   title="Distribution des écarts-types", 
                                   xlabel="Valeur de l'écart-type",
                                   ylabel="Nombre d'images",
                                   n_bins=32,
                                   features_names=["Écart-type R", "Écart-type G", "Écart-type B"])
        noise_representation = dataset.Representation(data=noise_feature, labels=images.labels)
        viz.plot_features_distribution(noise_representation, 
                                   title="Distribution du bruit", 
                                   xlabel="Bruit", 
                                   ylabel="Ratio haut/bas fréquence",
                                   n_bins=32,
                                   features_names=["Bruit"])
        
        ratio_vertical_horizontal_representation = dataset.Representation(data=ratio_vertical_horizontal, labels=images.labels)
        viz.plot_features_distribution(ratio_vertical_horizontal_representation, 
                                   title="Distribution du ratio vertical/horizontal", 
                                   xlabel="Ratio vertical/horizontal", 
                                   ylabel="Nombre d'images",
                                   n_bins=32,
                                   features_names=["Ratio vertical/horizontal"])
        
        contrast_representation = dataset.Representation(data=contrast, labels=images.labels)
        viz.plot_features_distribution(contrast_representation,
                                      title="Distribution du contraste", 
                                      xlabel="Contraste", 
                                      ylabel="Nombre d'images",
                                      n_bins=32,
                                      features_names=["Contraste"])
        most_common_color_representation = dataset.Representation(data=most_common_color_top_left, labels=images.labels)
        viz.plot_features_distribution(most_common_color_representation,
                                      title="Distribution de la couleur la plus fréquente dans le coin supérieur gauche", 
                                      xlabel="Valeur de la couleur la plus fréquente dans le coin supérieur gauche", 
                                      ylabel="Nombre d'images",
                                      n_bins=32,
                                      features_names=["Couleur R Top-Gauche", "Couleur G Top-Gauche", "Couleur B Top-Gauche"])
        ratio_high_low_freq_representation = dataset.Representation(data=ratio_high_low_freq, labels=images.labels)
        viz.plot_features_distribution(ratio_high_low_freq_representation,
                                      title="Distribution du ratio haut/bas fréquence", 
                                      xlabel="Ratio haut/bas fréquence", 
                                      ylabel="Nombre d'images",
                                      n_bins=32,
                                      features_names=["Ratio haut/bas fréquence"])
        
        ratio_symmetry_representation = dataset.Representation(data=ratio_symmetry, labels=images.labels)
        viz.plot_features_distribution(ratio_symmetry_representation,
                                        title="Distribution du ratio de symétrie", 
                                        xlabel="Ratio de symétrie", 
                                        ylabel="Nombre d'images",
                                        n_bins=32,
                                        features_names=["Ratio de symétrie"])
        
        subrepresentation = dataset.Representation(data=features[:, 0:3], labels=images.labels)
        viz.plot_data_distribution(subrepresentation,
                            title="Distribution des 3 premières features",
                              xlabel="Bruit",
                              ylabel="Couleur R",
                              zlabel="symétrie", isNormalized=True)                  
        plt.show(block=True)
    return features, feature_names

def etape2_pretraitement(features: np.ndarray, feature_names: list, labels: np.ndarray, show_plots: bool = True, use_pca : bool = True) -> dataset.Representation:
    print("--- Étape 2 : Prétraitement avec Normalisation et réduction de dimensionnalité (PCA)---")
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(features)
    if show_plots:
        correlations = numpy.corrcoef(normalized_features, rowvar=False)
        plt.figure(figsize=(8, 6))
        plt.imshow(correlations, cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar(label="Coefficient de corrélation (Pearson)")
        plt.xticks(ticks=numpy.arange(len(feature_names)), labels=feature_names, rotation=45, ha='right')
        plt.yticks(ticks=numpy.arange(len(feature_names)), labels=feature_names)
        plt.title("Matrice de corrélation AVANT PCA")
        plt.tight_layout()
        plt.show(block=True)
    
    if not use_pca:
        representation = dataset.Representation(data=normalized_features, labels=labels)
        return representation
    
    pca = PCA(n_components=min(10, normalized_features.shape[1]))
    pca_features = pca.fit_transform(normalized_features)
    
    print("\n--- Analyse en Composantes Principales (PCA) ---")
    print(f"Variance expliquée : {numpy.round(pca.explained_variance_ratio_ * 100, 2)}")
    print(f"Information totale conservée : {numpy.sum(pca.explained_variance_ratio_) * 100:.2f}%\n")
    
    if show_plots:
        correlations_apres = numpy.corrcoef(pca_features, rowvar=False)
        plt.figure(figsize=(8, 6))
        plt.imshow(correlations_apres, cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar(label="Coefficient de corrélation (Pearson)")
        plt.xticks(ticks=numpy.arange(len(feature_names)), labels=[f"Composante {i}" for i in range(len(feature_names))], rotation=45, ha='right')
        plt.yticks(ticks=numpy.arange(len(feature_names)), labels=[f"Composante {i}" for i in range(len(feature_names))])
        plt.title("Matrice de corrélation APRÈS PCA (Décorrélation)")
        plt.tight_layout()
        plt.show(block=True)
    representation = dataset.Representation(data=pca_features, labels=labels)
    return representation

def etape3_classificateur_bayesien(representation: dataset.Representation, feature_names: list, show_plots: bool = True):
    print("--- Étape 3 :Entraînement et évaluation du Classificateur Bayésien ---")
    bayes = classifier.BayesClassifier(density_function=analysis.GaussianPDF)
    bayes.fit(representation)

    predictions_indices = bayes.predict(representation.data)
    predictions_labels = representation.unique_labels[predictions_indices]
    error_rate, _ = analysis.compute_error_rate(representation.labels, predictions_labels)
    print(f"Taux d'erreur Bayésien : {error_rate * 100:.2f}%")
    
    if show_plots:
        viz.show_confusion_matrix(representation.labels, predictions_labels, representation.unique_labels, plot=True, title="Matrice de confusion du classificateur Bayésien")
    print("\n")
    
    utils.get_impact_each_features_pred(bayes, representation.data, representation.labels, representation.unique_labels, feature_names)
    return error_rate

def etape4_classificateur_knn(representation: dataset.Representation, feature_names: list, show_plots: bool = True):
    print("--- Étape 4 : Entraînement et évaluation du Classificateur k-moy, k-PPV ---")
    best_params = utils.get_best_parameters_knn(representation)
    knn = classifier.KNNClassifier(n_neighbors=best_params['k'], use_kmeans=best_params['use_kmeans'], n_representatives=best_params['n_representatives'])
    knn.fit(representation)
    predictions = knn.predict(representation.data)
    error_rate, _ = analysis.compute_error_rate(representation.labels, predictions)
    if show_plots:
        viz.show_confusion_matrix(representation.labels, predictions, representation.unique_labels, plot=True, title="Matrice de confusion du classificateur KNN")
    print("\n")
    utils.get_impact_each_features_pred(knn, representation.data, representation.labels, representation.unique_labels, feature_names)
    return error_rate




def etape5_classificateur_rna(representation: dataset.Representation, show_plots: bool = True, feature_names: list = None):
    """
    Étape 5 : Entraînement et évaluation du Réseau de Neurones Artificiels.
    """
    print("--- Étape 5 : Classificateur RNA ---")
    nn_classifier = classifier.NeuralNetworkClassifier(
        input_dim=representation.dim,
        output_dim=len(representation.unique_labels),
        n_hidden=3,
        n_neurons=8,
        lr=0.001,
        n_epochs=70,
        batch_size=16
    )
    nn_classifier.fit(representation)
    save_dir = pathlib.Path(__file__).parent / "saves"
    save_dir.mkdir(parents=True, exist_ok=True)
    nn_classifier.save(save_dir / "multimodal_classifier.keras")
    if show_plots:
        viz.plot_metric_history(nn_classifier.history)
    plt.savefig(save_dir / "training_history.png")

    data = nn_classifier.preprocess_data(representation.data)
    predictions = nn_classifier.predict(data)
    predictions_labels = np.array([representation.unique_labels[i] for i in predictions])

    error_rate, indexes_errors = analysis.compute_error_rate(representation.labels, predictions_labels)
    print(f"\n{len(indexes_errors)} erreurs de classification sur {len(predictions)} échantillons ({error_rate * 100:.2f}%).")
    
    if show_plots:
        viz.show_confusion_matrix(representation.labels, predictions_labels, representation.unique_labels, plot=True, title="Matrice de confusion du classificateur RNA")
    print("\n")
    utils.get_impact_each_features_pred(nn_classifier, representation.data, representation.labels, representation.unique_labels, feature_names)
    return error_rate

def etape6_discussion_et_justifications(resultats: dict, show_plots: bool = True):
    if show_plots:
        print("--- Étape 6 : Compilation des résultats, Discussion et justifications ---")
        print("Récapitulatif des performances :")
        for nom, erreur in resultats.items():
            print(f"- {nom} : {erreur * 100:.2f}% d'erreur")
        plt.show(block=True)
    
def problematique():
    images = dataset.ImageDataset("data/image_dataset/")
    SHOW_PLOTS = True
    features, feature_names = etape1_representation(images)
    representation = etape2_pretraitement(features, feature_names=feature_names, labels=images.labels,use_pca=True)
    err_bayes = etape3_classificateur_bayesien(representation,feature_names=feature_names)
    err_knn = etape4_classificateur_knn(representation, feature_names=feature_names)
    err_rna = etape5_classificateur_rna(representation, feature_names=feature_names)
    resultats = {
        "Bayésien": err_bayes,
        "K-PPV (KNN)": err_knn,
        "RNA": err_rna
    }
    etape6_discussion_et_justifications(resultats)

    if SHOW_PLOTS:
        print("\nAffichage des figures (Fermez les fenêtres pour terminer le script)...")
        plt.show(block=True)

if __name__ == "__main__":
    problematique()
