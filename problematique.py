
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



def problematique():
    images = dataset.ImageDataset("data/image_dataset/")
    
    noise_feature = calculate_noise(images).reshape(-1, 1)
    
    colors_top_left = calculate_most_common_color_in_top_left_corner(images).reshape(-1, 3)
    ratio_high_low = vertical_horizontal_ratio(images).reshape(-1, 1)
    symmetry = calculate_ratio_symmetry(images).reshape(-1, 1)
    number_lab_b_peaks = calculate_lab_b_peaks(images).reshape(-1, 1)
    ecart_type = calculate_std_dev(images).reshape(-1, 3)
    features = np.hstack((noise_feature, colors_top_left, symmetry, ratio_high_low, number_lab_b_peaks, ecart_type))
  
    print("Features shape:", features.shape)    

    # TODO Problématique: Générez une représentation des images appropriée
    # pour la classification comme dans le laboratoire 1.
    # -------------------------------------------------------------------------
    representation = dataset.Representation(data=features, labels=images.labels)
    # -------------------------------------------------------------------------

    # TODO: Problématique: Visualisez cette représentation
    # -------------------------------------------------------------------------
    # 
    # -------------------------------------------------------------------------
    if True:
        ecart_representation = dataset.Representation(data=ecart_type, labels=images.labels)
        viz.plot_features_distribution(ecart_representation, 
                                   title="Distribution des écarts-types", 
                                   xlabel="Écart-type R", 
                                   ylabel="Écart-type G",
                                   n_bins=32,
                                   features_names=["Écart-type R", "Écart-type G", "Écart-type B"])
        noise_representation = dataset.Representation(data=noise_feature, labels=images.labels)
        viz.plot_features_distribution(noise_representation, 
                                   title="Distribution du bruit", 
                                   xlabel="Bruit", 
                                   ylabel="Ratio haut/bas fréquence",
                                   n_bins=32,
                                   features_names=["Bruit"])

        ratio_representation = dataset.Representation(data=ratio_high_low, labels=images.labels)
        viz.plot_features_distribution(ratio_representation, 
                                   title="Distribution du ratio haut/bas fréquence", 
                                   xlabel="Ratio haut/bas fréquence", 
                                   ylabel="Bruit",
                                   n_bins=32,
                                   features_names=["Ratio haut/bas fréquence"])
        
        ratio_representation = dataset.Representation(data=colors_top_left, labels=images.labels)
        viz.plot_features_distribution(ratio_representation, 
                                   title="Distribution des couleurs du coin supérieur gauche", 
                                   xlabel="Couleur R", 
                                   ylabel="Couleur G",
                                   n_bins=32,
                                   features_names=["Couleur R", "Couleur G", "Couleur B"])
        
        ratio_representation = dataset.Representation(data=symmetry, labels=images.labels)
        viz.plot_features_distribution(ratio_representation, 
                                   title="Distribution du ratio de symétrie", 
                                   xlabel="Ratio de symétrie", 
                                   ylabel="Nombre d'images",
                                   n_bins=32,
                                   features_names=["Ratio de symétrie"])
        
        spickes_representation = dataset.Representation(data=number_lab_b_peaks, labels=images.labels)
        viz.plot_features_distribution(spickes_representation, 
                                   title="Distribution du nombre de pics dans le canal b du Lab", 
                                   xlabel="Nombre de pics dans le canal b du Lab", 
                                   ylabel="Nombre d'images",
                                   n_bins=32,
                                   features_names=["Nombre de pics dans le canal b du Lab"])
        
        subrepresentation = dataset.Representation(data=features[:, 0:3], labels=images.labels)
        viz.plot_data_distribution(subrepresentation,
                            title="Distribution basée sur le bruit",
                              xlabel="Bruit",
                              ylabel="Couleur R",
                              zlabel="symétrie", isNormalized=True)
                              
        plt.show()
    if True:
 
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(features)
        feature_names = [
            "Bruit", 
            "Coul R", "Coul G", "Coul B", 
            "Symétrie", 
            "Ratio Freq", 
            "Pics Lab(b)", 
            "Écart R", "Écart G", "Écart B"
        ]

        correlations = numpy.corrcoef(normalized_features, rowvar=False)
        
        plt.figure(figsize=(8, 6))
        plt.imshow(correlations, cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar(label="Coefficient de corrélation (Pearson)")
        plt.xticks(ticks=numpy.arange(len(feature_names)), labels=feature_names, rotation=45, ha='right')
        plt.yticks(ticks=numpy.arange(len(feature_names)), labels=feature_names)
        plt.title("Matrice de corrélation des caractéristiques")
        plt.tight_layout()
        plt.show()
        
        pca = PCA(n_components=min(10, normalized_features.shape[1]))
        pca_features = pca.fit_transform(normalized_features)
        
        print("\n--- Analyse en Composantes Principales (PCA) ---")
        print(f"Variance expliquée : {numpy.round(pca.explained_variance_ratio_ * 100, 2)}")
        print(f"Information totale conservée : {numpy.sum(pca.explained_variance_ratio_) * 100:.2f}%\n")
        
    
        representation = dataset.Representation(data=pca_features, labels=images.labels)
    # =========================================================================
    # TODO: Problématique: Comparez différents classificateurs sur cette
    # représentation, comme dans le laboratoire 2 et 3.
    # -------------------------------------------------------------------------
    # 
    # -------------------------------------------------------------------------
        # L2.E4.1, L2.E4.2, L2.E4.3 et L2.E4.4
    # Complétez la classe NeuralNetworkClassifier dans helpers/classifier.py
    # -------------------------------------------------------------------------

    if True:
        nn_classifier = classifier.NeuralNetworkClassifier(input_dim=representation.data.shape[-1],
                                                        output_dim=len(representation.unique_labels),
                                                        n_hidden=3,
                                                        n_neurons=8,
                                                        lr=0.001,
                                                        n_epochs=70,
                                                        batch_size=16)
        # -------------------------------------------------------------------------
        nn_classifier.fit(representation)
        save_dir = pathlib.Path(__file__).parent / "saves"
        save_dir.mkdir(parents=True, exist_ok=True)

        # 2. Save the model
        nn_classifier.save(save_dir / "multimodal_classifier.keras")

        # 3. Plot and save training metrics
        viz.plot_metric_history(nn_classifier.history)
        plt.savefig(save_dir / "training_history.png")
        
        # viz.plot_numerical_decision_regions(nn_classifier, representation)
        
        data = nn_classifier.preprocess_data(representation.data)
        
        predictions = nn_classifier.predict(data)
        predictions = numpy.array([representation.unique_labels[i] for i in predictions])

        error_rate, indexes_errors = analysis.compute_error_rate(representation.labels, predictions)
        print(f"\n\n{len(indexes_errors)} erreurs de classification sur {len(predictions)} échantillons ({error_rate * 100:.2f}%).")

        # 4. Affichage de la matrice de confusion et blocage de la fenêtre
        viz.show_confusion_matrix(representation.labels, predictions, representation.unique_labels, plot=True)
        
        plt.show(block=True)

if __name__ == "__main__":
    problematique()
