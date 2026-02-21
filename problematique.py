
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

from helpers import analysis, classifier, viz
import helpers.dataset as dataset
from features import *



def problematique():
    images = dataset.ImageDataset("data/image_dataset/")
    
    noise_feature = calculate_noise(images).reshape(-1, 1)
    
    colors_top_left = calculate_most_common_color_in_top_left_corner(images).reshape(-1, 3)
    ratio_high_low = vertical_horizontal_ratio(images).reshape(-1, 1)
    symmetry = calculate_ratio_symmetry(images).reshape(-1, 1)
    lab_spikes_width = calculate_lab_spikes_width(images).reshape(-1, 1)
    features = np.hstack((noise_feature, colors_top_left, symmetry, ratio_high_low, lab_spikes_width))
    
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
    if False:
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
        
        spickes_representation = dataset.Representation(data=lab_spikes_width, labels=images.labels)
        viz.plot_features_distribution(spickes_representation, 
                                   title="Distribution de la largeur des pics dans l'espace Lab", 
                                   xlabel="Largeur des pics", 
                                   ylabel="Nombre d'images",
                                   n_bins=32,
                                   features_names=["Largeur des pics"])
        
        subrepresentation = dataset.Representation(data=features[:, 0:3], labels=images.labels)
        viz.plot_data_distribution(subrepresentation,
                            title="Distribution basée sur le bruit",
                              xlabel="Bruit",
                              ylabel="Couleur R",
                              zlabel="symétrie", isNormalized=True)
                              
        plt.show()
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
                                                        lr=0.01,
                                                        n_epochs=100,
                                                        batch_size=16)
        # -------------------------------------------------------------------------
        nn_classifier.fit(representation)

        # Save the model
        nn_classifier.save(pathlib.Path(__file__).parent / "saves/multimodal_classifier.keras")

        # Plot training metrics
        viz.plot_metric_history(nn_classifier.history)

        #viz.plot_numerical_decision_regions(nn_classifier, representation)
        
        data = nn_classifier.preprocess_data(representation.data)
        
        predictions = nn_classifier.predict(data)
        predictions = numpy.array([representation.unique_labels[i] for i in predictions])

        error_rate, indexes_errors = analysis.compute_error_rate(representation.labels, predictions)
        print(f"\n\n{len(indexes_errors)} erreurs de classification sur {len(predictions)} échantillons ({error_rate * 100:.2f}%).")

        viz.show_confusion_matrix(representation.labels, predictions, representation.unique_labels, plot=True)
        
        plt.show()

if __name__ == "__main__":
    problematique()
