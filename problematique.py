
import os
import pathlib

# Must be call before any other TensorFlow/Keras import
# Suppress oneDNN custom operations info
# Suppress INFO and WARNING messages from TF (0=all, 1=no INFO, 2=no INFO/WARN, 3=no error)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from matplotlib import pyplot as plt

from helpers import classifier, viz
import helpers.dataset as dataset
from features import *



def problematique():
    images = dataset.ImageDataset("data/image_dataset/")

    noise_feature = calculate_noise(images).reshape(-1, 1)
    colors_top_left = calculate_most_common_color_in_top_left_corner(images).reshape(-1, 3)
    
    features = np.hstack((noise_feature, colors_top_left))
    
    # normalize each feature to [-1, 1]
    for i in range(features.shape[1]):
        features[:, i] = (features[:, i] - np.min(features[:, i])) / (np.max(features[:, i]) - np.min(features[:, i])) * 2 - 1
    
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
        subrepresentation = dataset.Representation(data=features[:,0:2], labels=images.labels)
        viz.plot_data_distribution(subrepresentation,
                            title="Distribution basée sur le bruit",
                              xlabel="Bruit",
                              ylabel="Couleur R",
                              zlabel="Couleur G")
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
        nn_classifier = classifier.NeuralNetworkClassifier(input_dim=representation.data.shape[1],
                                                        output_dim=len(representation.unique_labels),
                                                        n_hidden=3,
                                                        n_neurons=8,
                                                        lr=0.01,
                                                        n_epochs=1000,
                                                        batch_size=16)
        # -------------------------------------------------------------------------
        nn_classifier.fit(representation)

        # Save the model
        nn_classifier.save(pathlib.Path(__file__).parent / "saves/multimodal_classifier.keras")

        # Plot training metrics
        viz.plot_metric_history(nn_classifier.history)

if __name__ == "__main__":
    problematique()
