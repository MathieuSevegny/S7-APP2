import numpy as np


def one_hot_for_labels(labels:np.ndarray, unique_labels:list) -> np.ndarray:
    number_of_classes = len(unique_labels)
    labels_one_hot = np.zeros((labels.data.shape[0], number_of_classes))
    label_name_to_index = {label: index for index, label in enumerate(unique_labels)}

    for i, label in enumerate(labels):
        label_index = label_name_to_index[label]
        labels_one_hot[i, label_index] = 1
    
    return labels_one_hot

def labels_from_one_hot(one_hot_labels:np.ndarray, unique_labels:list) -> np.ndarray:
    index_to_label_name = {index: label for index, label in enumerate(unique_labels)}
    labels = np.zeros(one_hot_labels.shape[0], dtype=object)

    for i, one_hot in enumerate(one_hot_labels):
        labels[i] = index_to_label_name[one_hot]
    
    return labels