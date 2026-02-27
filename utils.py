import numpy as np

def one_hot_for_labels(labels:np.ndarray, unique_labels:list) -> np.ndarray:
    number_of_classes = len(unique_labels)
    labels_one_hot = np.zeros((labels.data.shape[0], number_of_classes))
    label_name_to_index = {label: index for index, label in enumerate(unique_labels)}

    for i, label in enumerate(labels):
        label_index = label_name_to_index[label]
        labels_one_hot[i, label_index] = 1
    
    return labels_one_hot

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
