import numpy as np

def knn_classifier(X_train: list, y_train: list, X_test: list, k: int) -> list:
    """
    Returns one predicted label for each test row.
    """
    # Rank training rows by distance to the test row, using the original training index to break equal-distance ties
    # if class counts tie, choose the smallest label.
    # Convert the feature matrices to NumPy float64 arrays.
    # Rank neighbors by distance and then by training index.
    # Resolve equal vote counts by choosing the smallest label.
    # Return one integer prediction for each test row.
    # Constraints
    # Training and test rows have the same positive feature width.
    # k is between one and the number of training rows.
    # Labels are integers.
    X_test = np.array(X_test,dtype=np.float64)
    X_train = np.array(X_train,dtype=np.float64)
    y_train = np.array(y_train)


    #normalizamos y estandarizamos porque si estan las features estan a escala distinta el calculo de distancia es afectado y features numericamente grandes dominan la distancia.
    # mean = X_train.mean(axis=0)
    # std = X_train.std(axis=0)
    # std = np.where(std == 0, 1.0, std)
    
    # X_train = (X_train - mean) / std
    # X_test = (X_test - mean) / std
    preds = []
    for x in X_test:
        #distancia a todos los puntos
        distances = np.linalg.norm(X_train - x, axis=1)
        # devuelve los índices que ordenarian las distancias de menor a mayor
        indices = np.argsort(distances, kind="stable")[:k]
        labels = y_train[indices]
        # hacemos la votacion por mayoria con los labels
        vals, counts = np.unique(labels, return_counts=True)
        preds.append(int(vals[np.argmax(counts)]))

    return preds
