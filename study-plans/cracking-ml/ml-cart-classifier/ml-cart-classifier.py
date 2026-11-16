import numpy as np

def cart_classify(X_train: list, y_train: list, X_test: list, max_depth: int = 5, min_samples: int = 2) -> list:
    """
    Returns one predicted class label per test row.
    """
    X_train = np.asarray(X_train, dtype=np.float64)
    X_test = np.asarray(X_test, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.int64)

    #arbol de decision
    root = {}
    #guarda los nodos que hay que hacer
    # ahora solo depende de la raiz, guarda (nodo, indices de las filas que ve el nodo, profundidad)
    stack = [(root, np.arange(len(y_train)), 0)]

    while stack:
        node, indices, depth = stack.pop()
        X = X_train[indices]
        y = y_train[indices]
        n = len(y)

        etiquetas, counts = np.unique(y, return_counts=True)

        # En un empate, elige la etiqueta más pequeña.
        node["prediction"] = int(etiquetas[np.argmax(counts)])

        # pre-pruning
        if depth >= max_depth or n < min_samples or len(etiquetas) == 1:
            continue

        #calculamos la impureza gini y despues el criterio de particion con gini
        parent_gini = 1.0 - np.sum((counts / n) ** 2)
        best_gain = 0.0
        best_split = None

        # pasamos por cada feature y por cada valor observado de la feature
        for feature in range(X.shape[1]):
            for threshold in np.unique(X[:, feature]):
                left_mask = X[:, feature] <= threshold
                n_left = int(left_mask.sum())
                n_right = n - n_left

                if n_left == 0 or n_right == 0:
                    continue

                _, left_counts = np.unique(
                    y[left_mask], return_counts=True
                )
                _, right_counts = np.unique(
                    y[~left_mask], return_counts=True
                )

                left_gini = 1.0 - np.sum((left_counts / n_left) ** 2)
                right_gini = 1.0 - np.sum((right_counts / n_right) ** 2)

                gain = parent_gini - (
                    n_left * left_gini + n_right * right_gini
                ) / n

                # Conserva el primer split en caso de empate.
                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best_split = (feature, threshold, left_mask)
        # Si ninguna partición produce una ganancia positiva, el nodo queda como hoja.
        if best_split is None:
            continue

        #nos quedamos con el mejor split y armamos el nodo que contiene:
        # feature de particion
        # threshold de particion
        # el nodo izquierda y el nodo derecha
        feature, threshold, left_mask = best_split

        node["feature"] = feature
        node["threshold"] = threshold
        node["left"] = {}
        node["right"] = {}
        # se agrega a la pila de nodos con los indices de las filas correspondientes
        stack.append((node["right"], indices[~left_mask], depth + 1))
        stack.append((node["left"], indices[left_mask], depth + 1))

    predictions = []

    for row in X_test:
        node = root

        while "feature" in node:
            if row[node["feature"]] <= node["threshold"]:
                node = node["left"]
            else:
                node = node["right"]

        predictions.append(node["prediction"])

    return predictions

