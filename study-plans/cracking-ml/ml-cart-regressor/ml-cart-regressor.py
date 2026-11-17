import numpy as np

def cart_regress(X_train: list, y_train: list, X_test: list, max_depth: int = 5, min_samples: int = 2) -> list:
    """
    Returns one regression prediction per test row.
    """

    #codigo muy parecido al del arbol de clasificacion, cambia principalmente la impureza usada y como predecir
    X_train = np.asarray(X_train, dtype=np.float64)
    X_test = np.asarray(X_test, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)

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

        # La predicción del nodo es la media de los valores que ve.
        node["prediction"] = float(np.mean(y))

        # pre-pruning
        if depth >= max_depth or n < min_samples or np.all(y == y[0]):
            continue

        #calculamos MSE
        parent_mse = np.mean((y - np.mean(y)) ** 2)

        # Inicializamos la búsqueda del mejor split para este nodo.
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

                y_left = y[left_mask]
                y_right = y[~left_mask]
                
                left_mse = np.mean((y_left - np.mean(y_left)) ** 2)
                right_mse = np.mean((y_right - np.mean(y_right)) ** 2)
                
                gain = parent_mse - (
                    n_left * left_mse + n_right * right_mse
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

        predictions.append(round(node["prediction"], 4))

    return predictions


