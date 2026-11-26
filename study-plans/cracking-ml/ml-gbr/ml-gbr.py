import numpy as np

def gbr_predict(X_train: list, y_train: list, X_test: list, n_estimators: int = 10, max_depth: int = 3, learning_rate: float = 0.1, seed: int = 42) -> list:
    """
    Returns one boosted regression prediction per test row.
    """
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    X_test = np.asarray(X_test, dtype=np.float64)

    # F0 es el promedio de todos los targets
    initial_mean = float(y.mean())
    train_predictions = np.full(len(y), initial_mean)
    trees = []

    # Hacemos n_estimadores iteraciones
    for _ in range(n_estimators):
        # Calculamos el gradiente de la loss en este caso es el residual
        residuals = y - train_predictions

        tree = {}
        stack = [(tree, np.arange(len(y)), 0)]

        while stack:
            node, indices, depth = stack.pop()
            r = residuals[indices]

            node["value"] = float(r.mean())

            if depth >= max_depth or len(indices) <= 1:
                continue

            parent_mse = np.mean((r - node["value"]) ** 2)
            best_gain = 0.0
            best_split = None

            # buscamos el mejor split en el nodo actual del arbol
            for feature in range(X.shape[1]):
                values = X[indices, feature]

                for threshold in np.unique(values):
                    left_mask = values <= threshold
                    right_mask = ~left_mask

                    if not left_mask.any() or not right_mask.any():
                        continue

                    r_left = r[left_mask]
                    r_right = r[right_mask]

                    mse_left = np.mean((r_left - r_left.mean()) ** 2)
                    mse_right = np.mean((r_right - r_right.mean()) ** 2)

                    weighted_mse = (
                        len(r_left) * mse_left
                        + len(r_right) * mse_right
                    ) / len(r)

                    gain = parent_mse - weighted_mse

                    if gain > best_gain:
                        best_gain = gain
                        best_split = (
                            feature,
                            threshold,
                            indices[left_mask],
                            indices[right_mask],
                        )

            if best_split is None:
                continue

            feature, threshold, left_indices, right_indices = best_split

            node["feature"] = feature
            node["threshold"] = threshold
            node["left"] = {}
            node["right"] = {}

            stack.append((node["right"], right_indices, depth + 1))
            stack.append((node["left"], left_indices, depth + 1))

        #una vez terminado el arbol lo agregamos a la lista de arboles
        trees.append(tree)

        # actualizamos F sobre entrenamiento para el siguiente árbol.
        for i, row in enumerate(X):
            node = tree

            while "feature" in node:
                if row[node["feature"]] <= node["threshold"]:
                    node = node["left"]
                else:
                    node = node["right"]

            train_predictions[i] += learning_rate * node["value"]

    # Predicción final: F_0 + learning_rate * suma de árboles.
    test_predictions = np.full(len(X_test), initial_mean)

    for tree in trees:
        for i, row in enumerate(X_test):
            node = tree

            while "feature" in node:
                if row[node["feature"]] <= node["threshold"]:
                    node = node["left"]
                else:
                    node = node["right"]

            test_predictions[i] += learning_rate * node["value"]

    return np.round(test_predictions, 4).tolist()