import numpy as np

def random_forest_classify(X_train: list, y_train: list, X_test: list, n_estimators: int = 10, max_depth: int = 5, max_features = "sqrt", seed: int = 42) -> list:
    """
    Returns one random-forest prediction per test row.
    """
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.int64)
    X_test = np.asarray(X_test, dtype=np.float64)

    n, d = X_train.shape

    # Cantidad de features que evaluamos en cada nodo.
    if max_features == "sqrt":
        m = max(1, int(np.sqrt(d)))
    elif max_features == "log2":
        m = max(1, int(np.log2(d)))
    else:
        m = min(d, int(max_features))

    # Compartido por bootstrap y selección de features.
    rng = np.random.RandomState(seed)
    tree_predictions = []

    for _ in range(n_estimators):
        root = {}

        # Los índices repetidos conservan las repeticiones del bootstrap.
        bootstrap_indices = rng.randint(0, n, size=n)
        stack = [(root, bootstrap_indices, 0)]

        while stack:
            node, indices, depth = stack.pop()
            X = X_train[indices]
            y = y_train[indices]
            n_node = len(y)

            etiquetas, counts = np.unique(y, return_counts=True)
            node["prediction"] = int(etiquetas[np.argmax(counts)])

            # No sorteamos features si el nodo ya debe ser una hoja.
            if depth >= max_depth or len(etiquetas) == 1:
                continue

            parent_gini = 1.0 - np.sum((counts / n_node) ** 2)
            best_gain = 0.0
            best_split = None

            # Nuevo subconjunto por nodo, sin reemplazo y ordenado.
            features = np.sort(rng.choice(d, size=m, replace=False))

            for feature in features:
                for threshold in np.unique(X[:, feature]):
                    left_mask = X[:, feature] <= threshold
                    n_left = int(left_mask.sum())
                    n_right = n_node - n_left

                    if n_left == 0 or n_right == 0:
                        continue

                    _, left_counts = np.unique(
                        y[left_mask], return_counts=True
                    )
                    _, right_counts = np.unique(
                        y[~left_mask], return_counts=True
                    )

                    left_gini = 1.0 - np.sum(
                        (left_counts / n_left) ** 2
                    )
                    right_gini = 1.0 - np.sum(
                        (right_counts / n_right) ** 2
                    )

                    gain = parent_gini - (
                        n_left * left_gini + n_right * right_gini
                    ) / n_node

                    # Ganancia estrictamente mayor: conserva el primer empate.
                    if gain > best_gain:
                        best_gain = gain
                        best_split = (feature, threshold, left_mask)

            if best_split is None:
                continue

            feature, threshold, left_mask = best_split
            node["feature"] = feature
            node["threshold"] = threshold
            node["left"] = {}
            node["right"] = {}

            # Pila LIFO: procesamos primero todo el subárbol izquierdo.
            stack.append(
                (node["right"], indices[~left_mask], depth + 1)
            )
            stack.append(
                (node["left"], indices[left_mask], depth + 1)
            )

        # Predicciones del árbol actual.
        predictions = []

        for row in X_test:
            node = root

            while "feature" in node:
                if row[node["feature"]] <= node["threshold"]:
                    node = node["left"]
                else:
                    node = node["right"]

            predictions.append(node["prediction"])

        tree_predictions.append(predictions)

    # Votación entre árboles para cada fila de test.
    tree_predictions = np.asarray(tree_predictions, dtype=np.int64)
    predictions = []

    for votes in tree_predictions.T:
        etiquetas, counts = np.unique(votes, return_counts=True)
        predictions.append(int(etiquetas[np.argmax(counts)]))

    return predictions
