import numpy as np

def stacking_classify(X_train: list, y_train: list, X_test: list, n_folds: int = 3, seed: int = 42) -> list:
    """
    Returns a list of integer class predictions in test-row order.
    """
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.int64)
    X_test = np.asarray(X_test, dtype=np.float64).reshape(-1, X.shape[1])

    n, n_features = X.shape
    classes = np.unique(y)

    indices = np.arange(n)
    np.random.RandomState(seed).shuffle(indices)
    folds = np.array_split(indices, n_folds)

    meta_train = np.zeros((n, 2), dtype=np.float64)
    meta_test = np.zeros((len(X_test), 2), dtype=np.float64)

    # primero se generan las out of fold y el despues se genera el test
    for fold_idx in range(n_folds + 1):
        if fold_idx < n_folds:
            train_idx = np.concatenate([
                folds[j] for j in range(n_folds) if j != fold_idx
            ])
            val_idx = folds[fold_idx]
            X_current = X[train_idx]
            y_current = y[train_idx]
            X_predict = X[val_idx]
        else:
            X_current = X
            y_current = y
            X_predict = X_test

        size = len(y_current)
        labels, counts = np.unique(y_current, return_counts=True)
        majority = labels[np.argmax(counts)]
        parent_gini = 1.0 - np.sum((counts / size) ** 2)

        best_gain = 0.0
        best_feature = None
        best_threshold = None
        best_left_label = majority
        best_right_label = majority

        # Entrenar el decision stump
        for feature in range(n_features):
            for threshold in np.unique(X_current[:, feature]):
                left = X_current[:, feature] <= threshold
                n_left = np.count_nonzero(left)
                n_right = size - n_left

                if n_left == 0 or n_right == 0:
                    continue

                left_labels, left_counts = np.unique(
                    y_current[left], return_counts=True
                )
                right_labels, right_counts = np.unique(
                    y_current[~left], return_counts=True
                )

                left_gini = 1.0 - np.sum((left_counts / n_left) ** 2)
                right_gini = 1.0 - np.sum((right_counts / n_right) ** 2)

                gain = parent_gini - (
                    n_left * left_gini + n_right * right_gini
                ) / size

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
                    best_left_label = left_labels[np.argmax(left_counts)]
                    best_right_label = right_labels[np.argmax(right_counts)]

        predictions = np.zeros((len(X_predict), 2), dtype=np.float64)

        # Primera meta-feature, clase predicha por el stump
        if best_feature is None:
            predictions[:, 0] = majority
        else:
            predictions[:, 0] = np.where(
                X_predict[:, best_feature] <= best_threshold,
                best_left_label,
                best_right_label,
            )

        # Segunda meta-feature, clase predicha por KNN con k=3
        for row_idx, row in enumerate(X_predict):
            distances = np.sum((X_current - row) ** 2, axis=1)

            neighbors = np.argsort(distances, kind="stable")[:3]
            neighbor_labels, neighbor_counts = np.unique(
                y_current[neighbors], return_counts=True
            )
            predictions[row_idx, 1] = neighbor_labels[
                np.argmax(neighbor_counts)
            ]

        if fold_idx < n_folds:
            meta_train[val_idx] = predictions
        else:
            meta_test[:] = predictions

    # Una regresión logística binaria por clase
    scores = np.zeros((len(X_test), len(classes)), dtype=np.float64)

    for class_idx, label in enumerate(classes):
        targets = (y == label).astype(np.float64)
        weights = np.zeros(2, dtype=np.float64)
        bias = 0.0

        for _ in range(500):
            logits = np.clip(meta_train @ weights + bias, -500, 500)
            probabilities = 1.0 / (1.0 + np.exp(-logits))
            errors = probabilities - targets

            weights -= 0.1 * (meta_train.T @ errors) / n
            bias -= 0.1 * np.mean(errors)

        test_logits = np.clip(meta_test @ weights + bias, -500, 500)
        scores[:, class_idx] = 1.0 / (1.0 + np.exp(-test_logits))

    # classes está ordenado, argmax elige la primera clase en empates
    return classes[np.argmax(scores, axis=1)].tolist()
