import numpy as np

def permutation_importance(X: list, y: list, predict_fn, n_repeats: int = 5, seed: int = 42) -> list:
    """
    Returns one permutation importance per feature.
    """
    #computar baseline score
    # para cada feature j, hacer R repeticiones en las que se calcula baseline - permuted score
    # promediar las j caidas en score para obtener importance de j
    
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)

    rng = np.random.RandomState(seed)
    baseline_acc = np.mean(np.asarray(predict_fn(X)) == y)

    importances = []

    for j in range(X.shape[1]):
        caidas = []

        for r in range(n_repeats):
            X_permuted = X.copy()
            X_permuted[:, j] = rng.permutation(X[:, j])

            predictions = np.asarray(predict_fn(X_permuted))
            permuted_accuracy = np.mean(predictions == y)

            caidas.append(baseline_acc - permuted_accuracy)

        importances.append(round(float(np.mean(caidas)), 4))

    return importances