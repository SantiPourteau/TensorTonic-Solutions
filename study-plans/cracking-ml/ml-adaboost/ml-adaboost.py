import numpy as np

def adaboost_classify(X_train: list, y_train: list, X_test: list, n_estimators: int = 10, seed: int = 42) -> list:
    """
    Returns one boosted label per test row.
    """
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=int)

    if len(X_test) == 0:
        return []

    X_test = np.asarray(X_test, dtype=float)
    n_samples, n_features = X.shape

    weights = np.full(n_samples, 1.0 / n_samples)
    scores = np.zeros(len(X_test))
    # posibles tresholds de los features para los decision stumps
    thresholds = [np.unique(X[:, j]) for j in range(n_features)]

    for _ in range(n_estimators):
        best_error = np.inf
        #guardamos best stump para aplicarlo a datos nuevos y para calcular los scores
        best_stump = None
        best_predictions = None

        for j in range(n_features):
            for threshold in thresholds[j]:
                base_predictions = np.where(X[:, j] <= threshold, -1, 1)

                for polarity in (1, -1):
                    predictions = polarity * base_predictions
                    error = np.sum(weights[predictions != y])

                    if error < best_error:
                        best_error = error
                        best_stump = (j, threshold, polarity)
                        best_predictions = predictions

        error = np.clip(best_error, 1e-12, 1.0 - 1e-12)
        alpha = 0.5 * np.log((1.0 - error) / error)

        weights *= np.exp(-alpha * y * best_predictions)
        weights /= weights.sum()

        j, threshold, polarity = best_stump
        test_predictions = polarity * np.where(
            X_test[:, j] <= threshold, -1, 1
        )
        scores += alpha * test_predictions

    return np.where(scores >= 0, 1, -1).tolist()
