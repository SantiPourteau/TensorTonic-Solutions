import numpy as np

def gaussian_naive_bayes(X_train: list, y_train: list, X_test: list) -> list:
    """
    Returns one predicted label for each test row.
    """
    # y_pred = argmax sobre c de log[P(c|x)]
    # P(c|x) es proporcional a log P(c) + sum(logP(xj|c))
    # P(xj|c) = -0.5 * log (2πσ**2) - (xj-μ)**2/(2σ**2)
    # Hay que estimar el prior P(c), feature means, y population variances con ddof zero
    # Con eso podemos calcular todo lo anterior y llegar a y_pred
    
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test, dtype=np.float64)

    if X_test.size == 0:
        return []

    classes, counts = np.unique(y_train, return_counts=True)
    log_priors = np.log(counts / y_train.size)

    
    means = []
    variances = []
    
    for c in classes:
        # Seleccionar los ejemplos de entrenamiento de esta clase.
        class_mask = y_train == c
        class_samples = X_train[class_mask]
    
        # Calcular una media y una varianza por feature.
        class_mean = class_samples.mean(axis=0)
        class_variance = class_samples.var(axis=0, ddof=0)
    
        # Aplicar el suavizado a cada varianza.
        class_variance = class_variance + 1e-9
    
        means.append(class_mean)
        variances.append(class_variance)
    
    means = np.array(means)
    variances = np.array(variances)

    # Dimensiones: (ejemplos de test, clases, features).
    # Restamos la media de la clase a cada ejemplo
    differences = X_test[:, None, :] - means[None, :, :]
    
    log_likelihoods = -0.5 * np.sum(
        np.log(2 * np.pi * variances)
        + differences**2 / variances,
        axis=2,
    )

    scores = log_priors + log_likelihoods
    predictions = classes[np.argmax(scores, axis=1)]

    result = []

    for prediction in predictions:
        integer_prediction = int(prediction)
        result.append(integer_prediction)
    
    return result