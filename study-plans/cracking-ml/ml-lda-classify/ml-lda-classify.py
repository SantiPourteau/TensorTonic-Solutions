import numpy as np

def lda_classify(X_train: list, y_train: list, X_test: list) -> list:
    """
    Returns one predicted label for each test row.
    """
    # Modela las clases como gaussianas multivariadas con una covarianza compartida.
    # Usa las etiquetas para estimar las medias, los priors y la dispersión dentro de las clases.
    train = np.asarray(X_train, dtype=np.float64)
    target = np.asarray(y_train)
    test = np.asarray(X_test, dtype=np.float64)
    
    clases = np.unique(target)
    means = []
    priors = []
    covarianza = np.zeros((train.shape[1], train.shape[1]), dtype=np.float64)

    for c in clases:
        samples = train[target == c]
        #calculamos μk y πk 
        mean = np.mean(samples, axis=0)
        prior = samples.shape[0] / train.shape[0]

        means.append(mean)
        priors.append(prior)

        for i in range(samples.shape[0]):
            diferencia = samples[i] - mean
            # Acumulamos la dispersión dentro de cada clase respecto de su media.
            covarianza += np.outer(diferencia, diferencia)

    covarianza /= train.shape[0] - len(clases)
    covarianza += 1e-6 * np.eye(train.shape[1])
    
    pesos = []
    biases = []

    for i in range(len(clases)):
        w = np.linalg.solve(covarianza, means[i])
        b = -0.5 * np.dot(means[i], w) + np.log(priors[i])

        pesos.append(w)
        biases.append(b)

    preds = []

    for i in range(test.shape[0]):
        scores = []

        for j in range(len(clases)):
            score = np.dot(test[i], pesos[j]) + biases[j]
            scores.append(score)
        # argmax devuelve el primer máximo, las clases ya están ordenadas.
        pred = clases[np.argmax(scores)]
        preds.append(int(pred))

    return preds
