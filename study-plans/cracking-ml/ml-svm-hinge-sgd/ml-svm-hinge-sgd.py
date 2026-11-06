import numpy as np

def svm_hinge_sgd(X: list, y: list, lr: float, lam: float, n_epochs: int) -> dict:
    """
    Returns fitted parameters (rounded to 4 decimals) and training predictions.
    """
    train = np.asarray(X, dtype=np.float64)
    target = np.asarray(y)
    
    w = np.zeros(train.shape[1], dtype=np.float64)
    b = np.float64(0.0)

    for epoch in range(n_epochs):

        for i in range(train.shape[0]):
            margen = target[i] * (train[i] @ w + b)
            if margen < 1:
                w -= lr * (lam * w - target[i] * train[i])
                b += lr * target[i]
            else: 
                w -= lr * lam * w
                #derivada del sesgo es 0

    preds = np.where(X @ w + b > 0, 1, -1)

    return {
    "weights": np.round(w, 4).tolist(),
    "bias": float(np.round(b, 4)),
    "predictions": preds.tolist(),
    }

