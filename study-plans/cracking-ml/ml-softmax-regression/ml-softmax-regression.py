import numpy as np

def softmax_regression(X: list, y: list, num_classes: int, lr: float, n_iters: int) -> tuple:
    """
    Returns the fitted weight matrix and bias vector.
    """
    data=np.array(X,dtype=np.float64)
    y_target = np.array(y, dtype=np.int64)
    
    n, d = data.shape
    W = np.zeros((d,num_classes),dtype=np.float64)
    b = np.zeros(num_classes,dtype=np.float64)


    y_one_hot = np.zeros((n, num_classes), dtype=np.float64)
    y_one_hot[np.arange(n), y_target] = 1.0
    for iter in range(n_iters):
        Z = data @ W + b

        # Estabilidad numerica:
        Z_stable = Z - np.max(Z, axis=1, keepdims=True)

        # funcion softmax
        exp_Z = np.exp(Z_stable)
        probs = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

        # La derivada conjunta de softmax + cross-entropy es:
        # dL/dZ = probs - y_one_hot
        error = probs - y_one_hot

        # Promedio de los gradientes del batch completo
        grad_W = (1 / n) * (data.T @ error)
        grad_b = (1 / n) * np.sum(error, axis=0)

        # Gradient descent
        W = W - lr * grad_W
        b = b - lr * grad_b

    return (np.round(W, 4).tolist(),np.round(b, 4).tolist())

        