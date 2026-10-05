import numpy as np

def logistic_regression(X: list, y: list, lr: float, n_iters: int) -> tuple:
    """
    Returns the fitted weight list and bias.
    """
    data=np.array(X, dtype=np.float64)
    y_target=np.array(y,dtype=np.float64)

    n, d= data.shape
    w = np.zeros(d, dtype=np.float64)
    b = np.float64(0.0)

    for iter in range(n_iters):
        z = (data @ w) + b
        z = np.clip(z,-500,500)
        p = 1 / (1 + np.exp(-z))
        error = p - y_target

        #Use the batch gradients X.T(p−y)/nX T(p−y)/n and mean(p−y)mean(p−y).

        # loss es -1/n sum (y_target*log(y_pred) + (1-y_target)log(1-y_pred))
        # la derivada de L respecto de los pesos es:
        # 1/n X^T(y_pred-y_target)

        
        grad_w = (1 / n) * (data.T @ error)

        # la derivada de L respecto del bias es:
        # 1/n @ 1.T @ (y_pred - y_target)
        # es el mean del error
        grad_b = np.mean(error)

        w = w - lr * grad_w
        b = b - lr * grad_b

    return np.round(w, 4).tolist(), round(float(b), 4)

        
        
        