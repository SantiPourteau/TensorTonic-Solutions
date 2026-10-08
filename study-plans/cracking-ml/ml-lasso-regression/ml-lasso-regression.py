import numpy as np

def lasso_regression(X: list, y: list, alpha: float, lr: float, epochs: int) -> tuple:
    """
    Returns the fitted weight list and bias.
    """
    data=np.array(X,dtype=np.float64)
    y_target=np.array(y,dtype=np.float64)

    n, d= data.shape

    # media = data.mean(axis=0)
    # desvio = data.std(axis=0)
    
    # # Evita division por cero en features constantes
    # desvio[desvio == 0] = 1.0
    
    # data_std = (data - media) / desvio
    data_train = np.column_stack((np.ones(n), data))
    
    w = np.zeros(d + 1, dtype=np.float64)


    for epoch in range(epochs):
        y_pred = data_train @ w
        error = y_pred - y_target

        grad_error = (2 / n) * (data_train.T @ error)

        # Ahora es regularizacion l1 no l2, por lo tanto la derivada es +-1 o 0
        grad_penalizacion = alpha * np.sign(w)

        grad_penalizacion[0] = 0.0
        
        grad_w = grad_error + grad_penalizacion
        w = w - lr * grad_w

    return np.round(w[1:], 4).tolist(), round(float(w[0]), 4)
    
        
        
