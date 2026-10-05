import numpy as np

def linear_regression_from_scratch(X: list, y: list, lr: float, epochs: int) -> tuple:
    """
    Returns the fitted weight list and bias.
    """
    
    data=np.array(X,dtype=np.float64)
    y_target=np.array(y,dtype=np.float64)

    n, d= data.shape
    w = np.zeros(d, dtype=np.float64)
    b = np.float64(0.0)
    
    for epoch in range(epochs):
        y_pred = (data @ w) + b
        error = y_pred - y_target
        loss = (1/n) * np.sum(error**2)
        # Queremos derivada de la loss respecto de los pesos.
        # Al derivar respecto de los pesos, no importa la derivada de y_target.
        # Importa la derivada de y_pred
        # dL/dw = sum (dL/derror * derror/dy_pred * dy_pred/dw)
        # dL/derror = (2/n) * error
        # derror/dy_pred = 1
        # dy_pred/dw = data.T
        grad_w = (2 / n) * (data.T @ error)
        
        # Aplicamos la regla de la cadena:
        #     dL/db = sum_i(dL/derror * derror/dy_pred * dy_pred/db)
        #
        # dL/derror = (2/n) * error
        # derror/dy_pred = 1
        # dy_pred/db = 1
        grad_b = (2 / n) * np.sum(error)

        w = w - lr * grad_w
        b = b - lr * grad_b

    return np.round(w, 4).tolist(), round(float(b), 4)
        
        

