import numpy as np

def ridge_regression(X: list, y: list, alpha: float, lr: float, epochs: int) -> tuple:
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

        penalizacion = np.sum(w ** 2)
        L = (1/n) * (error**2) + alpha * penalizacion
        # como L no usa directamente w, hay que usar regla de la cadena:
        # L depende del error, que depende de y_pred que depende de w
        # Al querer saber cuanto cambia L cuando cambian los pesos, tenemos que saber como cambia L cuando cambia el error
        # dL/dw = dL/derror * derror/dw
        # dL/derror * derror/dw = dL/derror * derror/dy_pred * dy_pred/dw
        # Como L = 1/n * error**2 (la primera parte de L por lo menos)
        # dL/derror = (2/n) * error
        # como el error es y_pred - y_targer:
        # derror/dy_pred = 1 - 0 = 1
        # ahora tenemos que hacer dy_pred/dw
        # y_pred = Xw+b
        # por lo tanto dy_pred/dw = X
        
        

        # Ahora hay que derivar la penalizacion ridge, respecto de los pesos.
        # Es la suma de los pesos al cuadrado, por lo tanto la derivada es 2*w
        # # Por lo tanto queda dL/dw = (2/n) * error * X + 2 * alpha * w
        grad_error = (2 / n) * (data_train.T @ error)
        grad_penalizacion = 2 * alpha * w
        grad_penalizacion[0] = 0.0
        
        grad_w = grad_error + grad_penalizacion
        w = w - lr * grad_w

    return np.round(w[1:], 4).tolist(), round(float(w[0]), 4)
    
        
        