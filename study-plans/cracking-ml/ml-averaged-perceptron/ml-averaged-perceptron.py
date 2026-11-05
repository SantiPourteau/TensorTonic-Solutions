import numpy as np

def averaged_perceptron(X: list, y: list, lr: float, epochs: int) -> dict:
    """
    Returns a dictionary: weights (float list), bias (float), predictions (integer list).
    """
# Visit samples sequentially for exactly epochs passes.
# Update on zero or negative signed margin.
# The average includes the parameter state after every sample visit.
# Return weights, bias, and training-set predictions.
# Round numeric outputs to 4 decimal places.
# Constraints
# X is a nonempty list of equal-length, nonempty feature rows.
# y contains one label per row, each either minus one or plus one.
# lr is positive and epochs is a positive integer.
    w = np.array([0] * len(X[0]), dtype=np.float64)
    b = np.float64(0)
    train = np.asarray(X, dtype=np.float64)
    target = np.asarray(y, dtype=np.float64)
    weights = []
    biases = []

    for _ in range(epochs):
        for i in range(train.shape[0]):
            pred = np.dot(w, train[i]) + b
            if target[i] * pred <= 0:
                w += lr * target[i] * train[i]
                b += lr * target[i]

            weights.append(w.copy())
            biases.append(b)

    average_weight = np.mean(weights, axis=0)
    average_bias = np.mean(biases)

    predictions = []
    for i in range(train.shape[0]):
        pred = np.dot(average_weight, train[i]) + average_bias
        if pred > 0:
            pred = 1
        else:
            pred = -1
        predictions.append(pred)

    return {
        "weights": np.round(average_weight, 4).tolist(),
        "bias": float(np.round(average_bias, 4)),
        "predictions": predictions,
    }
        
    
