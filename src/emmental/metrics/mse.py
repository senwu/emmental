from sklearn.metrics import mean_squared_error


def mse(gold, probs):
    return {"mse": float(mean_squared_error(gold, probs))}
