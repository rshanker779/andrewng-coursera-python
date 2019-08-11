import numpy as np
import utilities as utils
import scipy.optimize as op

from utilities import sigmoid


def cost_function(theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Suppose we have n features and m training examples
    :param X: Matrix of size (n,m)
    :param y: Vector of size (n,1)
    :param theta: Vector of size (m,1)
    :return: cost of logistic regression error
    """
    m = len(y)
    h = sigmoid(np.matmul(X, theta)).reshape((m, 1))
    J = (1 / m) * np.sum(-np.log(h[y == 1])) + (1 / m) * np.sum(
        -np.log((1 - h[y == 0]))
    )
    return J


def grad_function(theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Suppose we have n features and m training examples
    :param X: Matrix of size (n,m)
    :param y: Vector of size (n,1)
    :param theta: Vector of size (m,1)
    :return: gradient of logistic regression error
    """
    m = len(y)
    n = len(theta)
    h = sigmoid(np.matmul(X, theta)).reshape((m, 1))
    grad = (1 / m) * np.sum((h - y) * X, axis=0).reshape((n, 1))
    return grad.reshape((n,))


def predict(theta: np.ndarray, X: np.ndarray) -> np.ndarray:
    m, _ = X.shape
    p = sigmoid(np.matmul(X, theta)) >= 0.5
    return 1.0 * p.reshape((m, 1))


def map_features(X1: np.ndarray, X2: np.ndarray, degree=6) -> np.ndarray:
    m = len(X1)
    out = np.ones((m, 1))
    for i in range(1, degree + 1):
        for j in range(i + 1):
            new_feature = np.power(X1, i - j) * np.power(X2, j)
            new_feature = new_feature.reshape((m, 1))
            out = np.append(out, new_feature, axis=1)
    return out


def cost_function_reg(
    theta: np.ndarray, X: np.ndarray, y: np.ndarray, lambda_: float
) -> np.ndarray:
    """
    Suppose we have n features and m training examples
    :param X: Matrix of size (n,m)
    :param y: Vector of size (n,1)
    :param theta: Vector of size (m,1)
    :param lambda_: float, regularization parameter
    :return:  logistic regression error with L2 reg
    """
    m, _ = X.shape
    J = cost_function(theta, X, y)
    J += (lambda_ / (2 * m)) * np.sum(np.power(theta, 2))
    J -= (lambda_ / (2 * m)) * theta[0] ** 2
    return J


def grad_function_reg(
    theta: np.ndarray, X: np.ndarray, y: np.ndarray, lambda_: float
) -> np.ndarray:
    """
    Suppose we have n features and m training examples
    :param X: Matrix of size (n,m)
    :param y: Vector of size (n,1)
    :param theta: Vector of size (m,1)
    :param lambda_: float, regularization parameter
    :return: gradient of logistic regression error
    """
    m, n = X.shape
    grad = grad_function(theta, X, y)
    grad += (lambda_ / m) * theta.reshape(n)
    grad[0] -= (lambda_ / m) * theta[0]
    return grad.reshape((n,))


def ex2_main():
    data = utils.open_text_file("ex2data1.txt")
    X = data[:, :2]
    m, _ = X.shape
    y = data[:, 2]
    y = y.reshape((m, 1))
    X = utils.add_column_of_ones(X)
    _, n = X.shape
    initial_theta = np.zeros((n, 1))
    cost = cost_function(initial_theta, X, y)
    grad = grad_function(initial_theta, X, y)
    print("Cost at initial theta (zeros): %f" % cost)
    print("Expected cost (approx): 0.693")
    print("Gradient at initial theta (zeros): ")
    print(" %s " % grad)
    print("Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628")

    test_theta = np.asarray([-24, 0.2, 0.2]).reshape((n, 1))
    cost = cost_function(test_theta, X, y)
    grad = grad_function(test_theta, X, y)
    print("\nCost at test theta: %f" % cost)
    print("Expected cost (approx): 0.218")
    print("Gradient at test theta: ")
    print(" %s " % grad)
    print("Expected gradients (approx):\n 0.043\n 2.566\n 2.647")

    # Use BFGS, similar to octave. Ignore warnings
    res = op.minimize(
        fun=cost_function,
        x0=initial_theta,
        args=(X, y),
        method="BFGS",
        jac=grad_function,
    )
    theta = res.x
    cost = cost_function(theta, X, y)
    # op.fmin_bfgs(cost_function, initial_theta,)
    print("Cost at theta found by fminunc: %f" % cost)
    print("Expected cost (approx): 0.203")
    print("theta: ")
    print(" %s " % theta)
    print("Expected theta (approx):")
    print(" -25.161\n 0.206\n 0.201")
    test_array = np.asarray([1, 45, 85]).reshape((1, n))
    prob = sigmoid(np.matmul(test_array, theta))
    print(
        "For a student with scores 45 and 85, we predict an admission probability of %f"
        % prob
    )
    print("Expected value: 0.775 +/- 0.002")

    p = predict(theta, X)
    print("Train Accuracy: %f" % (np.mean(p == y) * 100))
    print("Expected accuracy (approx): 89.0")


def ex2_reg_main():
    data = utils.open_text_file("ex2data2.txt")
    X = data[:, :2]
    y = data[:, 2]
    m, n = X.shape
    y = y.reshape((m, 1))
    X = map_features(X[:, 0], X[:, 1])
    m, n = X.shape
    initial_theta = np.zeros((n, 1))
    lambda_ = 0
    cost = cost_function_reg(initial_theta, X, y, lambda_)
    grad = grad_function_reg(initial_theta, X, y, lambda_)

    print("Cost at initial theta (zeros): %f" % cost)
    print("Expected cost (approx): 0.693")
    print("Gradient at initial theta (zeros) - first five values only:")
    print(" %s" % grad[0:5])
    print("Expected gradients (approx) - first five values only:")
    print(" 0.0085\n 0.0188\n 0.0001\n 0.0503\n 0.0115")

    test_theta = np.ones((n, 1))
    cost = cost_function_reg(test_theta, X, y, 10)
    grad = grad_function_reg(test_theta, X, y, 10)
    print("Cost at test theta  with lambda=10: %f" % cost)
    print("Expected cost (approx): 3.16")
    print("Gradient at test theta (zeros) - first five values only:")
    print(" %s" % grad[0:5])
    print("Expected gradients (approx) - first five values only:")
    print(" 0.3460\n 0.1614\n 0.1948\n 0.2269\n 0.0922")

    lambda_ = 1
    res = op.minimize(
        fun=cost_function_reg,
        x0=initial_theta,
        args=(X, y, lambda_),
        method="BFGS",
        jac=grad_function_reg,
    )
    theta = res.x
    p = predict(theta, X)
    print("Train Accuracy: %f" % (np.mean(p == y) * 100))
    print("Expected accuracy (with lambda = 1): 83.1 (approx)\n")


if __name__ == "__main__":
    ex2_main()
    ex2_reg_main()
