import os
import numpy as np
import matplotlib.pyplot as plt
from utilities import base_data_directory


def compute_cost(X: np.ndarray, y: np.ndarray, theta: np.ndarray):
    """
    Suppose we have n features and m training examples
    :param X: Matrix of size (n,m)
    :param y: Vector of size (n,1)
    :param theta: Vector of size (m,1)
    :return: Real
    """
    m = len(y)
    diff = np.matmul(X, theta) - y
    return (1 / (2 * m)) * np.matmul(diff.T, diff)


def gradient_descent(
    X: np.ndarray, y: np.ndarray, theta: np.ndarray, alpha: float, num_iters: int
):
    """
    Suppose we have n features and m training examples
    :param X: Matrix of size (n,m)
    :param y: Vector of size (n,1)
    :param theta: Vector of size (m,1)
    :param alpha: Learning rate
    :param num_iters: Number of rounds of boosting
    :returns theta, J_history: learn weight and history of loss
    """
    m = len(y)
    n = len(theta)
    J_history = np.zeros((num_iters, 1))
    for iter in range(num_iters):
        differential = (1 / m) * np.sum((np.matmul(X, theta) - y) * X, axis=0).reshape(
            (n, 1)
        )
        theta = theta - alpha * differential
    return theta, J_history


def plot_data(X, y):
    plt.scatter(X, y)


def feature_normalization(X:np.ndarray):
    """
    Computes the StandardScaling of X (as Sklearn)
    :param X: Any 2dim array
    :return: Scaled X, mean of X, and std of X
    """
    mu = np.mean(X,axis=0)
    #Changing delta dof to match octave/matlab
    sigma=np.std(X, axis=0, ddof=1)
    return (X-mu)/sigma, mu, sigma

def normal_equation(X:np.ndarray, y:np.ndarray):
    """
    Suppose we have n features and m training examples
    :param X: Matrix of size (n,m)
    :param y: Vector of size (m,1
    :return: Theta, parameters for linear regression found from normal eqn
    """
    return np.matmul( np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T), y)


def ex1_main():
    data = np.genfromtxt(
        os.path.join(base_data_directory, "ex1data1.txt"), delimiter=","
    )
    X = data[:, 0]
    y = data[:, 1]
    m = len(y)
    X = X.reshape((m, 1))
    y = y.reshape((m, 1))
    plot_data(X, y)
    # Add column of ones
    X = np.concatenate((np.ones((m, 1)), X), axis=1)
    theta = np.zeros((2, 1))
    iterations = 1500
    alpha = 0.01
    J = compute_cost(X, y, theta)
    print("With theta = [0 ; 0]\nCost computed = %s" % J)
    print("Expected cost value (approx) 32.07")
    J = compute_cost(X, y, np.asarray([-1, 2]).reshape(2, 1))
    print("With theta = [0 ; 0]\nCost computed = %s" % J)
    print("Expected cost value (approx) 54.24")
    theta, _ = gradient_descent(X, y, theta, alpha, iterations)
    print("Theta found by gradient descent: %s" % theta)
    print("Expected theta values (approx)")
    print(" -3.6303\n  1.1664\n")
    plt.plot(X[:, 1], np.matmul(X, theta))
    predict1 = np.matmul(np.asarray([1, 3.5]).reshape((1, 2)), theta)
    print("For population = 35,000, we predict a profit of %f" % (predict1 * 10000))
    print("Expected: 4519.77")

    predict2 = np.matmul(np.asarray([1, 7]).reshape((1, 2)), theta)
    print("For population = 70,000, we predict a profit of %f" % (predict2 * 10000))
    print("Expected:45342.45")
    plt.show()

def ex1_multi_main():
    data = np.genfromtxt(os.path.join(base_data_directory, 'ex1data2.txt'), delimiter=',')
    X = data[:,:2]
    y=data[:,2]
    m = len(y)
    X, mu, sigma =feature_normalization(X)
    y = y.reshape((m,1))
    X = np.concatenate((np.ones((m, 1)), X), axis=1)
    print("Mu calcualted as %s" %mu)
    print("Expected:    2000.6809      3.1702")
    print("Sigma calcualted as %s" %sigma)
    print("Expected:       794.70235     0.76098")
    alpha =0.001
    num_iters=40000
    theta = np.zeros((3,1))
    theta, J_history = gradient_descent(X, y, theta, alpha, num_iters)
    print('Theta computed from gradient descent: %s'%theta)
    X_ex = np.asarray([1650,3]).reshape((1,2))
    X_ex = (X_ex-mu)/sigma
    X_ex_reshape = np.ones((1,3))
    X_ex_reshape[:,1:] = X_ex
    X_ex = X_ex_reshape
    price = np.matmul(X_ex, theta)

    print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):\n $%f'% price)
    #Now do normal equation
    data = np.genfromtxt(os.path.join(base_data_directory, 'ex1data2.txt'), delimiter=',')
    X = data[:,:2]
    X = np.concatenate((np.ones((m, 1)), X), axis=1)
    theta = normal_equation(X, y)
    print('Theta computed from normal equation: %s'%theta)
    X_ex = np.asarray([1,1650,3])
    price = np.matmul(X_ex, theta)
    print('Predicted price of a 1650 sq-ft, 3 br house (using normal eqn):\n $%f'% price)


if __name__ == "__main__":
    ex1_main()
    ex1_multi_main()
