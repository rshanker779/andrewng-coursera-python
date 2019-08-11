import scipy.io
import utilities as utils
import os
import numpy as np
from ex2 import cost_function_reg, grad_function_reg
import scipy.optimize as op


def one_vs_all(X, y, num_labels, lambda_):
    X = utils.add_column_of_ones(X)
    m, n = X.shape
    all_theta = np.zeros((num_labels, n))
    for c in range(1, num_labels + 1):
        initial_theta = np.ones((n, 1))
        res = op.minimize(
            fun=cost_function_reg,
            x0=initial_theta,
            args=(X, (y == c), lambda_),
            method="BFGS",
            jac=grad_function_reg,
            options={
                # 'maxiter':50,
                "disp": True
            },
        )
        # print(res)
        theta = res.x
        all_theta[c - 1, :] = theta
    return all_theta


def predict_one_vs_all(all_theta, X):
    X = utils.add_column_of_ones(X)
    max_indices = np.argmax(np.matmul(X, all_theta.T), axis=1)
    max_indices += 1
    return max_indices.reshape((X.shape[0], 1))


def predict_nn(theta_1, theta_2, X):
    X = utils.add_column_of_ones(X)
    a1 = utils.sigmoid(np.matmul(X, theta_1.T))
    a1 = utils.add_column_of_ones(a1)
    a2 = utils.sigmoid(np.matmul(a1, theta_2.T))
    max_indices = np.argmax(a2, axis=1)
    max_indices += 1
    return max_indices.reshape(((X.shape[0], 1)))


def ex3():
    data = scipy.io.loadmat(os.path.join(utils.base_data_directory, "ex3data1.mat"))
    X = data["X"]
    y = data["y"]
    input_layer_size = 400
    num_labels = 10

    m, n = X.shape
    # Random test arrays, so sizes and shapes are hard coded
    theta_t = np.asarray([-2, -1, 1, 2]).reshape((4, 1))
    X_t = np.asarray([i / 10.0 for i in range(1, 16)]).reshape((3, 5)).T
    X_t = utils.add_column_of_ones(X_t).reshape((5, 4))
    y_t = np.asarray([1, 0, 1, 0, 1]).reshape((5, 1), order="F")
    J = cost_function_reg(theta_t, X_t, y_t, 3)
    grad = grad_function_reg(theta_t, X_t, y_t, 3)

    print("Cost: %s" % J)
    print("Expected cost: 2.534819")
    print("Gradients:")
    print(" %s " % grad)
    print("Expected gradients:")
    print(" 0.146561 -0.548558 0.724722 1.398003")
    lambda_ = 0.1
    all_theta = one_vs_all(X, y, num_labels, lambda_)
    pred = predict_one_vs_all(all_theta, X)
    # Note haven't been able to acheive accuracies as highas matlab even with lower learning rate
    print("Training Set Accuracy: %f" % (np.mean(1.0 * (pred == y)) * 100))
    print("Should be at least 94%")


def ex3_nn():
    data = scipy.io.loadmat(os.path.join(utils.base_data_directory, "ex3data1.mat"))
    X = data["X"]
    y = data["y"]
    data = scipy.io.loadmat(os.path.join(utils.base_data_directory, "ex3weights.mat"))
    theta_1 = data["Theta1"]
    theta_2 = data["Theta2"]
    pred = predict_nn(theta_1, theta_2, X)
    print("Training Set Accuracy: %f" % (np.mean(1.0 * (pred == y)) * 100))
    print("Expected 97.52%")


if __name__ == "__main__":
    ex3()
    ex3_nn()
