import numpy as np
import utilities as utils
from typing import Tuple
import scipy.optimize as op

def sigmoid(z:np.ndarray)->np.ndarray:
    return 1/(1+np.exp(-z))

def cost_function(theta:np.ndarray, X:np.ndarray, y:np.ndarray)->np.ndarray:
    """
    Suppose we have n features and m training examples
    :param X: Matrix of size (n,m)
    :param y: Vector of size (n,1)
    :param theta: Vector of size (m,1)
    :return: cost of logistic regression error
    """
    m=len(y)
    h = sigmoid(np.matmul(X, theta)).reshape((m ,1))
    J = (1 / m) * np.sum(-np.log(h[y==1])) + (1 / m) * np.sum(-np.log((1 - h[y == 0])))
    return J

def grad_function(theta:np.ndarray, X:np.ndarray, y:np.ndarray)->np.ndarray:
    """
    Suppose we have n features and m training examples
    :param X: Matrix of size (n,m)
    :param y: Vector of size (n,1)
    :param theta: Vector of size (m,1)
    :return: gradient of logistic regression error
    """
    m=len(y)
    n=len(theta)
    h = sigmoid(np.matmul(X, theta)).reshape((m ,1))
    grad = (1 / m) * np.sum(h - y* X, axis=0).reshape((n, 1))
    return grad

def predict(theta:np.ndarray, X:np.ndarray)->np.ndarray:
    m, _ = X.shape
    p = sigmoid(np.matmul(X, theta))>=0.5
    return 1.0*p.reshape((m,1))

def ex2_main():
    data = utils.open_text_file("ex2data1.txt")
    X = data[:,:2]
    m,_=X.shape
    y = data[:, 2]
    y = y.reshape((m,1))
    X = utils.add_column_of_ones(X)
    _,n=X.shape
    initial_theta = np.zeros((n, 1))
    cost = cost_function(initial_theta, X, y)
    grad =grad_function(initial_theta, X, y)
    print('Cost at initial theta (zeros): %f'% cost)
    print('Expected cost (approx): 0.693')
    print('Gradient at initial theta (zeros): ')
    print(' %s '% grad)
    print('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628')

    test_theta = np.asarray([-24,0.2,0.2]).reshape((n,1))
    cost = cost_function(test_theta, X, y)
    grad = grad_function(test_theta, X, y)
    print('\nCost at test theta: %f'%cost)
    print('Expected cost (approx): 0.218')
    print('Gradient at test theta: ')
    print(' %s '% grad)
    print('Expected gradients (approx):\n 0.043\n 2.566\n 2.647')

    #Use BFGS, similar to octave. Ignore warnings
    res = op.minimize(fun=cost_function,x0=initial_theta, args=(X,y), method='BFGS', hess=grad_function)
    theta =res.x
    cost = cost_function(theta, X, y)
    # op.fmin_bfgs(cost_function, initial_theta,)
    print('Cost at theta found by fminunc: %f'% cost)
    print('Expected cost (approx): 0.203')
    print('theta: ')
    print(' %s '% theta)
    print('Expected theta (approx):')
    print(' -25.161\n 0.206\n 0.201')
    test_array = np.asarray([1,45,85]).reshape((1,n))
    prob = sigmoid(np.matmul(test_array, theta))
    print('For a student with scores 45 and 85, we predict an admission probability of %f'% prob)
    print('Expected value: 0.775 +/- 0.002')

    p=predict(theta, X)
    print('Train Accuracy: %f'% (np.mean(p == y) * 100))
    print('Expected accuracy (approx): 89.0')
    return

if __name__ == '__main__':
    ex2_main()