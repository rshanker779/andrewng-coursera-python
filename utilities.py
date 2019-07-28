import os
base_data_directory = os.path.join(os.path.dirname(__file__), "data")
import numpy as np

def add_column_of_ones(X:np.ndarray)->np.ndarray:
    m,_ = X.shape
    return np.concatenate((np.ones((m, 1)), X), axis=1)

def open_text_file(name:str)->np.ndarray:
    """
    Opens a text file as an array as the open octave function
    :param name: name of file
    :return: array of data
    """
    return np.genfromtxt(os.path.join(base_data_directory, name), delimiter=',')
