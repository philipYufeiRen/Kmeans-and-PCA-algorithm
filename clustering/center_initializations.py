"""
CSCC11 - Introduction to Machine Learning, Winter 2022, Assignment 4
B. Chan, S. Wei, D. Fleet
"""

import numpy as np

def kmeans_pp(K, train_X):
    """ This function runs K-means++ algorithm to choose the centers.

    Args:
    - K (int): Number of centers.
    - train_X (ndarray (shape: (N, D))): A NxD matrix consisting N D-dimensional input data.

    Output:
    - centers (ndarray (shape: (K, D))): A KxD matrix consisting K D-dimensional centers.
    """
    centers = np.empty(shape=(K, train_X.shape[1]))

    # ====================================================
    # TODO: Implement your solution within the box
    
    # ====================================================

    return centers

def random_init(K, train_X):
    """ This function randomly chooses K data points as centers.

    Args:
    - K (int): Number of centers.
    - train_X (ndarray (shape: (N, D))): A NxD matrix consisting N D-dimensional input data.

    Output:
    - centers (ndarray (shape: (K, D))): A KxD matrix consisting K D-dimensional centers.
    """
    centers = train_X[np.random.randint(train_X.shape[0], size=K)]
    return centers
