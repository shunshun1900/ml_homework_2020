import numpy as np

import matplotlib.pyplot as plt
#%matplotlib inline
# This line is only needed if you have a HiDPI display
# %config InlineBackend.figure_format = 'retina' 

from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.preprocessing import StandardScaler

class SMOModel:
    """Container object for the model used for SMO  """
    def __init__(self,X, y, C, kernel, alphas, b, errors):
        self.X = X
        self.y = y
        self.C = C
        self.kernel = kernel 
        self.alphas = alphas
        self.b = b
        self.errors = errors
        self._obj = []
        self.m = len(self.X)

def linear_kernal(x, y, b=1):

    return x @ y.T + b

def gaussian_kernel(x, y, sigma=1):

    if np.ndim(x) == 1 and np.ndim(y) == 1:
        result = np.exp(-(np.linalg.norm(x-y,2)**2)/2*sigma**2)
    elif (np.ndim(x) == 1 and np.ndim(y)>1) or (np.ndim(y) == 1 and np.ndim(x)>1):
        result = np.exp(-(np.linalg.norm(x-y,axis=1)**2)/2*sigma**2)
    elif (np.ndim(x) > 1 and np.ndim(y)>1):
        result = np.exp(-(np.linalg.norm(x[:,np.newaxis]-y[:,np.newaxis],axis=2)**2)/2*sigma**2)
    return result

def objective_function(alphas, target, kernel, X_train):

    """Returns the SVM objective function based in the input model defined by
    alphas: vector of Lagrange multipliers
    target: vector of class labels (-1 or 1) for training data
    kernel: kernel function
    X_train: training data for model."""

    return np.sum(alphas) - 0.5*np.sum((target[:,None]*target[None,:])*kernel(X_train, X_train)*(alphas[:,None]*alphas[None,:]))

def decision_function(alphas, target, kernel, X_train, x_test, b):
    """Applies the SVM decision function to the input feature vectors in `x_test`."""
    return (alphas*target) @ kernel(X_train, x_test) + b