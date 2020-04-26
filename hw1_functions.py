import pandas as pd
import logging
import numpy as np
from numpy import *
import sys
from matplotlib import pyplot as plt
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

def feature_normalization(train, test):
    """Rescale the data so that each feature in the training set is in
    the interval [0,1], and apply the same transformations to the test
    set, using the statistics computed on the training set.

    Args:
        train - training set, a 2D numpy array of size (num_instances, num_features)
        test  - test set, a 2D numpy array of size (num_instances, num_features)
    Returns:
        train_normalized - training set after normalization
        test_normalized  - test set after normalization

    """
    # TODO
    scaler = MinMaxScaler()
    
    train_normalized = scaler.fit_transform(train)
    test_normalized = scaler.transform(test)
    
    return train_normalized, test_normalized

def compute_square_loss(X, y, theta):
    """
    Given a set of X, y, theta, compute the square loss for predicting y with X*theta

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D array of size (num_features)

    Returns:
        loss - the square loss, scalar
    """
    loss = 0 #initialize the square_loss
    #TODO
    X = mat(X)
    y = mat(y).T
    theta = mat(theta).T
    loss = ((X * theta - y).T * (X * theta - y))[0,0]
    
    return loss/size(y)

def compute_square_loss_gradient(X, y, theta):
    """
    Compute gradient of the square loss (as defined in compute_square_loss), at the point theta.

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D numpy array of size (num_features)

    Returns:
        grad - gradient vector, 1D numpy array of size (num_features)
    """
    #TODO
    grand = 0
    
    X = mat(X)
    y = mat(y).T
    theta = mat(theta).T
    
    grad = 2 * X.T * (X * theta - y )  / size(y)
     
    return grad.getA().reshape(size(grad))

def batch_grad_descent(X, y, alpha=0.1, num_iter=100, check_gradient=False):
    """
    In this question you will implement batch gradient descent to
    minimize the square loss objective

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        alpha - step size in gradient descent
        num_iter - number of iterations to run
        check_gradient - a boolean value indicating whether checking the gradient when updating

    Returns:
        theta_hist - store the the history of parameter vector in iteration, 2D numpy array of size (num_iter+1, num_features)
                    for instance, theta in iteration 0 should be theta_hist[0], theta in ieration (num_iter) is theta_hist[-1]
        loss_hist - the history of objective function vector, 1D numpy array of size (num_iter+1)
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    theta_hist = np.zeros((num_iter+1, num_features))  #Initialize theta_hist
    loss_hist = np.zeros(num_iter+1) #initialize loss_hist
    theta = np.zeros(num_features) #initialize theta
    #TODO
    
    for i in range(0,num_iter+1,1):
        
        theta_hist[i,:] = theta
        
        loss = compute_square_loss(X, y, theta)
        
        grad = compute_square_loss_gradient(X, y, theta)
        
        theta = theta - alpha*grad 
        loss_hist[i] = loss
        
    return theta_hist, loss_hist

def compute_regularized_square_loss_gradient(X, y, theta, lambda_reg):
    """
    Compute the gradient of L2-regularized square loss function given X, y and theta

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D numpy array of size (num_features)
        lambda_reg - the regularization coefficient

    Returns:
        grad - gradient vector, 1D numpy array of size (num_features)
    """
    #TODO
    grand = 0
    
    X = mat(X)
    y = mat(y).T
    theta = mat(theta).T
    
    grad = 2 * X.T * (X * theta - y )  / size(y) + 2*lambda_reg*theta
     
    return grad.T.getA()

def compute_regularized_square_loss(X, y, theta, lambda_reg):
    loss = 0 #initialize the square_loss
    #TODO
    X = mat(X)
    y = mat(y).T
    theta = mat(theta).T
    loss = ((X * theta - y).T * (X * theta - y)) + lambda_reg*theta.T*theta
    
    return loss[0,0] / size(y)

def regularized_grad_descent(X, y, alpha=0.1, lambda_reg=1, num_iter=1000):
    """
    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        alpha - step size in gradient descent
        lambda_reg - the regularization coefficient
        numIter - number of iterations to run

    Returns:
        theta_hist - the history of parameter vector, 2D numpy array of size (num_iter+1, num_features)
        loss_hist - the history of loss function without the regularization term, 1D numpy array.
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    theta_hist = np.zeros((num_iter+1, num_features))  #Initialize theta_hist
    loss_hist = np.zeros(num_iter+1) #initialize loss_hist
    theta = np.zeros(num_features) #initialize theta
    #TODO
    
    for i in range(0,num_iter+1,1):
        
        theta_hist[i,:] = theta
        
        loss = compute_regularized_square_loss(X, y, theta, lambda_reg)
        
        grad = compute_regularized_square_loss_gradient(X, y, theta, lambda_reg)
        
        theta = theta - alpha*grad 
        loss_hist[i] = loss
        
    return theta_hist, loss_hist

def stochastic_grad_descent(X, y, alpha=0.1, lambda_reg=1, num_iter=1000):
    """
    In this question you will implement stochastic gradient descent with a regularization term

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        alpha - string or float. step size in gradient descent
                NOTE: In SGD, it's not always a good idea to use a fixed step size. Usually it's set to 1/sqrt(t) or 1/t
                if alpha is a float, then the step size in every iteration is alpha.
                if alpha == "1/sqrt(t)", alpha = 1/sqrt(t)
                if alpha == "1/t", alpha = 1/t
        lambda_reg - the regularization coefficient
        num_iter - number of epochs (i.e number of times) to go through the whole training set

    Returns:
        theta_hist - the history of parameter vector, 3D numpy array of size (num_iter, num_instances, num_features)
        loss hist - the history of regularized loss function vector, 2D numpy array of size(num_iter, num_instances)
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    theta = np.ones(num_features) #Initialize theta


    theta_hist = np.zeros((num_iter, num_instances, num_features))  #Initialize theta_hist
    loss_hist = np.zeros((num_iter, num_instances)) #Initialize loss_hist
    #TODO
    
    for epoch in range(0,num_iter,1):
        #data = [d for d in zip(X,y)]
        data = np.hstack((X,y.T))
        np.random.shuffle(data)
        
        for i in range(data.shape[0]):
        #t = np.random.randint(num_instances, size=1)
            x_i = data[i,:-1]
            y_i = data[i,-1]
        
            loss = compute_regularized_square_loss(x_i, y_i, theta, lambda_reg)
            grad = compute_regularized_square_loss_gradient(x_i, y_i, theta, lambda_reg)
    
            loss_hist[epoch,i] = loss
            theta_hist[epoch,i,] = theta
        
            theta = theta - alpha*grad 
            alpha = alpha/sqrt(i+1)
    
    return theta_hist, loss_hist 