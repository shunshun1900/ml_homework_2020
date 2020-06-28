import matplotlib.pyplot as plt
import setup_problem
from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np
import nodes
import graph
import plot_utils
from sklearn.datasets import make_blobs
import pandas as pd
import os
import patsy 
from patsy import dmatrices
import statsmodels.api as sm

class PoissonRegression(BaseEstimator, RegressorMixin):
    ''' Poisson regression with computation graph   '''
    def __init__(self, step_size=0.0005, max_num_epochs= 500, init_param_scale=0.01):
        self.max_num_epochs = max_num_epochs
        self.step_size = step_size
        self.init_param_scale = init_param_scale

        # Build computation graph
        self.x = nodes.ValueNode(node_name="x") # to hold a vector input
        self.y = nodes.ValueNode(node_name="y") # to hold a scalar response
        self.w = nodes.ValueNode(node_name="w") # to hold the parameter vector
        self.b = nodes.ValueNode(node_name="b") # to hold the bias parameter (scalar)

        self.affine = nodes.VectorScalarAffineNode(self.w, self.x, self.b, node_name="affine") # to hold a affine transform
        #self.active = nodes.ExpNode(self.affine, node_name="active") # to hold a activation function node using exp
        self.prediction = nodes.ExpNode(self.affine, node_name="predict") # to hold a prediction node
        self.objective = nodes.PoissonLikehoodNode(self.prediction, self.y, 
                                                    node_name="objective") # to hold a negative log likehood node
        # Group nodes into types to construct computation graph function
        self.inputs = [self.x]
        self.outcomes = [self.y]
        self.parameters = [self.w, self.b]

        self.graph = graph.ComputationGraphFunction(self.inputs, self.outcomes,
                                                          self.parameters, self.prediction,
                                                          self.objective)


    def fit(self, X, y):
        num_instances, num_ftrs = X.shape
        y = y.reshape(-1)

        ## TODO: Initialize parameters (small random numbers -- not all 0, to break symmetry )
        s = self.init_param_scale
        #init_values = None 
        # ## TODO

        #init_values = {"w": np.zeros(num_ftrs), "b": np.array(0.0)}
        init_values = {"w": np.random.randn(num_ftrs), "b": np.random.randn(1)}

        self.graph.set_parameters(init_values)

        for epoch in range(self.max_num_epochs):
            shuffle = np.random.permutation(num_instances)
            epoch_obj_tot = 0.0
            for j in shuffle:
                obj, grads = self.graph.get_gradients(input_values = {"x": X[j]},
                                                    outcome_values = {"y": y[j]})
                #print(obj)
                epoch_obj_tot += obj
                # Take step in negative gradient direction
                steps = {}
                for param_name in grads:
                    steps[param_name] = -self.step_size * grads[param_name]
                    self.graph.increment_parameters(steps)
                #pdb.set_trace()

            if epoch % 5 == 0:
                print("Epoch ",epoch,": average objective value=",epoch_obj_tot/num_instances)
    
    def Poisson(self, lamda, k):
        ''' computing Poisson value
            parameters: 
            lamda: Poisson parameter
            k: class 
        '''
        pro = lamda**k/np.math.factorial(k)*np.exp(-lamda)
        return pro



    def predict(self,X, y=None):
        try:
            getattr(self, "graph")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")

        num_instances = X.shape[0]
        k_space = range(4000)
        preds = np.zeros(num_instances)
        probabilities = np.zeros((num_instances, len(k_space)))
        for j in range(num_instances):
            preds[j] = self.graph.get_prediction(input_values={"x":X[j]})
            for k in range(len(k_space)):
                probabilities[j][k] = self.Poisson(preds[j], k)

        
        return preds, probabilities

def main():
    '''
    # Create the  training data
    np.random.seed(2)
    X, y = make_blobs(n_samples=300,cluster_std=.25, centers=np.array([(3,5),(7,1),(10,8)]))
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50)
    plt.show()

    print(X.shape, y.shape)
    print(X[1],y[1])
    ''' 
    #Create a pandas DataFrame for the counts data set.
    cur_dir = os.path.split(os.path.realpath(__file__))[0]
    file_name = "%s\\nyc_bb_bicyclist_counts.csv"%(cur_dir)
    print(file_name)
    df = pd.read_csv(file_name, header=0, infer_datetime_format=True, parse_dates=[0], index_col=[0])

    #Add a few derived regression variables.
    ds = df.index.to_series()
    df['MONTH'] = ds.dt.month
    df['DAY_OF_WEEK'] = ds.dt.dayofweek
    df['DAY'] = ds.dt.day

    #Create the training and testing data sets.
    mask = np.random.rand(len(df)) < 0.8
    df_train = df[mask]
    df_test = df[~mask]
    print('Training data set length='+str(len(df_train)))
    print('Testing data set length='+str(len(df_test)))

    #Setup the regression expression in patsy notation. We are telling patsy that BB_COUNT is our dependent variable and
    # it depends on the regression variables: DAY, DAY_OF_WEEK, MONTH, HIGH_T, LOW_T and PRECIP.
    expr = """BB_COUNT ~ DAY  + DAY_OF_WEEK + MONTH + HIGH_T + LOW_T + PRECIP"""

    #Set up the X and y matrices
    y_train, X_train = dmatrices(expr, df_train, return_type='dataframe')
    y_test, X_test = dmatrices(expr, df_test, return_type='dataframe')

    print(type(y_train),type(X_train))
    X = np.array(X_train)
    y = np.array(y_train)
    print(type(y),type(X))

    estimator = PoissonRegression(step_size=0.005, max_num_epochs=100)
    estimator.fit(X, y)
    lamda, probability = estimator.predict(X) 
    #print(estimator.w.out, estimator.b.out)
    #print(y,lamda,probability)
    print(np.argmax(probability,1))

    '''
    plt.figure()
    line = plt.plot(range(5))[0]                # plot函数返回的是一个列表,因为可以同时画多条线的哦;
    line.set_color('r')
    line.set_linewidth(2.0)
    plt.show()
    '''
if __name__ == '__main__':
    main()