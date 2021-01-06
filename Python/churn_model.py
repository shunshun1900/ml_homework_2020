import warnings

import pandas as pd

import numpy as np

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier as KNN

from sklearn.linear_model import LogisticRegression as LR

from sklearn.ensemble import RandomForestClassifier as RF

from sklearn.model_selection import KFold

warnings.filterwarnings("ignore", category=FutureWarning)

#read files
churn_df = pd.read_csv('churn.csv')
#data cleaning
to_drop = ['State', 'Area Code', 'Phone', 'Churn?']

df = churn_df.drop(to_drop, axis=1)

label = churn_df['Churn?']

y = np.where(label=='True.',1,0)

yes_no_cols = ["Int'l Plan","VMail Plan"]

df[yes_no_cols] = df[yes_no_cols] == 'yes'

features = df.columns

X = df.values.astype(np.float)
#z_score method
scaler = StandardScaler()

X = scaler.fit_transform(X)

# Training
def train_cv = (X, y, clf_class, **kwargs):

    kf = KFold(n_splits=5, shuffle=True)

    y_pred = y.copy()

    for train_index, test_index in kf.split(X)
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
        clf = clf_class(**kwargs)
        clf.fit(X_train, y_train)
        y_pred[test_index] = clf.pred(X_test)
        return y_pred

def accuracy(y_ture, y_pred):

    return np.mean(y_ture==y_pred)

print("logistic regression:")
print("%.3f"% accuracy(y,train_cv(X, y, LR)))  
print("Random Forest:")
print("%.3f"% accuracy(y,train_cv(X, y, RF))) 
print("k-nearst-neighbors:")
print("%.3f"% accuracy(y,train_cv(X, y, KNN))) 

