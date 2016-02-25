'''
Tampere University of Technology
SGN 41006 - Signal Interpretation (Spring 2016)
Exercise 7 Solutions

Contact:    oguzhan.gencoglu@tut.fi (Office: TE406)
            andrei.cramariuc@tut.fi (Office: TE314)
'''

# load required packages
from __future__ import print_function
import numpy as np
from scipy.io import loadmat
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score


if __name__ == '__main__':
    
    '''
    Question 4
    '''
    # Read data
    X_train, X_test, y_train, y_test = [loadmat('arcene.mat')[field] for field in ['X_train', 'X_test', 'y_train', 'y_test']]
    
    # Select features
    rfecv = RFECV(estimator = LogisticRegression(), step = 100, cv = 10)
    rfecv.fit(X_train, y_train.flatten())
    
    # Print number of selected features
    print("\nNumber of selected features:", rfecv.n_features_)
    
    # Train the whole training set with the selected features
    lr1 = LogisticRegression()
    lr1.fit(X_train[:, rfecv.support_], y_train.flatten())
    
    # Performance on the test set
    score_lr1 = accuracy_score(y_test.flatten(), lr1.predict(X_test[:, rfecv.support_]))
    print("RFECV accuracy score on the test set:", score_lr1)
    
    '''
    Question 5
    '''
    # 10 fold CV with L1 regularization
    parameters =     {
        'penalty': ['l1'],
        'C': np.logspace(-3, 5, 20)
    } 
    clf = GridSearchCV(estimator=LogisticRegression(), 
                                        param_grid = parameters,
                                        cv = 10, 
                                        n_jobs=-1)
    clf.fit(X_train, y_train.flatten())
    best_params = clf.best_params_
    print("\nBest parameters are:", best_params)
    
    # Train the whole training set with the selected parameters
    lr2 = LogisticRegression(penalty = best_params['penalty'], C = best_params['C'])
    lr2.fit(X_train, y_train.flatten())
    print("Number of selected features:", np.count_nonzero(lr2.coef_))
    
    # Performance on the test set
    score_lr2 = accuracy_score(y_test.flatten(), lr2.predict(X_test))
    print("L1-regularized LR accuracy score on the test set:", score_lr2)
    