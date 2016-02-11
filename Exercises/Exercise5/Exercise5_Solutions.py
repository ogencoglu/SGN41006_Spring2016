'''
Tampere University of Technology
SGN 41006 - Signal Interpretation (Spring 2016)
Exercise 5 Solutions

Contact:    oguzhan.gencoglu@tut.fi (Office: TE406)
            andrei.cramariuc@tut.fi (Office: TE314)
'''

# load required packages
import numpy as np
import matplotlib.pyplot as plt
import glob
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from simplelbp import local_binary_pattern
from sklearn.preprocessing import scale
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

if __name__ == '__main__':
    
    '''
    Question 4
    '''
    #------------------- Copy & Paste from Exercise 4 Solutions --------------
    # LBP parameters
    P = 8
    R = 5

    # histograms and corresponding classes
    X = []
    y = []

    # class folders
    class_folders = sorted(glob.glob('GTSRB_subset/*'))
    
    for i,folder in enumerate(class_folders):
        # images in class folder
        name_list = glob.glob(folder+'/*')
        for name in name_list:
            image = plt.imread(name)
            # histogram of lbp
            lbp = local_binary_pattern(image, P, R)
            hist = np.histogram(lbp, bins=range(257))[0]
            X.append(hist)
            # corresponding class
            y.append(i)

    # convert to numpy
    X = np.array(X)
    print(X.shape)
    y = np.array(y)
    #------------------- Copy & Paste from Exercise 4 Solutions --------------

    # zero mean, unity variance normalization of each feature
    X = scale(X)

    # split train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
    
    # hyperparameter tuning
    clf_list = [LogisticRegression(), SVC()]
    clf_name = ['LR', 'SVC']
    penalty_list = ["l1", "l2"]
    
    C_range = np.logspace(-5, 0, 6)
    for clf, name in zip(clf_list, clf_name):
        for C in C_range:
            for penalty in penalty_list:
                clf.C = C
                clf.penalty = penalty
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                score = accuracy_score(y_test, y_pred)
                print(name, C, penalty, score)
                
    '''
    Question 5
    '''
    ensembles = [RandomForestClassifier(n_estimators  = 100), 
                    ExtraTreesClassifier(n_estimators  = 100),
                    AdaBoostClassifier(n_estimators  = 100),
                    GradientBoostingClassifier(n_estimators  = 100)]
    for ens in ensembles:
        ens.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        print(score)