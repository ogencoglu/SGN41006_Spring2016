'''
Tampere University of Technology
SGN 41006 - Signal Interpretation (Spring 2016)
Exercise 4 Solutions

Contact:    oguzhan.gencoglu@tut.fi (Office: TE406)
            andrei.cramariuc@tut.fi (Office: TE314)
'''

# load required packages
import numpy as np
import matplotlib.pyplot as plt
import glob
from sklearn.neighbors import KNeighborsClassifier
from sklearn.lda import LDA
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from simplelbp import local_binary_pattern


if __name__ == '__main__':
    
    '''
    Question 4
    '''

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

    '''
    Question 5
    '''
    
    # split train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

    # test the classifiers
    clf_list = [KNeighborsClassifier(), LDA(), SVC()]
    clf_name = ['KNN', 'LDA', 'SVC']

    for clf,name in zip(clf_list, clf_name):
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(name, accuracy_score(y_test, y_pred))
