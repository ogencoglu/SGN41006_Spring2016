'''
Tampere University of Technology
SGN 41006 - Signal Interpretation (Spring 2016)
Exercise 1 Solutions

Contact:    oguzhan.gencoglu@tut.fi (Office: TE406)
            andrei.cramariuc@tut.fi (Office: TE314)
'''

# load required packages
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import imread
from scipy.ndimage.morphology import white_tophat
from scipy.io import loadmat

if __name__ == '__main__':
    
    '''
    Question 1 & 2
    '''
    # read data into a 2D numpy array
    data = np.loadtxt('locationData.csv')
    
    # print shape
    print(data.shape)
    
    # plot first two columns
    plt.figure()
    plt.plot(data[:,0], data[:,1])
    plt.show()
    
    # plot all columns in 3D
    plt.figure()
    ax = plt.subplot(1, 1, 1, projection = '3d')
    plt.plot(data[:,0], data[:,1], data[:,2])
    ax.set_xlabel('First Column')
    ax.set_ylabel('Second Column')
    ax.set_zlabel('Third Column')
    plt.show()
    
    '''
    Question 3
    '''
    # read image as numpy array
    im = imread('oulu.jpg')
    
    # show image
    plt.figure()
    plt.imshow(im)
    
    # check type and shape
    print(type(im))
    print(im.shape)
    
    # mean of all image
    print(np.mean(im))

    # mean of each channel (RGB)
    print(np.mean(im, axis = tuple((0,1))))
    
    # apply white tophead transform
    plt.figure()
    plt.imshow(white_tophat(im, size = 10))
    plt.show()   
    
    '''
    Question 4 & 5
    '''
    # read the data
    mat = loadmat("twoClassData.mat")
    print(mat.keys())
    X = mat['X']
    y = mat['y'].ravel()
    
    # plot two class data
    plt.figure()
    plt.plot(X[y == 0, 0], X[y == 0, 1], 'ro')
    plt.plot(X[y == 1, 0], X[y == 1, 1], 'bo')


    # define scaling function
    def normalize_data(array):
        # column-wise zero mean, unity variance normalization
        
        return(array - array.mean(axis = 0)) / array.std(axis = 0)
    
    
    # apply the function    
    X_norm = normalize_data(X)
    print(np.mean(X_norm, axis = 0)) # should be 0s (or very small values)
    print(np.std(X_norm, axis = 0)) # should be 1s