'''
Tampere University of Technology
SGN 41006 - Signal Interpretation (Spring 2016)
Exercise 3 Solutions

Contact:    oguzhan.gencoglu@tut.fi (Office: TE406)
            andrei.cramariuc@tut.fi (Office: TE314)
'''

# load required packages
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import glob

def generate_signals(f0, sigma):
    # generate clean and noisy signals
    
    n = np.arange(100)
    y = np.concatenate((
                np.zeros(500), # vector of zeros
                np.cos(2 * np.pi * f0 * n),
                np.zeros(300)  # vector of zeros
                ))
    y_n = y + np.sqrt(sigma) * np.random.randn(y.size)
    
    return y, y_n # clean and noisy signals, respectively
    
    
def plots(sig1, sig2, sig3):
    # plot 3 plots in the same figure

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(sig1)
    plt.subplot(3, 1, 2)
    plt.plot(sig2)    
    plt.subplot(3, 1, 3)
    plt.plot(sig3)     
    plt.show()

if __name__ == '__main__':
    
    '''
    Question 2 & 3
    '''
        
    # implement a detector for a signal of known waveform
    s1, s_n1 = generate_signals(0.1, 0.5)
    h1 = np.cos(2 * np.pi * 0.1 * np.arange(100)) # lecture notes slide 21
    det1 = np.convolve(h1, s_n1, 'same')
    plots(s1, s_n1, det1)
    
    # implement a detector for a random signal
    s2, s_n2 = generate_signals(0.03, 0.5)
    h2 = np.exp(-2 * np.pi * 1j * 0.03 * np.arange(100)) # lecture notes slide 25
    det2 = np.abs(np.convolve(h2, s_n2, 'same'))    
    plots(s2, s_n2, det2)
    
    '''
    Question 4 & 5
    '''
    # read all jpg files under a specified directory
    name_list = glob.glob('copper_images/*.jpg')
    print("Names of image files:", name_list)
    
    # extract histogram features and class labels for each image
    X = []
    y = []
    for name in name_list:
        temp = plt.imread(name) # read image
        print("Image shape:", temp.shape)
        histograms = []
        for i in range(temp.shape[-1]): # for each channel in the image
            temp_hist, _ = np.histogram(temp[:,:,i]) 
            histograms.append(temp_hist)
        X.append(np.hstack(histograms))
        c = int(name.split(".")[0][-1]) # string manipulations
        y.append(c)
        
    # conversion to numpy array
    X = np.array(X)
    y = np.array(y)
    print("Shape of training data:", X.shape)
    print("Shape of target vectors:", y.shape)
      
    # train a logistic regression classifier on the histogram features
    classifier = LogisticRegression()
    classifier.fit(X, y) # train the classifier
    
    # predict classes and class probabilities
    print("True classes: ", y)
    print("Predicted classes", classifier.predict(X)) 
    print("Predicted class probabilities:\n", classifier.predict_proba(X))