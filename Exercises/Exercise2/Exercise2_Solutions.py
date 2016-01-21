'''
Tampere University of Technology
SGN 41006 - Signal Interpretation (Spring 2016)
Exercise 2 Solutions

Contact:    oguzhan.gencoglu@tut.fi (Office: TE406)
            andrei.cramariuc@tut.fi (Office: TE314)
'''

# load required packages
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    '''
    Question 3
    '''

    # initialize an empty list
    data_1 = []

    # open the file
    with open('locationData.csv', 'r') as fp:
        # read from the file one line at a time
        for line in fp:
            # split the line in to numbers
            values = line.split(' ')
            # cast the values to float
            values = map(float, values)
            # add to the end of data_1
            data_1.append(values)

    # cast the list to a numpy array
    data_1 = np.array(data_1)

    # directly read data into a 2D numpy array
    data_2 = np.loadtxt('locationData.csv')

    # compare the two arrays
    print(np.all(data_1 == data_2))

    '''
    Question 4
    '''

    def gaussian(x, mu, sigma):
        return 1/np.sqrt(2*np.pi*sigma**2)*np.exp(-1.0/(2*sigma**2)*(x-mu)**2)
    
    # generate the x axis with a 0.1 interval
    x = np.arange(-5,5,0.1)

    # plot the gaussian
    plt.figure()
    plt.plot(x, gaussian(x, 0, 1))
    plt.show()

    def log_gaussian(x, mu, sigma):
        return np.log(1/np.sqrt(2*np.pi*sigma**2))-1.0/(2*sigma**2)*(x-mu)**2

    # plot the logarithmic gaussian
    plt.figure()
    plt.plot(x, log_gaussian(x, 0, 1))
    plt.show()

    '''
    Question 5
    '''

    # original frequency
    f0 = 0.017
    # noise
    w = np.sqrt(0.25)*np.random.randn(100)
    # sinusoid with the noise
    n = np.arange(0,100)
    x = np.sin(2*np.pi*f0*n) + w
    
    # plot the original and the noised sinusoid
    plt.figure()
    plt.plot(n, x - w, 'b')
    plt.plot(n, x, 'g')
    plt.show()

    # find the original frequency
    scores = []
    frequencies = []

    for f in np.linspace(0,0.5,1000):
        e = np.exp(-2*np.pi*1j*f*n)
        score = np.abs(np.dot(x, e))
        scores.append(score)
        frequencies.append(f)

    # find the frequency that corresponds to the maximum score
    fHat = frequencies[np.argmax(scores)]
    print(fHat)
