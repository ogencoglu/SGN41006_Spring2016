# -*- coding: utf-8 -*-
"""
Tampere University of Technology
SGN 41006 - Signal Interpretation (Spring 2016)
Exercise 7 Solutions - Question 3

Contact:    oguzhan.gencoglu@tut.fi (Office: TE406)
            andrei.cramariuc@tut.fi (Office: TE314)
"""

# load required packages
import numpy as np
import matplotlib.pyplot as plt
import pickle


def L2_regularized_log_loss(w, X, y, C):
    """ 
    Computes the L2-regularized log-loss function at w. The 
    computation uses the data in X with
    corresponding labels in y. 
    """
    
    L = 0 # Accumulate loss terms here.
        
    # Process each sample in X:
    for n in range(X.shape[0]):
        L += np.log(1 + np.exp(y[n] * np.dot(w, X[n])))
    
    # regularization term (L2)    
    L += C * np.sum(w**2)
    
    return L
    
    
def grad(w, X, y, C):
    """ 
    Computes the gradient of the L2-regularized log-loss function
    at w. The computation uses the data in X with
    corresponding labels in y. 
    """
        
    G = 0 # Accumulate gradient here.
    
    # Process each sample in X:
    for n in range(X.shape[0]):
        
        numerator = np.exp(y[n] * np.dot(w, X[n])) * y[n] * X[n]
        denominator = 1 + np.exp(y[n] * np.dot(w, X[n]))
        
        G += numerator / denominator
        
    # regularization term (L2) 
    G += 2 * C * w
    
    return G
    
    
if __name__ == "__main__":
        
    # Add your code here:
        
    # 1) Load X and y from pickle.        
    with open("log_loss_data.pkl", "r") as f:
        data = pickle.load(f)
    X = data["X"]
    y = data["y"]
    print(X.shape, y.shape)
    
    # 2) Initialize w at random: w = np.random.randn(2)
    w = np.random.randn(2)
    
    # 3) Set step_size to a small positive value.
    step = 1e-4
    
    # regularization coefficient
    C = 1.0
    
    # 4) Initialize empty lists for storing the path and
    # accuracies: W = []; accuracies = []
    W = []
    accuracies = []
    
    for iteration in range(200):

        # 5) Apply the gradient descent rule.
        w = w - step * grad(w, X, y, C)

        # 6) Print the current state.
        print ("Iteration %d: w = %s (log-loss = %.2f)" % \
              (iteration, str(w), L2_regularized_log_loss(w, X, y, C)))
        
        # 7) Compute the accuracy:
        scores = np.dot(X, w)
        yHat = (-1)**(scores > 0)
        accuracy = np.mean(yHat == y)
        accuracies.append(accuracy)
        
        W.append(w)
    
    # 8) Below is a template for plotting. Feel free to 
    # rewrite if you prefer different style.
    
    W = np.array(W)
    
    plt.figure(figsize = [5,5])
    plt.subplot(211)
    plt.plot(W[:,0], W[:,1], 'ro-')
    plt.xlabel('w$_0$')
    plt.ylabel('w$_1$')
    plt.title('Optimization path')
    
    plt.subplot(212)
    plt.plot(100.0 * np.array(accuracies), linewidth = 2)
    plt.ylabel('Accuracy / %')
    plt.xlabel('Iteration')
    plt.tight_layout()
    plt.savefig("log_loss_minimization.pdf", bbox_inches = "tight")
    
