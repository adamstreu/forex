import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks_cwt
from scipy.stats import linregress
from sklearn.preprocessing import MinMaxScaler as minmaxscaler
from sklearn.preprocessing import StandardScaler as standardscaler

def get_groups(values, interval, plot=False):
    keep = []
    keep.append(values[0])
    for i in range(values.shape[0] - 1):
        if values[i + 1] > values[i] + interval:
            keep.append(values[i + 1])
    return np.array(keep)
      
    # Plot groups as colored dots in index range
    if plot:
        total = np.arange(values.min(), values.max())
        plt.figure()
        plt.plot(total, [np.nan] * total.shape[0])
        plt.plot(values, np.zeros(values.shape[0]), 'o', color='blue' )
        plt.plot(values[keep], np.ones(keep.shape[0]), 'o', color='red')
        plt.show()


def pips_walk(values, step = .00001):
    '''
    Converts (closing) values (as array) form a function based off of time to 
    one based off of a series of discrete same valued moves.
    '''
    walk = [values[0]]
    value_differences = values[1:] - values[:-1]
    value_differences /= step
    value_differences = value_differences.astype(int)
    for diff in value_differences:
        for i in range(abs(diff)):
            walk.append(walk[-1] + (step * np.sign(diff)))
    return np.array(walk)


if __name__ == '__main__':
    pass



    '''
    # Pickle model
    ###############################################################################
    pkl_filename = '/Users/user/Desktop/mod.pkl'
    with open(pkl_filename, 'wb') as file:  
        pickle.dump(mod, file)
    '''


