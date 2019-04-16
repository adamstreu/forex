import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import os; os.chdir('/forex')
from libraries.oanda           import get_candles
from classes.channel           import Channel
from libraries.outcomes        import simple_outcomes


# Request and Calculate candles, Results, outcomes
###############################################################################



peaks = []
for i in range(window, candles.shape[0] - search_interval):
    # Print progress.
    if i % 10000 == 0:  
        print('Peaks Percent complete: {:.2f}'.format(i / candles.shape[0]))
    

    closings = candles.loc[i - window + 1: i,  'midclose']
    hist = np.histogram(closings, bins = 10)
    x = (hist[1] + ((hist[1][1] - hist[1][0]) / 2))[:-1]
    
    
    peaks_index = []
    # Get the middle peaks
    left = (hist[0][1:] > hist[0][:-1])  # Don't use first
    right = hist[0][:-1] > hist[0][1:]
    peaks_index = left[:-1] & right[1:]
    # get the left and right peaks - careful with this
    if hist[0][0] > hist[0][1]:
        peaks_index = np.insert( peaks_index, 0, True)
    else:
        peaks_index = np.insert( peaks_index, 0, False)
    if hist[0][-1] > hist[0][-2]:
        peaks_index = np.insert( peaks_index, peaks_index.shape[0], True)
    else:
        peaks_index = np.insert( peaks_index, peaks_index.shape[0], False)    
    
    # Get closing Support (peak ) values
    peaks = x[peaks_index]