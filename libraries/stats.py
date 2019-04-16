import pandas as pd
import numpy as np
from scipy.stats import rv_discrete
import os; os.chdir('/northbend')
from libraries.transformations import get_groups


def get_distribution_boundary(values, bound, rolling_window=15):
    
    '''
    Create Distribution of given values.
    For specified cdf %, return :
        upper and lower boundary values.
        upper and lower index of values gt and lt boundary values
    '''
    
    # Create distribution
    perc = np.ones(values.shape[0]) / values.shape[0]
    dist = rv_discrete(values=(values, perc))
    
    # Get bounding of distribution
    upper_bound = dist.ppf(1 - bound)
    lower_bound = dist.ppf(bound)
    
    # Get index where values are past bounds (grouped)
    upper_index = np.arange(values.shape[0])[values > upper_bound]
    lower_index = np.arange(values.shape[0])[values < lower_bound]

    return {
            'upper_index': upper_index,
            'upper_bound': upper_bound,
            'lower_index': lower_index,
            'lower_bound': lower_bound
            }    
