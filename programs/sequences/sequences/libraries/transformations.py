import pandas as pd
import numpy as np
from scipy.stats import linregress


def horizontal_transform(closing_values):
    '''
    Purpose:    Transforms closing values to 'horizontal' position 
                    by subtracting all values from a linear regression line
                First Shifts closing values to start at Zero.
    Input:      Array of (closing) values
    output:     Dictionary:
                    trnasformed closing values
                    linear regreession line
                    slope
                    r_value
                    p_value
                    std_err
    '''
     # Shift closing values downward.  Start at zero and get x values.
    y = closing_values - closing_values[0]
    x = np.arange(y.shape[0])
    # Compute linear regresssion statistics
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    line = intercept + slope * x
    # Transform closings horizontal using regression line
    y -= line
    # Return Dictionary
    d = {'closing': y,
         'linregress': line,
         'slope': slope,
         'r_value': r_value,
         'p_value': p_value,
         'std_err': std_err}
    return d



def create_channels(y):
    '''
    Create 5 channels.
    SO FAR HAVE USED ON FLATTENED CLOSING VALUES.
        Perhaps I should not:
            Use on unflattened values.
            Flatten inside function.
            Return unflattened.
            Or pass 'flattened' as perameter. ????
            
    Should I also be able to call a channel graph?
    
    '''
    x = np.arange(y.shape[0])
    # Find top and bottom channels that are most perpendicular to each other.
    min_distance = 1000 # set min mean variable for iterations
    min_distance_perc = 0
    for perc in np.arange(2, 25, 1):  
        # Get indices and solution of largest and smallest % of numbers in closings
        qty = int(perc * y.shape[0] / 100)
        top = np.argpartition(y, -qty)[-qty:]
        bottom = np.argpartition(y, qty)[:qty]
        # Calculate linear Regression on Top and Bottom Line
        slope, intercept, r_value, p_value, std_err = linregress(top, y[top])
        top_line = intercept + slope * x
        slope, intercept, r_value, p_value, std_err = linregress(top, y[bottom])
        bottom_line = intercept + slope * x
        # Bring lines together
        mid = int(y.shape[0] / 2)
        top_lowered = top_line - (top_line[mid] - bottom_line[mid])
        bottom_raised = (top_line[mid] - bottom_line[mid]) + bottom_line
        # differences in lines.  Collect smallest
        if abs(top_line[0] - bottom_raised[0]) < min_distance:
            min_distance = abs(top_line[0] - bottom_raised[0])
            min_distance_perc = perc

    # Get indices and solution of largest and smallest % of numbers in closings
    qty = int(min_distance_perc * y.shape[0] / 100)
    top = np.argpartition(y, -qty)[-qty:]
    bottom = np.argpartition(y, qty)[:qty]
    # Calculate linear Regression on Top and Bottom Line
    slope, intercept, r_value, p_value, std_err = linregress(top, y[top])
    top_line = intercept + slope * x
    slope, intercept, r_value, p_value, std_err = linregress(top, y[bottom])
    bottom_line = intercept + slope * x
    # Bring lines together
    mid = int(y.shape[0] / 2)
    top_lowered = top_line - (top_line[mid] - bottom_line[mid])
    bottom_raised = (top_line[mid] - bottom_line[mid]) + bottom_line
    # Calculate average top and bottom lines to form channel
    channel_top          = (top_line + bottom_raised) / 2
    channel_bottom       = (bottom_line + top_lowered) / 2
    channel_mid          = (channel_top + channel_bottom) / 2
    # Calculate Outer channels (somehow - not sure yet)
    outer_top    = y[np.argmax(y)] - channel_top[np.argmax(y)]
    outer_bottom = y[np.argmin(y)] - channel_bottom[np.argmin(y)]
    outer = (abs(outer_top) + abs(outer_bottom)) / 2
    channel_top_outer = channel_top + outer
    channel_bottom_outer = channel_bottom - outer
    d = {'c1': channel_bottom_outer,
         'c2': channel_bottom,
         'c3': channel_mid,
         'c4': channel_top,
         'c5': channel_top_outer,
         'degree': min_distance_perc,
         'slope': slope,
         'range': channel_top[-1] - channel_bottom[-1]}
    '''
    # Very Good plots
    plt.figure()
    plt.plot((channel_bottom + hor['linregress']) + closings[0])
    plt.plot((channel_top + hor['linregress']) + closings[0])
    plt.plot(closings)
    plt.figure()
    plt.plot(channel_top)
    plt.plot(channel_bottom)
    plt.plot(channel_mid)
    plt.plot(channel_top_outer)
    plt.plot(channel_bottom_outer)
    plt.plot(y)
    '''
    return d



if __name__ == '__main__':
    pass






