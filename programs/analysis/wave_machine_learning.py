import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot
from scipy.optimize import leastsq

import os; os.chdir('/northbend')
from libraries.oanda           import get_candles
from classes.channel           import Channel
from libraries.outcomes        import outcomes
from libraries.transformations import pips_walk
from libraries.transformations import autocorrelation

import sklearn 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC as svc


'''
Trying ML with new wave function data.


Distance using percenage of channel range


'''



def get_results_bars(candles, window, search_interval, peaks_window, distance):
    # Instantiation
    results     = []
    long_target = []
    long_loss   = []
    short_target = []
    short_loss  = []
    peaks = []
    start = max(window, peaks_window)
    for i in range(start, candles.shape[0] - search_interval, int(window / 2)):
        # Print progress.
        if i % 10000 == 0:  
            print('Percent complete: {:.2f}'.format(i / candles.shape[0]))
        
        # Fetch channel transformation on window. 
        values = candles.loc[i - window + 1: i, 'midclose'].values
        channel = Channel(values)
        
        # Get Wave information
        corr                 = autocorrelation(channel.scaled)
        margin               = int(window * .10) 
        corr                 = corr[margin:-margin]
        maximum              = corr[:, 1].argmax()
        minimum              = corr[:, 1].argmin()
        corr_period          = min(int(window * .75), 2 * abs(corr[maximum, 0] - corr[minimum, 0]))
        
        # Get corr peaks ( maybe good indicator of 'smoothness'
        smoothness = int(window  * .25)
        corr_smoothed        = corr[:, 1]
        corr_smoothed        = pd.DataFrame(corr_smoothed).rolling(smoothness).mean().values.ravel()
        corr_smoothed        = corr_smoothed[smoothness:]
        left = (corr_smoothed[1:] > corr_smoothed[:-1])
        right = (corr_smoothed[1:] < corr_smoothed[:-1])
        auto_peaks    = (left[:-1] & right[1:]).sum()
        if corr_smoothed[0] > corr_smoothed[1]:
            auto_peaks += 1
        if corr_smoothed[-1] > corr_smoothed[-2]:
            auto_peaks += 1

        
        # Assign first guesses for wave
        amplitude            = (channel.c7[0] - channel.c1[0]) / 2
        frequency_guess      = window / corr_period  
        phase_shift_guess    = - np.argmax(channel.scaled < channel.c1)
        vertical_shift_guess = amplitude + channel.c1[0]
        # Ge Real Wave
        t = np.linspace(0, 2*np.pi, channel.scaled.shape[0])
        optimize_func = lambda x: amplitude * np.sin(x[0] * t + x[1]) + x[2]  - channel.scaled
        est_frequency, est_phase_shift, est_vertical_shift = \
                leastsq(optimize_func, [frequency_guess, phase_shift_guess, vertical_shift_guess], full_output=True)[0]
        wave_parameter_fits = leastsq(optimize_func, [frequency_guess, phase_shift_guess, vertical_shift_guess], full_output=True)[2]['qtf']
        # assess fit
        wave = amplitude * np.sin(est_frequency * t + est_phase_shift) + est_vertical_shift
        fit = ((wave - channel.scaled) ** 2).mean()
        
        # Get Supports
        support_interval = window * 2
        support_values = candles.loc[start - support_interval: start, 'midclose'].values
        supports = channel.get_supports(support_values)
        support_by_channel = (supports - channel.closings_c1[-1]) / (channel.closings_c7[-1] - channel.closings_c1[-1]).tolist()
        if len(support_by_channel) == 0:
            support_by_channel = [0, 0, 0]
        elif len(support_by_channel) == 1:
            support_by_channel = [support_by_channel[0], support_by_channel[0], support_by_channel[0]]
        elif len(support_by_channel) == 2:
            support_by_channel = [support_by_channel[0], support_by_channel[1], support_by_channel[1]]
        elif len(support_by_channel) >= 4:
            support_by_channel = support_by_channel[: 3]
        
        # Build up results
        results.append([i,
                        channel.channel_slope,
                        channel.closings_slope,
                        channel.closing_position,
                        channel.channel_range,
                        channel.channel_degree,
                        channel.linear_p_value,
                        channel.largest_spike,
                        channel.largest_spike_5,
                        channel.within_range,
                        candles.loc[i, 'spread'],
                        candles.loc[i, 'volume'],
                        channel.c1[-1],
                        channel.c7[-1],
                        wave[-1],
                        wave[-1] - wave[-2],
                        fit,
                        amplitude,
                        est_frequency, 
                        est_phase_shift, 
                        est_vertical_shift,
                        wave_parameter_fits[0],
                        wave_parameter_fits[1],
                        wave_parameter_fits[2],
                        support_by_channel[0],
                        support_by_channel[1],
                        support_by_channel[2],   
                        corr[:, 1].max(),
                        corr[:, 1].mean(),
                        auto_peaks
                        ])
    
        # Get Peaks
        peaks_collection = channel.get_supports(peaks_window)
        for peak in peaks_collection:
            peaks.append([i, peak])
            
        # Set distance for outcome
        if type(distance) == str:
            distance = np.array([.25, .5, .75, 1, 1.25, 1.5, 2]) * (channel.channel_range)
        # Get long outcomes
        outs = outcomes('long', candles, i, search_interval, distance, False)
        long_target.append([i] + outs['target'])
        long_loss.append([i] + outs['loss'])
        # get short outcomes
        outs = outcomes('short', candles, i, search_interval, distance, False)
        short_target.append([i] + outs['target'])
        short_loss.append([i] + outs['loss'])
    
    # Assemble Dataframes
    results_columns = ['location',
                       'channel_slope', 
                       'closings_slope',
                       'channel_closing_position',
                       'channel_range',
                       'channel_degree',
                       'linear_p_value',
                       'largest_spike',
                       'largest_spike_5',
                       'within_range',
                       'spread',
                       'volume',
                       'c1',
                       'c7',
                       'wave_position',
                       'wave_tangent',
                       'wave_fit',
                       'amplitude',
                       'frequency',
                       'phase_shift',
                       'vertical_shift',
                       'frequency_fit', 
                       'phase_fit',
                       'vert_fit',
                       'support_0',
                       'support_1',
                       'support_2',
                       'auto_max',
                       'auto_mean',
                       'auto_peaks'                      
                       ]
    
    
    results      = pd.DataFrame(np.array(results), columns = results_columns)
    long_target  = pd.DataFrame(np.array(long_target))
    long_loss    = pd.DataFrame(np.array(long_loss))
    short_target = pd.DataFrame(np.array(short_target))
    short_loss   = pd.DataFrame(np.array(short_loss))
    peaks        = pd.DataFrame(np.array(peaks), columns=['location', 'peaks'])
    # Set indexes
    results      = results.set_index('location', drop=True)
    long_target  = long_target.set_index(0, drop=True)
    long_loss    = long_loss.set_index(0, drop=True)
    short_target = short_target.set_index(0, drop=True)
    short_loss   = short_loss.set_index(0, drop=True)   
    # Correct Indexes
    long_target.index  = long_target.index.rename('location')
    long_loss.index    = long_loss.index.rename('location')
    short_target.index = short_target.index.rename('location')
    short_loss.index   = short_loss.index.rename('location')
    # Set index type
    results.index      = results.index.astype(int)
    long_target.index  = long_target.index.astype(int)
    long_loss.index    = long_loss.index.astype(int)
    short_target.index = short_target.index.astype(int)
    short_loss.index   = short_loss.index.astype(int)
    # Return
    return {'results': results, 
            'long_target': long_target,
            'long_loss': long_loss, 
            'short_target': short_target, 
            'short_loss': short_loss, 
            'peaks': peaks}











###############################################################################
# Request and Calculate candles, Results, outcomes, peaks
###############################################################################
if False:
    # Candles
    instrument      = 'EUR_USD'
    granularity     = 'M1'
    _from           = '2010-01-01T00:00:00Z'
    _to             = '2018-01-01T00:00:00Z'
    # Outcomes
    distance        = 'relative'# np.arange(1, 11) * .0005 # or 'relative'
    # Windows 
    window          = 500
    search_interval = window * 2
    # Peaks
    peaks_window    = 500
    
    # Fetch candles, Results and Outcomes
    candles      = get_candles(instrument, granularity, _from, _to)
    get_results  = get_results_bars(candles, window, search_interval, 
                                    peaks_window, distance)
    long_target  = get_results['long_target']
    long_loss    = get_results['long_loss']
    short_target = get_results['short_target']
    short_loss   = get_results['short_loss']
    peaks        = get_results['peaks']
    results      = get_results['results']




'''
Let's try long positions first

# Print
weights = sklearn.utils.class_weight.compute_class_weight('balanced', 
                                                          np.array([0,1]), 
                                                          y_train)
logreg = svc()
logreg = KNeighborsClassifier(n_neighbors=2)
logreg = DecisionTreeClassifier(random_state=0)
logreg = MLPClassifier(solver='lbfgs', alpha=1e-5, 
                               hidden_layer_sizes=(500, 4), random_state=1)
logreg = LogisticRegression(class_weight={0: weights[0], 1: weights[1]})


'''

for t in range(1, long_target.columns.shape[0] + 1):
    for l in range(1, long_target.columns.shape[0] + 1):
    
        print('------  {}, {}  ------'.format(t, l))
        x = results.copy()
        y = (short_target.loc[:, t] < short_loss.loc[:, l]).astype(int)
        # create sets
        x.reset_index(inplace=True, drop=True)
        y.index = x.index
        row = int(x.shape[0] * .8)
        x_train = x.loc[: row]
        x_test  = x.loc[row :]
        y_train = y.loc[: row]
        y_test  = y.loc[row :]
        # scale values
        scaler  = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test  = scaler.transform(x_test)
        # model
        logreg = LogisticRegression()
        logreg.fit(x_train, y_train)
        predictions = logreg.predict(x_test)
        print(classification_report(y_test, predictions))
        # plt.barh(results.columns, logreg.coef_.ravel())
        
    
