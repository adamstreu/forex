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
from libraries.stats           import autocorrelation
from sklearn.preprocessing     import MinMaxScaler

'''
Runs through some cycles.
Just let it be this.
        
Some oher things to look for - 
positioned for breakout over different granulariteis / time periods

run this inline - get to see the graphs go by.


Try:
    Finding a spike.
    When the weighted volume ( channeled )turns against it - you do too.
    Or - when they move in sink - maybe that....
    
    ok.  what excites me so far.  Is seeing a large spike up in price.
    Then some time laer a large drop in weighted price.
    It appears casually that a price drop off follows.

'''

fit_avg = 0    
for s in [1, 2, 3, 4]:
    

    # Window
    window = 7000
    start  = 160000 + s  * 6600
    outcome_interval = window 
    values = candles.loc[start - window + 1: start , 'midclose'].values
    volume = candles.loc[start - window + 1: start , 'volume'].values
    # values = candles_5.loc[start / 5  - window + 1: start / 5 , 'midclose'].values
    
    # Supports
    support_interval = window * 4
    support_bins = 50
    
    # Outcomes
    outcome_values = candles.loc[start: start + outcome_interval, 'midclose'].values
    # outcome_values = candles_5.loc[start / 5 : start / 5 + outcome_interval, 'midclose'].values
    
    # Get Channels
    channel = Channel(values)
    outcome_channel = Channel(outcome_values)
    
    # get period guess
    corr                 = autocorrelation(channel.scaled)
    corr_orig            = corr.copy()
    margin               = int(window * .10) 
    corr                 = corr[margin:-margin]
    maximum              = corr[:, 1].argmax()
    minimum              = corr[:, 1].argmin()
    corr_period          = min(int(window * .75), 2 * abs(corr[maximum, 0] - corr[minimum, 0]))
    
    '''
    corr_peaks  = np.arange((left[:-1] & right[1:]).shape[0])[left[:-1] & right[1:]]
    if corr_peaks.shape[0] == 1:
        corr_period = corr_peaks[0]
    else:
        corr_period = int((corr_peaks[1:] - corr_peaks[:-1]).mean())
    '''
    
    
    # Get corr peaks ( maybe good indicator of 'smoothness'
    smoothness = int(window  * .10)
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

    
    # Set guesses and get wave
    amplitude            = (channel.c7[0] - channel.c1[0]) / 2
    frequency_guess      = window / corr_period  
    phase_shift_guess    = - np.argmax(channel.scaled < channel.c1)
    vertical_shift_guess = amplitude + channel.c1[0]
    t = np.linspace(0, 2*np.pi, channel.scaled.shape[0])
    data = channel.scaled 
    # data_first_guess = amplitude * np.sin(frequency_guess * t + phase_shift_guess) + vertical_shift_guess
    optimize_func = lambda x: amplitude * np.sin(x[0] * t + x[1]) + x[2]  - data
    est_frequency, est_phase_shift, est_vertical_shift = \
            leastsq(optimize_func, [frequency_guess, phase_shift_guess, vertical_shift_guess], full_output=True)[0]
    wave = amplitude * np.sin(est_frequency * t + est_phase_shift) + est_vertical_shift 
    # Get Fit Error
    fit = ((wave - data) ** 2).mean()
    fit_avg += fit
    
    # Get Supports
    support_values = candles.loc[start - support_interval:start, 'midclose'].values
    supports  = channel.get_supports(support_values)
    strengths = supports['strengths']
    supports  = supports['supports']
    support_by_channel = (supports - channel.closings_c1[-1]) / (channel.closings_c7[-1] - channel.closings_c1[-1]).tolist()
    
    
    '''
    if len(support_by_channel) == 0:
        support_by_channel = [0, 0, 0]
    elif len(support_by_channel) == 1:
        support_by_channel = [support_by_channel[0], support_by_channel[0], support_by_channel[0]]
    elif len(support_by_channel) == 2:
        support_by_channel = [support_by_channel[0], support_by_channel[1], support_by_channel[1]]
    elif len(support_by_channel) >= 4:
        support_by_channel = support_by_channel[: 3]
    '''
    
    
    # Stretch wave - put back into regular closing position
    wave_stretched = channel.scaler.inverse_transform(wave.reshape(-1, 1)).ravel()
    
    


    
    # Adding in the weighted stuff   -   Here only for now 
    ###########################################################################
    weighted_closing = ((values[1:] - values[:-1]) / volume[1:]).cumsum()
    w_channel = Channel(weighted_closing)
    # Weird graph
    weighted_weird = (values[1:] - values[:-1]).cumsum() / volume[1:]
    ###########################################################################

    
    
    
    
    
    
    
    
    
    
    # Plot Result
    ###########################################################################
    fig, ax = plt.subplots(3, 2, figsize=(14, 12))
    x1 = np.arange(channel.scaled.shape[0])
    x2 = np.arange(outcome_channel.closings.shape[0]) + x1[-1]
    # Plot 1.  Channels and wave fit for closing and weighed closings
    ax[0, 0].plot(x1, channel.scaled, color='blue')
    ax[0, 0].plot(x1, channel.c1, color='black')
    ax[0, 0].plot(x1, channel.c7, color='black')
    ax[0, 0].plot(w_channel.scaled, color='grey')
    ax[0, 0].plot(x1, wave, color='orange')
    # Plot 2
    ax[1, 0].plot(corr_orig[:, 1])
    ax[1, 0].plot(np.arange(margin, corr[:, 1].shape[0] + margin), corr[:, 1])
    ax[1, 0].plot(corr_smoothed)
    # Plot 3
    ax[2, 0].plot(x1, wave_stretched + channel.regression_line, color='orange')
    ax[2, 0].plot(x1[1:], w_channel.closings + channel.closings[0], color='darkgrey', linewidth=5)
    ax[2, 0].plot(x1, channel.closings)
    ax[2, 0].plot(x1, channel.closings_c1, color='black')
    ax[2, 0].plot(x1, channel.closings_c7, color='black')
    ax[2, 0].plot(x2, outcome_channel.closings, color = 'green')
    ax[2, 0].plot(x2, np.ones(x2.shape[0]) * channel.closings_c1[-1], color='black')
    ax[2, 0].plot(x2, np.ones(x2.shape[0]) * channel.closings_c7[-1], color='black')
    for i in range(supports.shape[0]):
        ax[2, 0].plot(np.ones(x1.shape[0] + x2.shape[0]) * supports[i], label = str(round(strengths[i], 3)))
    # Plot 4 - Volume and weighted weird
    ax[0, 1].plot(MinMaxScaler().fit_transform(weighted_weird.reshape(-1, 1)), color = 'orange', label='weird')
    ax[0, 1].plot(MinMaxScaler().fit_transform(volume.reshape(-1, 1)), color = 'blue', label='volume')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
   
    
    # Print stats 
    ###########################################################################
    print('------------- ' + str(s + 1) + ' ----------------\n' + 
          str(round(corr_period, 0)) + '\t' + 
          str(round(est_frequency, 4)) + '\t' + 
          str(round(channel.closings_c7[0] - channel.closings_c1[0], 4)) + '\t' + 
          str(round(fit, 4)) + '\t' + 
          str(auto_peaks) + '\n')#str(leastsq(optimize_func, [frequency_guess, phase_shift_guess, vertical_shift_guess], full_output=True)[2]['qtf']) + '\n')
    
    
    weighted_distance = 0
    for i in range(supports.shape[0]):
        weighted_distance += ((support_by_channel[i] - channel.closing_position) * strengths[i])
        # print('{} \t {}'.format(support_by_channel[i], strengths[i]))
    print(s, weighted_distance)
    print()

          
    
    
    

    
    
    
    
    
    
    

print('Fit Average: {}'.format(fit_avg / (s + 1)))


