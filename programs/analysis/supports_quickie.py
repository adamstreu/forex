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


'''
supports only
keep it simple
        
'''

def get_supports(closings, bins=10, plot = False):
    hist = np.histogram(closings, bins = bins)
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
    # Peaks 
    supports    = x[peaks_index]
    strengths = hist[0][peaks_index] / hist[0].sum()
    # Verification Plotting
    if plot:
        plt.figure(figsize = (14, 3))
        plt.plot(x, hist[0])
        for each in peaks:
            plt.plot([each, each], [0, hist[0].max()], color='red')   
    # Return
    return {'supports': supports, 'strengths': strengths}



for i in range(20):
    
    # Window
    window            = 700
    start             = 123600 + i * 3300
    # Outcomes
    outcomes_interval = window
    # Supports
    bins = 30
    supports_interval = window * 2
    
    # Candle Values
    values   = candles.loc[start - window + 1: start ,        'midclose'].values
    supports = candles.loc[start - supports_interval: start , 'midclose'].values
    outcomes = candles.loc[start: start + outcomes_interval , 'midclose'].values
    
    # Get Supports
    supports  = get_supports(supports)
    strengths = supports['strengths']
    supports  = supports['supports'] 
    
    
    # Plot total
    plt.figure()
    x1 = np.arange(values.shape[0])
    x2 = np.arange(outcomes.shape[0]) + x1[-1]
    plt.plot(values)
    for i in range(supports.shape[0]):
        plt.plot(np.ones(x2[-1]) * supports[i], label=strengths[i])
    plt.legend()
    plt.tight_layout()
    plt.plot(x2, outcomes)
    










"""
fit_avg = 0    
for s in range(1):
    

    # Window
    window = 1500
    start  = 123600# + s * 10000
    outcome_interval = window
    values = candles.loc[start - window + 1: start , 'midclose'].values
    # Supports
    support_interval = window * 2
    support_bins = 50
    # Outcomes
    outcome_values = candles.loc[start: start + outcome_interval, 'midclose'].values
    
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
    
    # Stretch wave - ploting
    wave_stretched = channel.scaler.inverse_transform(wave.reshape(-1, 1)).ravel()
    
    # Plot Result
    x1 = np.arange(channel.scaled.shape[0])
    x2 = np.arange(outcome_channel.closings.shape[0]) + x1[-1]
    fig, ax = plt.subplots(3, 1, figsize=(8, 8))
    ax[0].set_title(str(pd.to_datetime(candles.loc[start, 'timestamp']).weekday()) + '    ' + 
                    str(pd.to_datetime(candles.loc[start, 'timestamp']).hour) +  '     ' + 
                    str(fit))
    ax[0].plot(x1, channel.scaled, color='blue')
    ax[0].plot(x1, channel.c1, color='black')
    ax[0].plot(x1, channel.c7, color='black')
#    for each in support_by_channel:
#        ax[0].plot(x1, np.ones(x1.shape[0]) * each, color='lightgrey')
    #ax[0].plot(x1, data_first_guess, color='orange')
    ax[0].plot(x1, wave, color='orange')
    ax[1].plot(corr_orig[:, 1])
    ax[1].plot(np.arange(margin, corr[:, 1].shape[0] + margin), corr[:, 1])
    ax[1].plot(corr_smoothed)
    ax[2].set_title('Closings position / slope:   {:.2f} / {}'.format(channel.closing_position, channel.closings_slope))
    ax[2].plot(x1, wave_stretched + channel.regression_line, color='orange')
    ax[2].plot(x1, channel.closings)
    ax[2].plot(x1, channel.closings_c1, color='black')
    ax[2].plot(x1, channel.closings_c7, color='black')
    ax[2].plot(x2, outcome_channel.closings, color = 'green')
    ax[2].plot(x2, np.ones(x2.shape[0]) * channel.closings_c1[-1], color='black')
    ax[2].plot(x2, np.ones(x2.shape[0]) * channel.closings_c7[-1], color='black')
    for i in range(supports.shape[0]):
        ax[2].plot(np.ones(x1.shape[0] + x2.shape[0]) * supports[i], label = str(round(strengths[i], 3)))
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    '''
    print(str(s + 1) + '\t' + 
          str(round(corr_period, 0)) + '\t' + 
          str(round(est_frequency, 4)) + '\t' + 
          str(round(channel.closings_c7[0] - channel.closings_c1[0], 4)) + '\t' + 
          str(round(fit, 4)) + '\t' + 
          str(auto_peaks) + '\n')#str(leastsq(optimize_func, [frequency_guess, phase_shift_guess, vertical_shift_guess], full_output=True)[2]['qtf']) + '\n')
    '''
    
    #print('{} \t {}'.format(channel.closing_position, channel.closings_slope * 100000))
    weighted_distance = 0
    for i in range(supports.shape[0]):
        weighted_distance += ((support_by_channel[i] - channel.closing_position) * strengths[i])
        # print('{} \t {}'.format(support_by_channel[i], strengths[i]))
    print(s, weighted_distance)
    print()

          
    
    
    

print('Fit Average: {}'.format(fit_avg / (s + 1)))
"""

