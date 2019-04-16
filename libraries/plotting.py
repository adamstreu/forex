import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from scipy.signal import find_peaks_cwt
#from scipy.stats import linregress
#from sklearn.preprocessing import MinMaxScaler as minmaxscaler
#from sklearn.preprocessing import StandardScaler as standardscaler
#import os; os.chdir('/northbend')
#from libraries.transformations import create_channels
#from libraries.transformations import horizontal_transform
#from libraries.transformations import get_groups


def plot_index_distribution(values, title = 'Distibution of Index', figsize=(12, 2)):
    
    # Create Values and Interval
    start = values.min()
    end = values.max()
    x = np.arange(start, end)
    y = np.zeros(x.shape[0])
    y[np.isin(x, values)] = 1
    
    #y = ['nan' if float(x) == 0 else x for x in y]
    # Plot 
    plt.figure(figsize=figsize)
    plt.plot(y, 'o')
    plt.title(title)
    plt.show()



def plot_channels(df, start, window_length, outcome_widths = 0,
                  peak_interval = 1000):
    # Define closing values
    df_window = df.loc[start - window_length: start].copy()
    closing_values = df_window.midclose.values
    # Define outcome values
    outcome_values = df.loc[start: start + (window_length * 1), 'midclose'].values
    # get x values for outcome candles
    x2 = np.arange(outcome_values.shape[0]) + np.arange(closing_values.shape[0])[-1]
    # Flatten closing_values    
    closings_flat = horizontal_transform(closing_values)
    # Rescale closing Values
    mms = minmaxscaler()
    mms.fit(closings_flat['closing'].reshape(-1, 1))
    scaled = mms.transform(closings_flat['closing'].reshape(-1, 1)).ravel()
    # Create Channels
    channels = create_channels(scaled)
    # get closing peaks
    closing_peaks = find_peaks_cwt(closing_values, np.arange(1, peak_interval))
    scaled_peaks_high = find_peaks_cwt(scaled, np.arange(1, peak_interval))
    scaled_peaks_low = find_peaks_cwt(-scaled, np.arange(1, peak_interval))
    # get histogram and it's peaks
    hist = np.histogram(scaled, bins=10)
    hist_peaks = find_peaks_cwt(hist[0], np.arange(1, 10))
    # Try my own peaks
    if hist[0][0] > hist[0][1]:
        keep = [True]
    else:
        keep = [False]
    for i in range(1, hist[0].shape[0]-1):
        if hist[0][i] > hist[0][i+1] and hist[0][i] > hist[0][i-1]:
            keep.append(True)
        else:
            keep.append(False)
    if hist[0][-1] > hist[0][-2]:
        keep.append(True)
    else:
        keep.append(False)   
    keep = np.array(keep)
    
    
    # Unscale and unflatten channels
    c1 = mms.inverse_transform(channels['c1'].reshape(-1, 1)).ravel()
    c2 = mms.inverse_transform(channels['c2'].reshape(-1, 1)).ravel()
    c3 = mms.inverse_transform(channels['c3'].reshape(-1, 1)).ravel()
    c4 = mms.inverse_transform(channels['c4'].reshape(-1, 1)).ravel()
    c5 = mms.inverse_transform(channels['c5'].reshape(-1, 1)).ravel()
    c6 = mms.inverse_transform(channels['c6'].reshape(-1, 1)).ravel()
    c7 = mms.inverse_transform(channels['c7'].reshape(-1, 1)).ravel()


    # Plot regular closing values with linregression line
    plt.figure('closing_values', figsize=(14,8))
    plt.plot(np.arange(closing_values.shape[0]), closing_values)
    plt.plot((c1 + closings_flat['linregress'] ), color = 'grey')
    plt.plot((c2 + closings_flat['linregress'] ), color = 'orange')
    plt.plot((c3 + closings_flat['linregress'] ), color = 'grey')
    plt.plot((c4 + closings_flat['linregress'] ), color = 'orange')
    plt.plot((c5 + closings_flat['linregress'] ) ,color = 'grey')
    plt.plot((c6 + closings_flat['linregress'] ), color = 'orange')
    plt.plot((c7 + closings_flat['linregress'] ), color = 'grey')
    plt.plot(x2, outcome_values)
#    plt.plot(np.arange(closing_values.shape[0])[closing_peaks], 
#             closing_values[closing_peaks], 'o', color='orange')
    
    '''
    # Plot outcome Width.  Might move this to seperate graph
    if type(outcome_widths) != 'int':
        for ow in outcome_widths:
            plt.plot(np.arange(outcome_values.shape[0]) + np.arange(closing_values.shape[0])[-1], np.ones(outcome_values.shape[0]) * (outcome_values[0] + ow))
            plt.plot(np.arange(outcome_values.shape[0]) + np.arange(closing_values.shape[0])[-1], np.ones(outcome_values.shape[0]) * (outcome_values[0] - ow))
    '''
    # Create Outcome Channels
    shift = c4[-1] - closing_values[-1]
    o1 = np.ones(outcome_values.shape[0]) * (c1[-1] - shift)
    o2 = np.ones(outcome_values.shape[0]) * (c2[-1] - shift)
    o3 = np.ones(outcome_values.shape[0]) * (c3[-1] - shift)
    o4 = np.ones(outcome_values.shape[0]) * (c4[-1] - shift)
    o5 = np.ones(outcome_values.shape[0]) * (c5[-1] - shift)
    o6 = np.ones(outcome_values.shape[0]) * (c6[-1] - shift)
    o7 = np.ones(outcome_values.shape[0]) * (c7[-1] - shift)    
    # Plot outcome channels
    outcome_channels = [o1, o2, o3, o4, o5, o6, o7]
    plot_colors = ['grey', 'orange', 'grey', 'orange', 'grey', 'orange', 'grey']
    for i in range(len(outcome_channels)):
        plt.plot(x2, outcome_channels[i], color=plot_colors[i])
    

    # Plot channel with flattened and scaled closing values
    colors = ['grey', 'grey', 'black', 'grey', 'black', 'grey', 'black', 'grey']
    plt.plot(closings_flat['linregress'])
    plt.figure('channels', figsize=(14,4))
    plt.plot(closings_flat['scaled'])
    for i in range(1, 8):
        plt.plot(channels['c'+str(i)], color=colors[i])
#    plt.plot(np.arange(scaled.shape[0])[scaled_peaks_high], 
#             scaled[scaled_peaks_high], 'o', color='orange')
#    plt.plot(np.arange(scaled.shape[0])[scaled_peaks_low], 
#             scaled[scaled_peaks_low], 'o', color='yellow')
    
    # Print the histogram
    plt.figure('histogram', figsize=(14,4))
    x = (hist[1] + ((hist[1][1] - hist[1][0]) / 2))[:-1]
    plt.plot(x, hist[0])
#    plt.plot(x[hist_peaks], hist[0][hist_peaks], 'o', markersize=8)
    max_hist = hist[0].max()
    
    # Plot the closing position
    plt.plot([channels['closing_position'], channels['closing_position']], 
             [0, max_hist / 2], color = 'black')
    for i in range(1, 8):
        plt.plot([channels['c'+str(i)].mean(), channels['c'+str(i)].mean()], 
                 [0, max_hist],
                 color='orange')
#    plt.plot(scaled[scaled_peaks_high], np.ones(scaled_peaks_high.shape[0]) * max_hist * .5,'o', color='grey')
#    plt.plot(scaled[scaled_peaks_low], np.ones(scaled_peaks_low.shape[0]) * max_hist * .25,'o', color='grey')
    plt.plot(x[keep], hist[0][keep], 'o', color = 'red')

    # Plot outcome widths
#    plt.figure()
#    if type(outcome_widths) != 'int':
#        plt.plot(outcome_values)
#        for ow in outcome_widths:
#            plt.plot(np.ones(outcome_values.shape[0]) * (outcome_values[0] + ow))
#            plt.plot(np.ones(outcome_values.shape[0]) * (outcome_values[0] - ow))
#        print(ow)
#    print(outcome_values.shape[0])
    
    
    
#    # Plot the volume
#    plt.figure('volume_spread', figsize=(14,4))
#    plt.plot(standardscaler().fit_transform(df[['volume', 'spread']])[:, 0])
#    plt.plot(standardscaler().fit_transform(df[['volume', 'spread']])[:, 1])
#    
#    # Plot polar distribution of flattened and scaled
#    plt.figure('polar', figsize=(14, 4))
#    ax = plt.subplot(111, projection='polar')
#    r = np.linspace(0, 1, window_length)
#    theta = 2 * np.pi * r
#    ax.plot(theta, scaled[:-1])
#
#    plt.figure('hist_polar_channel', figsize=(14, 4))
#    ax = plt.subplot(111, projection='polar')
#    r = np.linspace(0, 1, hist[1][1:].shape[0])
#    theta = 2 * np.pi * r
#    ax.plot(theta, hist[0])   


    scaled_heights_avg= abs((scaled[:-1] - scaled[1:])).mean()
    scaled_height_max = abs((scaled[:-1] - scaled[1:])).max()
    scaled_height_ratio = scaled_height_max / scaled_heights_avg
    # Plot and print and Retrun
    plt.show()
    print('closing position:      {:.2f}'.format(channels['closing_position']))
    print('Peaks:                 {}'.format(x[keep]))
    print('Amplitudes:            {}'.format(hist[0][keep]))
    print('Channels Range:        {:.4f}'.format(c7[-1] - c1[-1]))
    print('Average candle height  {}'.format(scaled_heights_avg))
    print('Max candle height      {}'.format(scaled_height_max))    
    print('avg/max candle height  {}'.format(scaled_height_ratio))


        
    
    return 



'''
some quick notes on exploration:
    closing posiiotn at long en of tail of unimodal distribution:
        it came back to other side ( passed uni top), bottom to top channels (90000)
    Saw same behavior as above but in reverse ( 50000)
        even as most peaks were in middle and deviaition was small and mean 
        was centered.
    slightly multimodal, closed at .2, went exactly back up to previosu peak
        even thoough peak was much less amplitude
    Poor peak choice was represented here. - did not line up well with dist.
        inconclusive by my eye right now
    Excellent bimodal but closing values in the middle.
        channel was blown out and move was against slope = 
        inconclusive perhaps when in middl
    exactly the same as above. ended on peak.
        
    hyp - in middle of channel it will be hard to determine direction probs.
    
    maybe sligtly bimodal but only found 1 in alg.  closing was at peak.
        not much movement from 
    Trimodal but alg only found 2.
        
    
        
    
        
        
'''

if __name__ == '__main__':
    pass