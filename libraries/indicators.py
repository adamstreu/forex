import pandas as pd
import numpy as np
import multiprocessing
from scipy.stats import rv_discrete
from sklearn.preprocessing import StandardScaler
import os; os.chdir('/northbend')
from libraries.transformations import get_groups
from classes.channel import Channel
from classes.wave import Wave

def get_rolling_boundary(closing_values, bound, window, direction):
    coll = [np.nan] * window
    for i in range(window, closing_values.shape[0]):
        values = closing_values[i - window: i]
        # Create distribution
        perc = np.ones(values.shape[0]) / values.shape[0]
        dist = rv_discrete(values=(values, perc))
        
        # Get bounding of distribution
        if direction == 'upper' or direction == 'long':
            coll.append(dist.ppf(1 - bound))
        else:
            coll.append(dist.ppf(bound))
    return np.array(coll)


def get_indicator_bins_frequency(indicator_df, outcomes, direction, bins=60, rank=False):
    coll = []
    indexes = []
    if not rank:
        for window in indicator_df.columns.tolist():
            indicator = indicator_df.loc[window:, window].values
            arange = np.arange(indicator_df.shape[0])
            hist = np.histogram(indicator, bins=bins)
            for i in range(bins):
                start = hist[1][i]
                end = hist[1][i+1]
                if direction == 'long':            
                    index = indicator >= start
                else:
                    index = indicator <= end
                index = np.insert(index, 0, [False] * window)
                group_count = get_groups(arange[index], 100).shape[0]
                shape = index.sum()
                outcomes_mean = outcomes.loc[index].mean().tolist()
                for each in list(zip(outcomes.columns.tolist(), outcomes_mean)):
                    coll.append([window, each[0], hist[1][i], i, shape, group_count, each[1]])
                    indexes.append(index)
        # Binned Freq Results Analysis DF
        columns = ['window', 'target', 'slice', 'bin', 'shape', 'groups', 'perc']
        coll = np.array(coll)
        results = pd.DataFrame(coll.T, columns).T
        # Set Location and final currency index]
        return {
                'results': results,
                'indexes': indexes
                }
    else:
        for window in indicator_df.columns.tolist():
            val_counts = list(zip(indicator_df.loc[:, window].value_counts().index,
                                  indicator_df.loc[:, window].value_counts().values))
            for vc in val_counts:
                index = indicator_df.loc[indicator_df.loc[:, window] \
                        == vc[0], window].index.values
                group_count = get_groups(index, 100).shape[0]
                shape = vc[1]
                long_mean = outcomes.loc[index].mean().tolist()
                for each in list(zip(outcomes.columns.tolist(), long_mean)):
                    coll.append([window, each[0], vc[0], shape, group_count, each[1]])
                    indexes.append(index)
                # Binned Freq Results Analysis DF
        columns = ['window', 'target', 'rank', 'shape', 'groups', 'perc']
        coll = np.array(coll)
        results = pd.DataFrame(coll.T, columns).T
        # Set Location and final currency index]
        return {
                'results': results,
                'indexes': indexes
                }






def stochastic_oscillator(close, high, low, periods_k, periods_d):
    high = pd.Series(high).rolling(periods_k).max()
    low = pd.Series(low,).rolling(periods_k).min() 
    k = ((close - low) / (high - low)) * 100
    d = k.rolling(periods_d).mean()
    return d




def rsi(candles):
    pass


def cmo(candles):
    # chandle momentum oscilator
    pass


def asi(candles):
    # Accumulative swing index
    pass


def binary_slope(candles, outcomes_sequence, window_length, setback):
    '''
    Determines 'slope' from avergage outcome (-1 or 1 for lose/ win)
        of binary position placed at each position.
    could add a window type to, say, exponentially weight the window samples
    '''
#    slope = outcomes_sequence[direction] + 0
    slope = (outcomes_sequence[:-setback] + 0).astype(float)
    slope = np.insert(slope, 0, [np.nan] * setback )
    slope[slope == 0] = -1
    slope = pd.rolling_mean(slope, window_length)
    return slope.astype(np.float)


def min_max_mean(candles, window_length):
    closings = candles.midclose.values[1:] - candles.midclose.values[:-1]
    closings = np.insert(closings, 0, np.nan)
    mmm = pd.rolling_mean((pd.rolling_max(closings, window_length) -  \
                           pd.rolling_min(closings, window_length)), 
                           window_length)
    return mmm


def min_max_slope(candles, lengths, current_window):
    lengths = sorted(lengths)
    df = pd.DataFrame(candles.timestamp)
    for length in lengths:
        df['max' + str(length)] = pd.rolling_max(candles.midclose, length)
        df['min' + str(length)] = pd.rolling_min(candles.midclose, length)
    # Slope Up  
    # Max
    values = np.ones(candles.shape[0]).astype(bool)
    dfmax = df.filter(regex='max').copy()
    for i in range(len(lengths) - 1):
        comp = (dfmax.iloc[:, i ] == dfmax.iloc[:, i + 1])
        values = comp & values
    df['up_max'] = values
    # Min
    values = np.ones(candles.shape[0]).astype(bool)
    dfmin = df.filter(regex='min').copy()
    for i in range(len(lengths) - 1):
        values = values & (dfmin.iloc[:, i] > dfmin.iloc[:, i + 1])
    df['up_min'] = values   
    # Current
    df['up_current'] = candles.midclose == \
                          pd.rolling_max(candles.midclose, current_window)
    # Slope Down
    # Max                       
    values = np.ones(candles.shape[0]).astype(bool)
    dfmax = df.filter(regex='max').copy()
    for i in range(len(lengths) - 1):
        comp = (dfmax.iloc[:, i ] < dfmax.iloc[:, i + 1])
        values = comp & values
    df['down_max'] = values
    # Min
    values = np.ones(candles.shape[0]).astype(bool)
    dfmin = df.filter(regex='min').copy()
    for i in range(len(lengths) - 1):
        values = values & (dfmin.iloc[:, i] == dfmin.iloc[:, i + 1])
    df['down_min'] = values   
    # Current
    df['down_current'] = candles.midclose == \
                          pd.rolling_min(candles.midclose, current_window)
    df['slope_up'] = df.up_max & df.up_min
    df['slope_down'] = df.down_max & df.down_min
    return df

    
    
def local_range(candles, window_length):
    df = pd.DataFrame(candles[['timestamp', 'midclose']])
    df['max'] = pd.rolling_max(candles.midhigh, window_length)
    df['min'] = pd.rolling_min(candles.midlow,  window_length)
    df['range'] = df['max'] - df['min']
    return df




def up_down_mean_percentage(closing_values, window):
    
    '''
    Purpose:    Computes change between closing prices (per 5 second typically)
                Designed to be used for short second intervals.
                Iterates over window period interval.
                
    Input:      Closing Values array.
    
    Returns:    Dictionary of 4 arrays:
                    Percent of values of positive change during window.
                    Percent of values of negative change during window.
                    Mean change in values of positive changes.
                    Mean change in values of negative changes.
    '''        
    print('Calculating up_down_mean_perc.  Window period: {}'.format(window))
    def up_down_stuff(window_values):
        '''
        Calculates and returns values for window period
        '''
        window_diff = window_values[1:] - window_values[:-1]
        up_perc     = (window_diff > 0).sum() / window_diff.shape[0]
        down_perc   = (window_diff < 0).sum() / window_diff.shape[0]
        up_mean     = window_diff[window_diff > 0].mean()
        down_mean   = window_diff[window_diff < 0].mean()
        return up_perc, down_perc, up_mean, down_mean

    up_perc_coll = []
    down_perc_coll = []
    up_mean_coll = []
    down_mean_coll = []
    # For all in closing prices, call up_down to calculate window values
    for i in range(window, closing_values.shape[0]):
        window_values = closing_values[i - window: i]
        up_down = up_down_stuff(window_values)
        up_perc_coll.append(up_down[0])
        down_perc_coll.append(up_down[1])
        up_mean_coll.append(up_down[2])
        down_mean_coll.append(up_down[3])
    # Convert to np arrays
    up_mean = np.array(up_mean_coll)
    down_mean = np.array(down_mean_coll)
    up_perc = np.array(up_perc_coll)
    down_perc = np.array(down_perc_coll)
    # Standardize values
    up_mean = (up_mean - up_mean.mean()) / up_mean.std()
    down_mean = (down_mean - down_mean.mean()) / down_mean.std()
    up_perc = (up_perc - up_perc.mean()) / up_perc.std()
    down_perc = (down_perc - down_perc.mean()) / down_perc.std()
    # Return Dictionary
    d = {'up_perc': up_perc,
         'down_perc': down_perc,
         'up_mean': up_mean,
         'down_mean': down_mean}
    return d

    
    
# =============================================================================
# Volume 
# =============================================================================


def get_volume_adjusted_position_difference(values, difference, volume, window):
    coll = [np.nan] * window
    for i in range(window, values.shape[0]):
        v1 = StandardScaler().fit_transform(values[i - window: i].reshape(-1, 1)).ravel()
        v2 = (difference[i - window: i] / volume[i - window: i]).cumsum()
        v2 = StandardScaler().fit_transform(v2.reshape(-1, 1)).ravel()
        coll.append(v1[-1] - v2[-1])
    return np.array(coll)
        
        
def get_volume_adjusted_position_difference_waves(values, difference, volume, windows):
    zeros = np.zeros((values.shape[0], windows.shape[0]))
    a = pd.DataFrame(zeros, columns=windows)
    for window in windows:
        coll = [np.nan] * window
        for i in range(window, values.shape[0]):
            v1 = StandardScaler().fit_transform(values[i - window: i].reshape(-1, 1)).ravel()
            v2 = (difference[i - window: i] / volume[i - window: i]).cumsum()
            v2 = StandardScaler().fit_transform(v2.reshape(-1, 1)).ravel()
            coll.append(v1[-1] - v2[-1])
        a[window] = coll
    return a












# Get rolling rank on a Set of values over a window
def get_rolling_rank(values, window, return_dict):
    print('Getting Rolling rank on {}'.format(window))
    rolling_rank = values.rolling(window).apply(lambda x: \
                       pd.Series(x).rank().values[-1], 
                       raw = False)
    return_dict[window] = rolling_rank
    




# =========== The one and only Wave Wrapper!!! ================================
    
def waves_wrapper(values, windows, function):
    '''
    To be used to call wave functions through to add multiprocessing.
    Maybe not FASTEST but still good and simple to write and use and update.
    Lets say that these functions will all expect a Series.
    Lets say that these functions will all return an array.
    '''
   
    # Call cpus number, processes, join, wait to complete, etc.
    cpus = multiprocessing.cpu_count()
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []
 
    # Run Jobs for first Rotations
    for w in range(windows.shape[0]):
        p = multiprocessing.Process(target=function, 
                                    args=(values,
                                          windows[w], 
                                          return_dict))
        jobs.append(p)
        p.start()
        # Pause to join at number of cpus before moving on.
        if (w % (cpus-1) == 0 and w > 0) or (w == windows.shape[0] - 1):
            for job in jobs:
                job.join()
             
    # Add Results in order and return
    df = pd.DataFrame()
    for w in windows:
         df[w]   = return_dict[w]
    df.columns = windows        
    return df





# Rolling Mean Position and distance from mean - slightly diff than wrapped
def get_rolling_mean_pos_std(values, windows):

    def get_mean_pos_std(values, window, mean_dict, std_dict, pos_dict):
#        print('Getting Rolling mean, std and pos on {}'.format(window))
        rolling_mean = values.rolling(window).mean()
        rolling_std = values.rolling(window).std()
        pos_dict[window] = ((values - rolling_mean) / rolling_std).values
        mean_dict[window] = rolling_mean
        std_dict[window] = rolling_std
    
    # Call cpus number, processes, join, wait to complete, etc.
    cpus = multiprocessing.cpu_count()
    manager = multiprocessing.Manager()
    mean_dict = manager.dict()
    std_dict = manager.dict()    
    pos_dict = manager.dict()
    jobs = []
    # Run Jobs for first Rotations
    for w in range(windows.shape[0]):
        p = multiprocessing.Process(target=get_mean_pos_std, 
                                    args=(values,
                                          windows[w], 
                                          mean_dict,
                                          std_dict,
                                          pos_dict))
        jobs.append(p)
        p.start()
        # Pause to join at number of cpus before moving on.
        if (w % (cpus-1) == 0 and w > 0) or (w == windows.shape[0] - 1):
            for job in jobs:
                job.join()
            
    # Add Results in order and return
    mean = pd.DataFrame()
    std = pd.DataFrame()
    pos = pd.DataFrame()
    for w in windows:
         mean[w]   = mean_dict[w]
         std[w]   = std_dict[w]
         pos[w]   = pos_dict[w]
    mean.columns = windows        
    std.columns = windows        
    pos.columns = windows      
    
    # Return
    return {
            'mean': mean,
            'std': std,
            'pos': pos
            }











# Channel Mean Position and distance from mean - slightly diff than wrapped
def get_channel_mean_pos_std(values, windows):

    def get_mean_pos_std(values, window, mean_dict, 
                         std_dict, pos_dict, slope_dict):
        #print('Getting Channel mean, std and pos on {}'.format(window))
        mean  = [np.nan] * window
        pos   = [np.nan] * window
        std   = [np.nan] * window
        slope = [np.nan] * window
        for i in range(window, values.shape[0]):
            channel = Channel(values[i - window: i])
            mean.append(channel.flattened.mean())
            std.append(channel.flattened.std())       
            pos.append(channel.position_distance_standard)
            slope.append(channel.slope)
        pos_dict[window] = np.array(pos)
        mean_dict[window] = np.array(mean)
        std_dict[window] = np.array(std)
        slope_dict[window] = np.array(slope)
        
        
    # Call cpus number, processes, join, wait to complete, etc.
    cpus = multiprocessing.cpu_count()
    manager = multiprocessing.Manager()
    mean_dict = manager.dict()
    std_dict = manager.dict()    
    pos_dict = manager.dict()
    slope_dict = manager.dict()
    jobs = [] 
    # Run Jobs for first Rotations
    for w in range(windows.shape[0]):
        p = multiprocessing.Process(target=get_mean_pos_std, 
                                    args=(values,
                                          windows[w], 
                                          mean_dict,
                                          std_dict,
                                          pos_dict,
                                          slope_dict))
        jobs.append(p)
        p.start()
        # Pause to join at number of cpus before moving on.
        if (w % (cpus-1) == 0 and w > 0) or (w == windows.shape[0] - 1):
            for job in jobs:
                job.join()
             
    # Add Results in order and return
    mean  = pd.DataFrame()
    std   = pd.DataFrame()
    pos   = pd.DataFrame()
    slope = pd.DataFrame()
    for w in windows:
         mean[w]  = mean_dict[w]
         std[w]   = std_dict[w]
         pos[w]   = pos_dict[w]
         slope[w] = slope_dict[w]
    mean.columns  = windows        
    std.columns   = windows        
    pos.columns   = windows      
    slope.columns = windows        
    # Return
    return {
            'mean' : mean,
            'std'  : std,
            'pos'  : pos,
            'slope': slope
            }




def get_rolling_currency_correlation(values1, values2, windows):
   
    def get_corrs(values1, values2, window, return_dict):
        #print('Getting Currency Correlation on {}'.format(window))
        roll_corr = pd.DataFrame({'a': values1, 'b': values2}).rolling(window).corr()
        roll_corr.index = roll_corr.index.swaplevel(0,1)
        roll_corr = roll_corr.loc['a', 'b']
        return_dict[window] = roll_corr.values
  
    # Call cpus number, processes, join, wait to complete, etc.
    cpus = multiprocessing.cpu_count()
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = [] 
    # Run Jobs for first Rotations
    for w in range(windows.shape[0]):
        p = multiprocessing.Process(target=get_corrs, 
                                    args=(values1,
                                          values2,
                                          windows[w], 
                                          return_dict))
        jobs.append(p)
        p.start()
        # Pause to join at number of cpus before moving on.
        if (w % (cpus-1) == 0 and w > 0) or (w == windows.shape[0] - 1):
            for job in jobs:
                job.join() 
    # Add Results in order and return
    corrs = pd.DataFrame()
    for w in windows:
         corrs[w] = return_dict[w] 
    corrs.columns = windows        
    # Return
    return corrs





# Channel Mean Position and distance from mean - slightly diff than wrapped
def get_rolling_waves(values, windows):

    def function(values, window, fit_dict, freq_dict):
        #print('Getting Channel mean, std and pos on {}'.format(window))
        fit  = [np.nan] * window
        freq   = [np.nan] * window
        for i in range(window, values.shape[0]):
            wave = Wave(values[i - window: i])
            fit.append(wave.fit)
            freq.append(wave.frequency)       
        fit_dict[window] = np.array(fit)
        freq_dict[window] = np.array(freq)
        
        
    # Call cpus number, processes, join, wait to complete, etc.
    cpus = multiprocessing.cpu_count()
    manager = multiprocessing.Manager()
    fit_dict = manager.dict()
    freq_dict = manager.dict()
    jobs = [] 
    # Run Jobs for first Rotations
    for w in range(windows.shape[0]):
        p = multiprocessing.Process(target=function, 
                                    args=(values,
                                          windows[w], 
                                          fit_dict,
                                          freq_dict))
        jobs.append(p)
        p.start()
        # Pause to join at number of cpus before moving on.
        if (w % (cpus-1) == 0 and w > 0) or (w == windows.shape[0] - 1):
            for job in jobs:
                job.join()
             
    # Add Results in order and return
    fit  = pd.DataFrame()
    freq   = pd.DataFrame()
    for w in windows:
         fit[w]  = fit_dict[w]
         freq[w]   = freq_dict[w]
    fit.columns  = windows        
    freq.columns  = windows          
    # Return
    return {
            'fit' : fit,
            'freq' : freq
            }







    




if __name__ == '__main__':
    pass

