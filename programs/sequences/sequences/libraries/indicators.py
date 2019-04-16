import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import linregress



def stochastic_oscillator(candles, periods_k, periods_d):
    close = candles.midclose.values
    high = pd.rolling_max(candles.midhigh, periods_k, how='max')
    low = pd.rolling_min(candles.midlow, periods_k, how='min') 
    k = ((close - low) / (high - low)) * 100
    d = pd.rolling_mean(k, periods_d)
    return d


def bollinger_bands(candles, length=20, std=2):
    '''
    Returns DataFrame with columns: [ timestamp, midclose, sma, upper, lower ]
    '''
    print(candles)
    df = candles[['timestamp', 'midclose']].copy()
    df['sma'] = candles.midclose.rolling(window=length).mean()
    df['upper'] = df.sma + (candles.midclose.rolling(window=length).std() * std)
    df['lower'] = df.sma - (candles.midclose.rolling(window=length).std() * std)
    return df


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

    
    
    




if __name__ == '__main__':
    pass

