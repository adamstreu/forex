###############################################################################
# Notes
###############################################################################
if True:

    '''
    Sum of all instruments to currency positions
    
    do a sum of all slopes
        
    
    
    '''


###############################################################################
# Imports 
###############################################################################
if 0:

    # Imports
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from time import sleep
    from scipy.optimize import leastsq
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    import os; os.chdir('/northbend')
    from classes.channel import Channel
    from classes.wave import Wave
    from libraries.currency_universe import get_universe_singular
    from libraries.currency_universe import backfill_with_singular
    from libraries.indicators import get_rolling_currency_correlation
    from libraries.indicators import get_rolling_mean_pos_std
    from libraries.indicators import get_channel_mean_pos_std
    from libraries.indicators import get_rolling_waves
    from libraries.oanda import get_time
    from libraries.oanda import market
    from libraries.correlation import get_autocorrelation
    from libraries.stats import get_distribution_boundary
    from libraries.taps import get_taps
    
    # Set Environment
    pd.set_option('display.width', 1000)
    pd.set_option('max.columns', 15)
    np.warnings.filterwarnings('ignore')   
    plt.rcParams['figure.figsize'] = [14, 6]
    
    
   

###############################################################################
# Backfill Currencies data to begin 
###############################################################################    
if 1:   

    # General Parameters
    _from = '2018-12-11T00:00:00Z'
    _to   = '2019-12-01T00:00:00Z'
    granularity = 'M5'
    # Currencies to use
    currencies = ['aud', 'cad', 'chf', 'eur', 'gbp', 'hkd', 'jpy', 'nzd', 'usd']     
    
    # Get instrument List
    instrument_list = []
    for mark in market:
        if mark.split('_')[0] in [x.upper() for x in currencies]: 
            if mark.split('_')[1] in [x.upper() for x in currencies]:
                instrument_list.append(mark)    
    
    # Start with Data and Ratios Backfilled
    cu = pd.DataFrame(columns = currencies)
    cu, ratios = backfill_with_singular(currencies, granularity, _from, _to)
    cu.index.names = ['timestamp']
    ratios.index.names = ['timestamp']
    cu.reset_index(inplace=True)
    ratios.reset_index(inplace=True)
    ratios.iloc[:, 1] = ratios.iloc[:, 1:].astype(float)
    cu.iloc[:, 1] = cu.iloc[:, 1:].astype(float)

    # Print average sum of Currencies ( must equal 1)
    print(cu.sum(axis=1).mean())


    
    


###############################################################################
# Get Channels and Slopes for currencies from multiple (many) windows
# See if there are many channel positions where a currency is at top of channel.
###############################################################################  
if 1:
    
    print('\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n')
    
    # Only use these currencies for placements (high leverage)
    currencies = ['aud', 'cad', 'nzd', 'chf', 'eur', 'usd']
    
    for cur in currencies:
        windows = np.arange(5, 480, 10)
        end = cu.last_valid_index()
        begin = end - windows.max() * 1
        begin_plot = end -  windows.max()
        
        # Get Slope and position for all values (instruments)
        positions = pd.DataFrame()
        deviations = pd.DataFrame()
        slopes = pd.DataFrame()
        pos = get_channel_mean_pos_std(cu.loc[begin: end, cur].values, windows)
        positions = pos['pos']
        positions.index = np.arange(begin, end + 1)
        slopes = pos['slope']
        slopes.index = np.arange(begin, end + 1)
        
        # Standardize
        # positions = StandardScaler().fit_transform(positions.fillna(0))
        # positions = pd.DataFrame(positions, columns = windows, index=np.arange(begin, end + 1)) 
              
        # Plot Currency
        plt.figure(figsize=(14,5))
        cu.loc[begin: end, cur].plot()
        plt.title(str(cur))
        plt.tight_layout()        
        
        # Plot Slope
        plt.figure(figsize=(14,5))
        slopes.mean(axis=1).plot()
        plt.plot(positions.index.values, np.zeros(positions.index.values.shape[0]), color='grey')
        plt.title('Slops on ' + str(cur))
        plt.tight_layout()
        
        # Plot Positions        
        plt.figure(figsize=(14,5))
        positions.mean(axis=1).plot()
        plt.plot(positions.index.values, np.zeros(positions.index.values.shape[0]), color='grey')
        plt.plot(positions.index.values, np.ones(positions.index.values.shape[0]) * 2, color='grey')
        plt.plot(positions.index.values, np.ones(positions.index.values.shape[0]) * -2, color='grey')
        plt.title('Positions on ' + str(cur))
        plt.tight_layout()

    # Print Timestamp
    print()
    print(cu.loc[cu.last_valid_index(), 'timestamp'])
    print(granularity)


    if False:
        
        ratios.loc[begin: end, 'NZD_USD'].plot(); plt.tight_layout()
        plt.plot((nzd_positions.dropna().values - usd_positions.dropna().values)); plt.tight_layout()




























































    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
###############################################################################
# Attempt a channel fit score.  If successful - move to channel and indic....
###############################################################################    
if 0:
    
    '''
    some possibel score measures:
        frequency of waves - want a few
        fit of wave ?
    '''
    
    for i in range(100, 1000, 10):
        start = 0
        end = i
    
        instrument = 'EUR_USD'
        
        inst = ratios.loc[start: end, instrument].astype(float)
        channel = Channel(inst.values)
        wave = Wave(inst.values)

        plt.figure()
        plt.plot(channel.flattened)
        plt.plot(channel.c5() - channel.line)
        plt.plot(channel.c1() - channel.line)    
        plt.plot(wave.wave - wave.linregress)
        plt.show()
        
        plt.figure()
        plt.plot(channel.flattened)
        plt.plot(channel.c5() - channel.line)
        plt.plot(channel.c1() - channel.line)    
        plt.plot(wave.wave - wave.linregress, 'o')
        plt.show()
        
        print(end)
        print('FIT: {}'.format(waves_fit.loc[end].values))
        print('FREQ: {}'.format(waves_freq.loc[end].values)) 

        raw_input = input('touch')
        if raw_input == 'd':
            break
    

###############################################################################
# Basic channel Finder.
###############################################################################    
if 0:
    
    # get all the desired dfs per window
    windows = np.array([60])
    positions = pd.DataFrame()
    deviations = pd.DataFrame()
    slopes = pd.DataFrame()
    instruments = ['EUR_USD', 'USD_CAD', 'NZD_USD', 
           'GBP_USD', 'USD_JPY', 'GBP_USD', 'AUD_USD']
    instruments = ['EUR_AUD', 'EUR_CHF', 'EUR_GBP', 
                   'EUR_JPY', 'EUR_USD']
    for instrument in instruments:
        pos = get_channel_mean_pos_std(ratios[instrument].values.astype(float), windows)
        positions[instrument] = pos['pos'].values.ravel()
        deviations[instrument] = pos['std'].values.ravel()
        slopes[instrument] = pos['slope'].values.ravel()
        
    # Measure cluster difference
    cluster_distance = 0
    for i in range(positions.shape[1] - 1):
        for j in range(i + 1, positions.shape[1]):
            cluster_distance += abs(positions.iloc[:, i] - positions.iloc[:, j])
            print(i, j)
   
    '''
    # Get Waves
    waves_fit = waves['fit']
    waves_freq = waves['freq']
    waves = get_rolling_waves(ratios[instrument].values.astype(float), windows)
    '''
    
    plt.figure()
    cu.eur.plot()
    plt.title('eur')
    
    plt.figure()
    ((slopes - slopes.mean(axis=0)) / slopes.std(axis=0)).plot()    
    plt.plot(slopes.dropna().index.values, np.zeros(slopes.dropna().index.values.shape[0]), color='grey')
    plt.title('slopes on eur')
    
    plt.figure()
    ratios.EUR_USD.plot()
    plt.title('eur_usd')
    
    
    
    
    
    
    
    
    '''
    # Attempt a score metric
    pos_from_2 = 2 - positions
    score = pos_from_2.mean(axis=1) - cluster_distance     
    
    # Plot all of everything
    plt.figure('score')
    score.plot()
    plt.title('score')
    plt.figure('deviations')
    ((deviations - deviations.mean(axis=0)) / deviations.std(axis=0)).plot()
    ((deviations - deviations.mean(axis=0)) / deviations.std(axis=0)).mean(axis=1).plot(color='black')
    plt.title('deviations')
    plt.figure('positions')
    positions.plot()
    plt.plot(slopes.dropna().index.values, np.zeros(slopes.dropna().index.values.shape[0]), color='grey')
    plt.plot(slopes.dropna().index.values, np.ones(slopes.dropna().index.values.shape[0]) * 2, color='grey')
    plt.plot(slopes.dropna().index.values, np.ones(slopes.dropna().index.values.shape[0]) * -2, color='grey')
    plt.title('positions')
    plt.figure('slopes')
    ((slopes - slopes.mean(axis=0)) / slopes.std(axis=0)).plot()    
    plt.plot(slopes.dropna().index.values, np.zeros(slopes.dropna().index.values.shape[0]), color='grey')
    plt.title('slopes')
    plt.figure('eur')
    cu.eur.plot()
    plt.title('eur')
    plt.figure()
    cu.usd.plot()
    plt.title('usd')
    plt.figure()
    ratios.EUR_USD.plot()
    plt.title('eur_usd')
    '''

###############################################################################
# Improve wave
###############################################################################    
if 0:
    
    '''
    Guess must be better
        can try taps ? 
        time crossing boundary ? 
        
        how well does upper hit and lower hit ? 
        
    Should do for different timelines.
        on what timelines mdoes everything work right?
        
        
    
    '''
    
    
    def mse(self, x, y):
        mse = mean_squared_error(x, y)
        # print(mse)
        mse /= (x.std() * y.std())
        return mse  
    


    def extension(self, prediction_values):
        prediction_range = prediction_values.shape[0]
        channel_range = self.wave.shape[0]
        pred_channel_ratio = (prediction_range + channel_range) / channel_range
        t = np.linspace(0, pred_channel_ratio * 2 *np.pi, channel_range + prediction_range)
        x = np.arange(self.channel_wave.shape[0] + prediction_values.shape[0])
        line = self.channel.slope * x + self.channel.intercept
        prediction_wave =  self.amplitude * np.sin(self.frequency * t + self.phase_shift) + self.vertical_shift
        self.channel_wave_extension = prediction_wave[channel_range:]
        self.wave_extension = (prediction_wave + line)[channel_range:]
        self.extension_r2 = 0
        self.extension_fit = self.mse(self.wave_extension, prediction_values)
        return None        
    
    
    def plot(self, prediction_values):
        self.extension(prediction_values)

        # Instantiate Plot and Create x values
        plt.figure(figsize=(8, 3))
        x1 = np.arange(self.basis.shape[0])
        x2 = np.arange(prediction_values.shape[0]) + x1[-1] + 1
        # Plt fit values and wave
        plt.plot(self.basis)    
        plt.plot(x1, self.wave)
        # PLot prediction wave and values
        plt.plot(x2, prediction_values, color='black')
        plt.plot(x2, self.wave_extension)        
        # Plot lin line and channel
        plt.plot(x1, self.linregress, color='yellow')
        plt.plot(x1, (self.linregress + self.channel.channel_deviation * 2), color='yellow')
        plt.plot(x1, (self.linregress - self.channel.channel_deviation * 2), color='yellow')
        plt.plot(x1, (self.linregress + self.channel.channel_deviation * 1), color='yellow')
        plt.plot(x1, (self.linregress - self.channel.channel_deviation * 1), color='yellow')
        # Show plot
        plt.tight_layout()
        plt.show()
        
        # Pritn plot info
        print('Wave Fit:      {}'.format(self.fit))
        print('Pred Fit:      {}'.format(self.extension_fit))
        print('determ:        {}'.format(self.extension_r2))
        print('Frequency:     {}'.format(self.frequency))
        print('Tangent:       {}'.format(self.tangent))
        print('Phase shift:   {}'.format(self.phase_shift))
        print('Phase pos:     {}'.format(self.phase_position))    


    
    
    
    
    def cosine(self, values):
        # prediction_cosine = wave.amplitude * np.cos(wave.frequency * t + wave.phase_shift) + wave.vertical_shift
        pass
    
    
 
        
    def get_error(as_type, wave, values, cosine, std_dev):
        # All erros start the same
        errors = (values - wave) * (values - wave)
        errors /= std_dev
        if as_type != 'msse':
            # If values above wave when wave goin up, et error to 0
            errors[(cosine > 0) & (values > wave)] = 0
            # Same but opposte going down
            errors[(cosine < 0) & (values < wave)] = 0
        return errors
    
    
    
    

    start = 0
    end = 350
    instrument = 'AUD_CAD'
    values = ratios.loc[start: end, instrument].astype(float).rolling(15).mean().fillna(method='backfill').values
    values = ratios.loc[start: end, instrument].astype(float).values

    channel = Channel(values)


    # bounded thresholds on channel flattened
    bound = .03
    upper = get_distribution_boundary(channel.flattened, bound)['upper_bound']
    lower = get_distribution_boundary(channel.flattened, bound)['lower_bound']

    # Get Autocorreltaion to estimate frequency
    autocorr = get_autocorrelation(channel.flattened)['autocor']
    margin               = int(values.shape[0] * .10) 
    corr                 = autocorr[margin:-margin]
    maximum              = corr.argmax()#[:, 1].argmax()
    minimum              = corr.argmin()#[:, 1].argmin()
    corr_period          = min(int(values.shape[0] * .75), 2 * abs(corr[maximum] - corr[minimum])) #abs(corr[maximum, 0] - corr[minimum, 0]))

    # Get corr peaks ( maybe good indicator of 'smoothness'
    smoothness = int(values.shape[0]  * .25)
    corr_smoothed        = corr#[:, 1]
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
    c0 = channel.channel_deviation * -2 # channel.flattened[0] - channel.channel_deviation * 2     # WRONG
    amplitude            = channel.channel_deviation * 2
    amplitude            = np.average((abs(lower), upper))
    frequency_guess      = auto_peaks / 2     #     values.shape[0] / corr_period     # Really bad guess.
    phase_shift_guess    = - np.argmax(channel.flattened <  c0)
    vertical_shift_guess = amplitude + c0
    
    # Get Real Wave
    t = np.linspace(0, 2*np.pi, channel.flattened.shape[0])
    optimize_func = lambda x: amplitude * np.sin(x[0] * t + x[1]) + x[2]  - channel.flattened
    est_frequency, est_phase_shift, est_vertical_shift = \
            leastsq(optimize_func, [frequency_guess, phase_shift_guess, vertical_shift_guess], full_output=True)[0]
    
    # Get Flattened wave
    wave = amplitude * np.sin(est_frequency * t + est_phase_shift) + est_vertical_shift
    
    # Assess Fit
    fit = ((channel.flattened - wave) ** 2 ).mean()
    # Assess fit from distribution
    range_min = np.minimum(wave, channel.flattened).min()    
    range_max = np.maximum(wave, channel.flattened).max()    
    wave_hist = np.histogram(wave, bins = 200, range=(range_min, range_max)) 
    channel_hist = np.histogram(channel.flattened, bins = 200, range=(range_min, range_max))     
    hist_fit = (channel_hist[0] - wave_hist[0])
    hist_fit = hist_fit[hist_fit > 0].mean() +  abs(hist_fit[hist_fit < 0].mean()) #hist_fit[hist_fit > 0].sum()

    print('Frequency:     {:.4f}'.format(est_frequency))
    print('Least Sqs Fit: {:.4f}'.format(fit * 100000))
    print('Histogram fit: {:.4f}'.format(hist_fit))
    #print('Product fit: {:.4f}'.format(hist_fit * fit * 100000))
    
    
    plt.plot(channel.flattened)
    plt.plot(np.ones(channel.flattened.shape[0]) * channel.channel_deviation * 2)
    plt.plot(np.ones(channel.flattened.shape[0]) * channel.channel_deviation * -2 )  
    plt.plot(np.ones(channel.flattened.shape[0]) * upper, color='blue')
    plt.plot(np.ones(channel.flattened.shape[0]) * lower, color = 'blue')
    plt.plot(wave)
 
    
    plt.figure()
    sns.distplot(channel.flattened, label = 'channel')
    sns.distplot(wave, label = 'wave')
    plt.legend()





    '''    
    plt.figure()
    values = ratios.loc[start: end, instrument].astype(float).rolling(120).mean().fillna(method='backfill').values
    taps = get_taps(values, 120, 30)
    plt.plot(values)
    plt.plot(np.arange(values.shape[0])[taps['upper']], values[taps['upper']], 'o')
    plt.plot(np.arange(values.shape[0])[taps['lower']], values[taps['lower']], 'o')    
    '''
    
    
    
    
    
    
    '''
    # Provide for the tangent
    cosine = amplitude * np.cos(est_frequency * t + est_phase_shift) + est_vertical_shift
    
 
    self.channel = Channel(values)
    self.amplitude = amplitude
    self.frequency = est_frequency
    self.phase_shift = est_phase_shift
    self.vertical_shift = est_vertical_shift
    self.channel_wave = wave
    self.phase_position = 0 # where in phase (%?) was last position
    self.cosine = cosine
    self.tangent = cosine[-1] / self.channel.channel_deviation / 2
    self.basis = values

    x = np.arange(values.shape[0])
    self.linregress = self.channel.slope * x + self.channel.intercept
    self.wave = self.channel_wave + self.linregress
    self.fit = self.mse(self.wave, values)
    
    # Shit.  Wanted to collect the error terms as well


    '''
    
    
    
###############################################################################
# Plotting the multiple channels on instruments
###############################################################################    
if 0:
    
    # get all the desired dfs per window
    window = 24*5
    after = 24
    positions = pd.DataFrame()
    deviations = pd.DataFrame()

    slopes = pd.DataFrame()
    instruments = ['EUR_AUD', 'EUR_CHF', 'EUR_GBP', 
                   'EUR_JPY', 'EUR_USD']
    
    
    
    
    for i in range(window, cu.shape[0] - after, 10):
        start = i - window
        middle = i
        end = i + after
        

        
        plt.figure(figsize=(14, 3))
        

        values = cu.loc[start: middle, 'eur'].astype(float).rolling(15).mean().fillna(method='backfill').values
        values = cu.loc[start: middle, 'eur'].astype(float).values
        
        channel = Channel(values)

        later_values = cu.loc[middle: end - 1 , 'eur'].astype(float).values
#        later_values -= later_values[0]
#        later_values += channel.flattened[-1]
        
        
        x = np.arange(values.shape[0])
        x2 = np.arange(end - middle) + x[-1] 
    
    
        # bounded thresholds on channel flattened
        bound = .03
        upper = get_distribution_boundary(channel.flattened, bound)['upper_bound']
        lower = get_distribution_boundary(channel.flattened, bound)['lower_bound']
    
        # Get Autocorreltaion to estimate frequency
        autocorr = get_autocorrelation(channel.flattened)['autocor']
        margin               = int(values.shape[0] * .10) 
        corr                 = autocorr[margin:-margin]
        maximum              = corr.argmax()#[:, 1].argmax()
        minimum              = corr.argmin()#[:, 1].argmin()
        corr_period          = min(int(values.shape[0] * .75), 2 * abs(corr[maximum] - corr[minimum])) #abs(corr[maximum, 0] - corr[minimum, 0]))
    
        # Get corr peaks ( maybe good indicator of 'smoothness'
        smoothness = int(values.shape[0]  * .25)
        corr_smoothed        = corr#[:, 1]
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
        c0 = channel.channel_deviation * -2 # channel.flattened[0] - channel.channel_deviation * 2     # WRONG
        amplitude            = channel.channel_deviation * 2
        amplitude            = np.average((abs(lower), upper))
        frequency_guess      = auto_peaks / 2     #     values.shape[0] / corr_period     # Really bad guess.
        phase_shift_guess    = - np.argmax(channel.flattened <  c0)
        vertical_shift_guess = amplitude + c0
        
        # Get Real Wave
        t = np.linspace(0, 2*np.pi, channel.flattened.shape[0])
        optimize_func = lambda x: amplitude * np.sin(x[0] * t + x[1]) + x[2]  - channel.flattened
        est_frequency, est_phase_shift, est_vertical_shift = \
                leastsq(optimize_func, [frequency_guess, phase_shift_guess, vertical_shift_guess], full_output=True)[0]
        
        # Get Flattened wave
        wave = amplitude * np.sin(est_frequency * t + est_phase_shift) + est_vertical_shift
        
        # Assess Fit
        fit = ((channel.flattened - wave) ** 2 ).mean()
        # Assess fit from distribution
        range_min = np.minimum(wave, channel.flattened).min()    
        range_max = np.maximum(wave, channel.flattened).max()    
        wave_hist = np.histogram(wave, bins = 200, range=(range_min, range_max)) 
        channel_hist = np.histogram(channel.flattened, bins = 200, range=(range_min, range_max))     
        hist_fit = (channel_hist[0] - wave_hist[0])
        hist_fit = hist_fit[hist_fit > 0].mean() +  abs(hist_fit[hist_fit < 0].mean()) #hist_fit[hist_fit > 0].sum()
        
        '''
        print('Currency:      {}'.format(instrument))
        print('Frequency:     {:.4f}'.format(est_frequency))
        print('Least Sqs Fit: {:.4f}'.format(fit * 100000))
        print('Histogram fit: {:.4f}'.format(hist_fit))
        print('Position:      {:.4f}'.format(channel.position_distance_standard))            
        '''
        
        
        
        plt.plot(x, channel.flattened + channel.line)
        plt.plot(x, np.ones(channel.flattened.shape[0]) * channel.channel_deviation * 2 + channel.line)
        plt.plot(x, np.ones(channel.flattened.shape[0]) * channel.channel_deviation * -2  + channel.line)  
        plt.plot(x, np.ones(channel.flattened.shape[0]) * upper  + channel.line, color='blue')
        plt.plot(x, np.ones(channel.flattened.shape[0]) * lower + channel.line, color = 'blue')
        plt.plot(x, wave + channel.line)
        plt.plot(x2, later_values  )
        plt.title('EUR')
        plt.show()
        
        
        
    
        
        
        
        
        
        for instrument in instruments:
            
            
        
        
            plt.figure(figsize=(14, 2))
            

            values = ratios.loc[start: middle, instrument].astype(float).rolling(15).mean().fillna(method='backfill').values
            values = ratios.loc[start: middle, instrument].astype(float).values
            
            
            
            channel = Channel(values)

            later_values = ratios.loc[middle: end - 1 , instrument].astype(float).values
#            later_values -= later_values[0]
#            later_values += channel.flattened[-1]
            
            
            x = np.arange(values.shape[0])
            x2 = np.arange(end - middle) + x[-1] 
        
        
            # bounded thresholds on channel flattened
            bound = .03
            upper = get_distribution_boundary(channel.flattened, bound)['upper_bound']
            lower = get_distribution_boundary(channel.flattened, bound)['lower_bound']
        
            # Get Autocorreltaion to estimate frequency
            autocorr = get_autocorrelation(channel.flattened)['autocor']
            margin               = int(values.shape[0] * .10) 
            corr                 = autocorr[margin:-margin]
            maximum              = corr.argmax()#[:, 1].argmax()
            minimum              = corr.argmin()#[:, 1].argmin()
            corr_period          = min(int(values.shape[0] * .75), 2 * abs(corr[maximum] - corr[minimum])) #abs(corr[maximum, 0] - corr[minimum, 0]))
        
            # Get corr peaks ( maybe good indicator of 'smoothness'
            smoothness = int(values.shape[0]  * .25)
            corr_smoothed        = corr#[:, 1]
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
            c0 = channel.channel_deviation * -2 # channel.flattened[0] - channel.channel_deviation * 2     # WRONG
            amplitude            = channel.channel_deviation * 2
            amplitude            = np.average((abs(lower), upper))
            frequency_guess      = auto_peaks / 2     #     values.shape[0] / corr_period     # Really bad guess.
            phase_shift_guess    = - np.argmax(channel.flattened <  c0)
            vertical_shift_guess = amplitude + c0
            
            # Get Real Wave
            t = np.linspace(0, 2*np.pi, channel.flattened.shape[0])
            optimize_func = lambda x: amplitude * np.sin(x[0] * t + x[1]) + x[2]  - channel.flattened
            est_frequency, est_phase_shift, est_vertical_shift = \
                    leastsq(optimize_func, [frequency_guess, phase_shift_guess, vertical_shift_guess], full_output=True)[0]
            
            # Get Flattened wave
            wave = amplitude * np.sin(est_frequency * t + est_phase_shift) + est_vertical_shift
            
            # Assess Fit
            fit = ((channel.flattened - wave) ** 2 ).mean()
            # Assess fit from distribution
            range_min = np.minimum(wave, channel.flattened).min()    
            range_max = np.maximum(wave, channel.flattened).max()    
            wave_hist = np.histogram(wave, bins = 200, range=(range_min, range_max)) 
            channel_hist = np.histogram(channel.flattened, bins = 200, range=(range_min, range_max))     
            hist_fit = (channel_hist[0] - wave_hist[0])
            hist_fit = hist_fit[hist_fit > 0].mean() +  abs(hist_fit[hist_fit < 0].mean()) #hist_fit[hist_fit > 0].sum()
            
            '''
            print('Currency:      {}'.format(instrument))
            print('Frequency:     {:.4f}'.format(est_frequency))
            print('Least Sqs Fit: {:.4f}'.format(fit * 100000))
            print('Histogram fit: {:.4f}'.format(hist_fit))
            print('Position:      {:.4f}'.format(channel.position_distance_standard))            
            '''
            
            plt.plot(x, channel.flattened + channel.line)
            plt.plot(x, np.ones(channel.flattened.shape[0]) * channel.channel_deviation * 2 + channel.line)
            plt.plot(x, np.ones(channel.flattened.shape[0]) * channel.channel_deviation * -2 + channel.line )  
            plt.plot(x, np.ones(channel.flattened.shape[0]) * upper + channel.line, color='blue')
            plt.plot(x, np.ones(channel.flattened.shape[0]) * lower + channel.line, color = 'blue')
            plt.plot(x, wave + channel.line)
            plt.plot(x2, later_values )
            plt.title(instrument)
            plt.show()
            
        

        
            
        raw_input = input(i)
        if raw_input == 'b':
            break

        

        
        
        
        
        
    
###############################################################################
# Machine learning eur slope from instrument slopes
###############################################################################    
if 0:
    
    # Indicators and outcomes for machine learning
    if True:
        
        # get all the desired dfs per window
        windows = np.array([120])
        
        instrument_positions = pd.DataFrame()
        instrument_deviations = pd.DataFrame()
        instrument_slopes = pd.DataFrame()    
        currency_positions = pd.DataFrame()
        currency_deviations = pd.DataFrame()
        currency_slopes = pd.DataFrame()   

        instruments = ['EUR_USD', 'USD_CAD', 'NZD_USD', 
                       'GBP_USD', 'USD_JPY', 'GBP_USD', 'AUD_USD']
        instruments = ['EUR_USD', 'EUR_CHF', 'EUR_GBP', 
                       'EUR_JPY', 'EUR_AUD', 'EUR_USD', 'USD_CAD', 'NZD_USD', 
                       'GBP_USD', 'USD_JPY', 'GBP_USD', 'AUD_USD']
        currency = 'usd'
        
        for instrument in instruments:
            pos = get_channel_mean_pos_std(ratios[instrument].values.astype(float), windows)
            instrument_positions[instrument] = pos['pos'].values.ravel()
            instrument_deviations[instrument] = pos['std'].values.ravel()
            instrument_slopes[instrument] = pos['slope'].values.ravel()
        
        pos = get_channel_mean_pos_std(cu[currency].values.astype(float), windows)
        currency_positions[currency] = pos['pos'].values.ravel()
        currency_deviations[currency] = pos['std'].values.ravel()
        currency_slopes[currency] = pos['slope'].values.ravel()
        
    # Run machine learning
    if True:

        ml_index = instrument_slopes.dropna().index.values
        ml = instrument_slopes.loc[ml_index]
        ml_outs = currency_slopes.loc[ml_index, currency]
        
        '''   Look ahead - try to predict slope in 15 moves '''
        ml_outs = cu.loc[ml_index, currency]
        ml_outs = currency_slopes.loc[ml_index, currency].shift(15)
        ml_index = ml_outs.dropna().index.values
        ml = instrument_slopes.loc[ml_index]
        ml_outs = ml_outs.loc[ml_index]
        
        # Split  
        train_index, test_index = train_test_split(ml.index.values, 
                                                   train_size = .8, 
                                                   shuffle = False)
        x_train    = ml.loc[train_index]
        y_train    = ml_outs.loc[train_index]
        x_test     = ml.loc[test_index]
        y_test     = ml_outs.loc[test_index]
        
        
        # Scale 
        scaler = StandardScaler().fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)
    
        
        # Implemnent Model 
        linreg = LinearRegression()
        linreg.fit(x_train, y_train)
        predictions = linreg.predict(x_test)
        
        plt.figure()
        plt.plot(predictions, label = 'predictions')
        plt.plot(y_test.values, label = 'currency')
        plt.legend()
        plt.show()
        
        score = linreg.score(x_test, y_test)
        print('SCORE: {:.2f}'.format(score))
        
        




    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

        
        
        
    
    # use slopes (and correlation) to predict future slope
    if False:

        if False:
            
            eur_predictions = predictions
            eur_test = y_test
            usd_predictions = predictions
            usd_test = y_test
            
            rolling_correlation = get_rolling_currency_correlation(eur_test.values.ravel(), 
                                                               usd_test.values.ravel(), 
                                                               windows)
            rolling_correlation.index = eur_test.index
           
            plt.plot(usd_test)
            plt.plot(eur_test)
            plt.figure()
            plt.plot(rolling_correlation)
        
           
        pos = get_channel_mean_pos_std(ratios['AUD_CAD'].values.astype(float), windows)
        ml_inst = pos['pos'].values.ravel()
        ml_inst = pd.DataFrame(ml_inst)#, columns = 'slope', index = ratios.index)
        
        
        ml_cur = pd.DataFrame({'eur': eur_test.values.ravel(),
                               'usd': usd_test.values.ravel() ,
                               'corr': rolling_correlation.values.ravel()}, 
                               index = test_index)
        
        
        ml_index = ml_cur.dropna().index.values
        ml_cur = ml_cur.loc[ml_index]

        ml_inst = ratios.loc[ml_index, 'AUD_CAD']
        
        # Split  
        train_again, test_again = train_test_split(ml_cur.index.values, 
                                                   train_size = .8, 
                                                   shuffle = False)
        x_train    = ml_cur.loc[train_again]
        y_train    = ml_inst.loc[train_again]
        x_test     = ml_cur.loc[test_again]
        y_test     = ml_inst.loc[test_again]
        
        
                
        # Implemnent Model 
        linreg = LinearRegression()
        linreg.fit(x_train, y_train)
        predictions = linreg.predict(x_test)
        
        pd.Series(predictions.ravel()).rolling(10).mean().plot() #plt.plot(predictions, label = 'predictions')
        plt.plot(y_test.values, label = 'currency')
        plt.legend()
        plt.show()
        
        score = linreg.score(x_test, y_test)
        print('SCORE: {:.2f}'.format(score))
    

        
    
     
###############################################################################
# Simple slope Prediction
###############################################################################    
if 0:
       

    # get all the desired dfs per window
    windows = np.array([60])
    positions = pd.DataFrame()
    deviations = pd.DataFrame()
    slopes = pd.DataFrame()

    currencies = ['aud', 'cad', 'eur', 'gbp', 'nzd', 'usd']
    for currency in currencies:
        pos = get_channel_mean_pos_std(cu[currency].values.astype(float), windows)
        positions[currency] = pos['pos'].values.ravel()
        deviations[currency] = pos['std'].values.ravel()
        slopes[currency] = pos['slope'].values.ravel()
        
        
        
        
    eur_high = (slopes.rolling(5).mean().eur.shift(1) < slopes.rolling(5).mean().eur) \
             & (slopes.rolling(5).mean().eur.shift(-1) < slopes.rolling(5).mean().eur)
    usd_low = (slopes.rolling(5).mean().usd.shift(1) > slopes.rolling(5).mean().usd) \
             & (slopes.rolling(5).mean().usd.shift(-1) > slopes.rolling(5).mean().usd)    
    
    plt.figure()
    ratios.EUR_USD.plot()
    plt.title('eur_usd')
    
    plt.figure()
    slopes.rolling(5).mean().loc[:, ['eur', 'usd']].plot()
    plt.plot(np.zeros(slopes.shape[0]), color='grey')
    
    slopes.eur.loc[eur_high].plot(style='o')
    slopes.usd.loc[usd_low].plot(style='o')        
    
    both = np.intersect1d(eur_high[eur_high].index.values,
                          usd_low[usd_low].index.values)
    
        
    
    plt.figure()
    ratios.EUR_USD.plot()
    ratios.loc[both, 'EUR_USD'].plot(style='o')
    plt.title('eur_usd')
    
    
    
        
        
        