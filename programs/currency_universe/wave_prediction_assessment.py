import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os; os.chdir('/northbend')
from classes.wave import Wave
from libraries.currency_creation import currency_matrix


'''
GOAL:
    
    DETERMINE:  Does a better model wave fit lead to better prediction fit?
        
        
TO DO:
    
    Lost:
        correlation of each variable to predictied fit
            and % over c5
            percent under c3
            percent under c0
        
    need to add R2
    not sure doing interval right on main
    didn't colect the error terms  need to
    check all values
        need position in phase at end
        check tangent,
        check cosine and tangent
        get coef of det
    do graph through results set so that we can filter on results
    check and improve wave fit
    make sure the error function is appropriate
    Can we speed up the wae fit?
    
    
Analysis:
    Is there a window (range) that works better / worse?
    Do currencies perform diff than corr, ratios?
    Do the currencies overall corr make a diff into the fit and prediction?
    Of course, filter on parameters after basic analysis.
    Basic analysis: are higher fits correlated with higher prediction.  one line.
    Do short or long curr rolls 'breath' more evenly ? 

'''
        
# Switches
get_values         = 0
get_results        = 0
analysis           = 0
graph_verification = 1


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


def filter_results(d):
    # Make sure not to fuck with original
    df = d.copy()
    # Create Filters
    cond2 = True # df.wave_fit > 1
    cond3 = True # df.frequency > 2
    cond4 = df.tangent < .5
    cond5 = df.tangent > -0.5
    
    # Apply Filter
    cond1 = np.ones(df.shape[0]).astype(bool)
    conditions = cond1 & cond2 & cond3 & cond3 & cond4 & cond5
    df = df[conditions]
    # Return
    print('Results filtered: {}'.format(d.shape[0] - df.shape[0]))
    return df


###############################################################################
# Get Values - particularly for second computer
###############################################################################
if get_values:
    
    # Get Values
    pair_values = np.load('/home/adam/Desktop/pair_values.npy')
    pair_names = np.load('/home/adam/Desktop/pair_names.npy')
    cur = currency_matrix(pair_names, pair_values)
    values = pair_values[0, :1000]
    
    # Work only with primary currencies
    df = pd.DataFrame(cur.loc['aud', 'aud'])
    df['cad'] = pd.DataFrame(cur.loc['cad', 'cad'])
    df['eur'] = pd.DataFrame(cur.loc['eur', 'eur'])
    df['gbp'] = pd.DataFrame(cur.loc['gbp', 'gbp'])
    df['nzd'] = pd.DataFrame(cur.loc['nzd', 'nzd'])
    df['usd'] = pd.DataFrame(cur.loc['usd', 'usd'])
    cur = df.copy()
    cur_backup = cur.copy()
    
    # Set environment
    pd.set_option('display.width', 1000)
    pd.set_option('max.columns', 15)
    

    # Main values to use throughout
    # values  = corrs[:1500]
    values = cur.loc[:, 'aud'].values

###############################################################################
# Fit channel and Wave.  Measure prediction Error.  Collect Errors, stats, data..
###############################################################################
if get_results:
    
    # Perform on what
    # Parameters
    channel_range =  500    
    prediction_range = 150
    # Instantiation
    results = []
    errors_coll = []
    samples = 1000
    # For each location, analyze wave fit (preds) from multiple channel ranges.
    # for location in range(channel_range, values.shape[0] - prediction_range):
    for location in np.linspace(channel_range, 
                                values.shape[0] - prediction_range, 
                                samples).astype(int): 
        if location % 1000 == 0:
            print(location)
        # Values to use for channel range and location
        fit_values = values[location - channel_range: location]
        prediction_values = values[location - channel_range: location + prediction_range]
        # Create Wave.  Use to create prediction wave over all values 
        wave = Wave(fit_values)
        wave.extension(prediction_values)
        results.append([location,
                        channel_range,
                        wave.fit,
                        wave.extension_fit,
                        wave.channel.slope, 
                        wave.channel.position_distance_standard,
                        wave.channel.channel_deviation,
                        wave.amplitude,
                        wave.phase_shift,
                        wave.frequency,
                        wave.phase_position,
                        wave.tangent,
                        ])
    
    # Creat DataFrame with Results for analysis       
    results_columns = ['location',
                       'channel_range',
                       'wave_fit',
                       'prediction_fit',
                       'channel_slope',
                       'channel_position',
                       'channel_devition',
                       'amplitude',
                       'phase_shift',
                       'frequency',
                       'phase_position',
                       'tangent',
                       ]
    results = pd.DataFrame(results, columns = results_columns)
    results.set_index('location', drop=True, inplace=True)
    # Set errors array
    errors_coll = np.array(errors_coll)
    error_backup = errors_coll.copy()


###############################################################################
# Analysis: Determine if better wave fit leads to better wave fit predictions 
###############################################################################
if analysis:

    # Need to do per interval
        # Also- i think I wnated error terms to do per outcome length
    # Also a dist of the varirbales would be good....need a better built in one
    
    # Filter Results
    filtered = filter_results(results)
    # Simple.  Any correlation between model and Prediction fits ?
    results_fit_correlation = (np.corrcoef(filtered.loc[:, 'wave_fit'].values, 
                                           filtered.loc[:, 'prediction_fit'].values))[0, 1]
    print('Results fit correlation: {}'.format(results_fit_correlation))
    # Plot a scatter matrix of wave and prediction fit
    sns.jointplot('wave_fit', 'prediction_fit', kind='reg', data=filtered)
    
    



    
    if False:
        
        # Process the errors - do not have the process down yet
        new_errors = np.empty((errors_coll.shape[0],
                           prediction_range))
        for i in range(1, prediction_range+1):
            column = errors_coll[: ,channel_range: channel_range + i].mean(axis=1)
            new_errors[:, i - 1] = column
        to_add = np.tile(errors_coll[:, :channel_range].mean(axis=1).reshape(-1, 1), new_errors.shape[1])
        new_errors = new_errors + to_add
        
        # PLot I don't know.
        plt.figure(figsize=(11,6))
        plt.title('MSSE After channel starting at MSSE AT channel end')
        for i in np.linspace(results.index.values[0], 
                             results.index.values[-1],
                             20).astype(int):
            plt.plot(new_errors[channel_range - i])
        # plt.xlim((0, 25))
        # plt.ylim((0, .000005))
    
        # PLot error value art each location starting at MMSE at channel end
        plt.figure(figsize=(11,6))
        plt.title('Errors After channel starting at MSSE AT channel end')
        errors_after_channel = errors_coll[:, channel_range: ]
        errors_after_channel += to_add 
        for i in np.linspace(results.index.values[0], 
                             results.index.values[-1],
                             20).astype(int):
            plt.plot(errors_after_channel[channel_range - i])
    
        # PLot MMSE Distribution at channel end
        plt.figure()
        plt.title('Distribuition of MSSE at channel end')
        sns.distplot(new_errors[:, 0])
      




###############################################################################
# Graph (verification) Setup: wave, values, etc, using results location
###############################################################################
if graph_verification:

    # This should be done through results set so that we can filter.
    # With a candle location could get a part of day cycle as well.
    # Do I want to add a volatility graph to this as well?
    # Do I want to add a volue graph (universal volume?)
    
    # Get index of remaining (unfiltered results)
    filtered = filter_results(results)
    # Parameters
    graphs_to_make = 2
    channel_range = 60
    prediction_range = 15
    locations = filtered.index.values# np.arange(channel_range, values.shape[0] - prediction_range)
    random_locations = np.random.choice(locations, graphs_to_make)
    for location in random_locations:
        # Get values to plot 
        fit_values = values[location - channel_range: location]
        prediction_values = values[location: location + prediction_range]
        # Get wave and call plot method
        wave = Wave(fit_values)       
        wave_extension = wave.plot(prediction_values)
        print(location)

