ngimport numpy as np
import pandas as pd
from scipy.stats import binom_test
from matplotlib import pyplot as plt
from sklearn.metrics import precision_score as ps
import os; os.chdir('/sequences')
from libraries.oanda import get_candles_from
from libraries.outcomes import get_ud_bars
from libraries.outcomes import get_position_bars
from libraries.sequences import get_positions
from libraries.indicators import stochastic_oscillator
from libraries.indicators import bollinger_bands
from libraries.indicators import binary_slope
from libraries.indicators import min_max_slope
from libraries.indicators import local_range

'''


Today:
    Binary slope combined with bollinger.
    Use binary slope to test how strong the slope is.
    
        or
        
        ust give slope a rating between -1 and 1.  If strong, up, and bolingers are 
        at bottom, see what happens.
    
    Also, 
    
        Try evaluating bollingers at different granularityes then peicing 
        them back toegeth, all using same window length.



We have three types of slopes right now:
    Binary
    Min_max
    bollinger
    
Can we combine these three for some predictive power?


'''

# Instrument
pair = 'EUR_AUD'
granularity = 'M1'
daily_alignment = 17
# Time
_from = '2016-04-01T00:00:00Z'
_to = '2019-01-01T00:00:00Z'
# Filters
bar_limit = 4000



'''
# Candles and Outcomes
###############################################################################
# Aquire Candles
candles = get_candles_from(pair, granularity, _from, _to)
# Aquire binary outcomes. Assign shift to be avg bars.
rwo5 ,  rwmin5   = get_position_bars(candles, (5 ,  5  ))
rwo10,  rwmin10  = get_position_bars(candles, (10,  10 ))
rwo20,  rwmin20  = get_position_bars(candles, (20,  20 ))
rwo40,  rwmin40  = get_position_bars(candles, (40,  40 ))
rwo60,  rwmin60  = get_position_bars(candles, (60,  60 ))
rwo80,  rwmin80  = get_position_bars(candles, (80,  80 ))
rwo100, rwmin100 = get_position_bars(candles, (100, 100))
rwo120, rwmin120 = get_position_bars(candles, (120,  120 ))
rwo140, rwmin140 = get_position_bars(candles, (140,  140 ))
rwo180, rwmin180 = get_position_bars(candles, (180,  180 ))
rwo240, rwmin240 = get_position_bars(candles, (240,  240 ))
rwo320, rwmin320 = get_position_bars(candles, (320,  320 ))
###############################################################################
'''


try:
    # Binary Slope
    ###############################################################################
    # Binary Slope Parameters
    high_cutoff = .65
    low_cutoff = -.65
    direction = 1
    shift = 0
    length = 30
    unique_length = 60
    # Set slope shift based on min bars average (not including unknonwn outcomes)
    outcomes = rwo5[direction] # or udo[0][0][direction]
    shift = np.ceil(rwmin[direction][rwmin[direction] != \
                    bar_limit].mean()).astype(int)
    # Aquire Slopes with varying windows
    s30  = binary_slope(candles, outcomes, 30,  shift)
    s60  = binary_slope(candles, outcomes, 60,  shift)
    s120 = binary_slope(candles, outcomes, 120, shift)
    # Determine where all binary slope indicators show slope to be highr than param
    take_binary_high = candles[(s30 > high_cutoff) & \
                               (s60 > high_cutoff) & \
                               (s120 > high_cutoff)].index + 1  
    take_binary_low  = candles[(s30 < low_cutoff) & \
                              (s60 < low_cutoff) & \
                              (s120 < low_cutoff)].index + 1
    take_binary_high = take_binary_high[:-1]
    take_binary_low = take_binary_low[:-1]
    take_binary_up_unique = [take_binary_high[0]]
    take_binary_down_unique = [take_binary_low[0]]
    for i in range(len(take_binary_high)):
        if take_binary_up_unique[-1] + unique_length < take_binary_high[i]:
            take_binary_up_unique.append(take_binary_high[i])
    for i in range(len(take_binary_low)):
        if take_binary_down_unique[-1] + unique_length < take_binary_low[i]:
            take_binary_down_unique.append(take_binary_low[i])
    # Print Out Results
    print()
    msg = 'Binary high positions, unique: {}, {}'                     
    print(msg.format(take_binary_high.shape[0],
                     len(take_binary_up_unique)))
    msg = 'Binary low positions, unique: {}, {}'  
    print(msg.format(take_binary_low.shape[0],
                     len(take_binary_down_unique)))
    
    for out, bar in zip([rwo5, rwo10, rwo20, rwo40, rwo60, rwo80, 
                         rwo100, rwo120, rwo140, rwo180, rwo240, rwo320],
                        [rwmin5, rwmin10, rwmin20, rwmin40, rwmin60, rwmin80, 
                         rwmin100, rwmin120, rwmin140, rwmin180, 
                         rwmin240, rwmin320]):
        print('{}:\t\t{}\t,\t{},\t{}'.format(namestr(out, globals())[0], 
                                          out[:, take_binary_high].mean(axis=1), 
                                          out[:, take_binary_low].mean(axis=1),
                                          bar[0][bar[0] != bar_limit].mean()))
    
    '''print()
    print('Unique')
    for out in [rwo5, rwo10, rwo20, rwo40, rwo60, rwo80, 
                     rwo100, rwo120, rwo140, rwo180, rwo240, rwo320]:
        print('{}:\t\t{}\t,\t{}'.format(namestr(out, globals())[0], 
                                          out[:, np.array(take_binary_up_unique)].mean(axis=1), 
                                          out[:, np.array(take_binary_down_unique)].mean(axis=1)))
    print()
    '''
    '''      
    # Add to results Df and graph
    # Add take locations to candles.  Add Binary slopes
    candles['positions'] = candles.midclose.min()
    candles.loc[take, 'positions'] = candles.midclose.max()
    candles['s30'] = s30
    candles['s60'] = s60
    candles['s120'] = s120
    '''
    ###############################################################################
except Exception as e:
    print('Binary Slop found no instances: {}'.format(e))
    

try:
    # Min Max Slope
    ###############################################################################
    # min_max_slope parameters
    lengths = [45, 90, 180, 320, 640, 1280, 2560]
    lengths = np.linspace(60, 1000, 3).astype(int)
    current_window = 30
    # Call min_max DataFrame
    mms = min_max_slope(candles, lengths, current_window)
    # Gather Takes
    take_up   = mms[mms.slope_up   & mms.down_current].index + 1
    take_down = mms[mms.slope_down & mms.up_current].index + 1
    take_up = take_up[:-1]
    take_down = take_down[:-1]
    # Gather outcomes from takes sub-selection
    take_coll_up = [take_up[0]]
    for i in range(len(take_up)):
        if take_coll_up[-1] + 60 < take_up[i]:
            take_coll_up.append(take_up[i])
    take_coll_down = [take_down[0]]
    for i in range(len(take_down)):
        if take_coll_down[-1] + 60 < take_down[i]:
            take_coll_down.append(take_down[i])
    # Print Results
    print()
    msg = 'Min_max high positions, unique: {}, {}'                     
    print(msg.format(take_up.shape[0], len(take_coll_up)))
    msg = 'Min_max low positions, unique:  {}, {}'  
    print(msg.format(take_down.shape[0], len(take_coll_down)))
    for out, bar in zip([rwo5, rwo10, rwo20, rwo40, rwo60, rwo80, 
                         rwo100, rwo120, rwo140, rwo180, rwo240, rwo320],
                        [rwmin5, rwmin10, rwmin20, rwmin40, rwmin60, rwmin80, 
                         rwmin100, rwmin120, rwmin140, rwmin180, 
                         rwmin240, rwmin320]):
        print('{}:\t\t{}\t,\t{}'.format(namestr(out, globals())[0], 
                                          out[:, take_up].mean(axis=1), 
                                          out[:, take_down].mean(axis=1)))
    print()
            
    ###############################################################################
except Exception as e:
    print('Min_Max slope found no instances: {}'.format(e))







#Compile results, intersections, etc
###############################################################################
print()
int_up = np.intersect1d(take_up, take_binary_high)
int_down = np.intersect1d(take_down, take_binary_low)
print(int_up.shape[0])
print(int_down.shape[0])
for out in [rwo5, rwo10, rwo20, rwo40, rwo60, rwo80, 
                 rwo100, rwo120, rwo140, rwo180, rwo240, rwo320]:
    print('{}:\t\t{}\t,\t{}'.format(namestr(out, globals())[0], 
                                      out[:, int_up].mean(axis=1), 
                                      out[:, int_down].mean(axis=1)))


'''
        
        
        
        
# Plot results
candles.positions.plot(figsize=(12, 1))
candles.midclose.plot(figsize=(12, 4))
plt.figure()
candles.loc[:1000, ['s30', 's60', 's120']].plot(figsize=(12, 4))
plt.figure()
candles.loc[:1000, 'midclose'].plot(figsize=(12, 4))

# Print Summary
print('Conditions found: \t\t\t{}'.format(take.shape[0]))
print('Predictions correct percentage: \t{:.2f}'.format(rwo[direction][take].mean()))
print()
print('Average Bars to outcome: \t\t{}'.format(shift))
print('Original rwo precentage: \t\t{:.2f}'.format(rwo[direction].mean()))





'''



'''
# Call min_max slope function.  

# Gather outcomes from takes selection
print('Slope up')
print('Qty Found  : {}'.format(take_up.shape[0]))
print('Position  5: {}'.format(rwo5[ :, take_up].mean(axis=1)))
print('Position 10: {}'.format(rwo10[:, take_up].mean(axis=1)))
print('Position 20: {}'.format(rwo20[:, take_up].mean(axis=1)))
print('Position 40: {}'.format(rwo40[:, take_up].mean(axis=1)))
print('Position 60: {}'.format(rwo60[:, take_up].mean(axis=1)))
print('Position 80: {}'.format(rwo80[:, take_up].mean(axis=1)))
print('Position 100: {}'.format(rwo100[:, take_up].mean(axis=1)))
print('Position 120: {}'.format(rwo120[:, take_up].mean(axis=1)))
print('Position 140: {}'.format(rwo140[:, take_up].mean(axis=1)))
print('Position 180: {}'.format(rwo180[:, take_up].mean(axis=1)))
print('Position 240: {}'.format(rwo240[:, take_up].mean(axis=1)))
print('Position 320: {}'.format(rwo320[:, take_up].mean(axis=1)))
print()
print('Slope down')
print('Qty Found  : {}'.format(take_down.shape[0]))
print('Position  5: {}'.format(rwo5[ :, take_down].mean(axis=1)))
print('Position 10: {}'.format(rwo10[:, take_down].mean(axis=1)))
print('Position 20: {}'.format(rwo20[:, take_down].mean(axis=1)))
print('Position 40: {}'.format(rwo40[:, take_down].mean(axis=1)))
print('Position 60: {}'.format(rwo60[:, take_down].mean(axis=1)))
print('Position 80: {}'.format(rwo80[:, take_down].mean(axis=1)))
print('Position 100: {}'.format(rwo100[:, take_down].mean(axis=1)))
print('Position 120: {}'.format(rwo120[:, take_down].mean(axis=1)))
print('Position 140: {}'.format(rwo140[:, take_down].mean(axis=1)))
print('Position 180: {}'.format(rwo180[:, take_down].mean(axis=1)))
print('Position 240: {}'.format(rwo240[:, take_down].mean(axis=1)))
print('Position 320: {}'.format(rwo320[:, take_down].mean(axis=1)))
print('------')
print('First Occurances from take_up_coll')
print('Seperate Occurances Found: {}'.format(len(take_coll_up)))
print('Position  5: {}'.format(rwo5[ :, take_coll_up].mean(axis=1)))
print('Position 10: {}'.format(rwo10[:, take_coll_up].mean(axis=1)))
print('Position 20: {}'.format(rwo20[:, take_coll_up].mean(axis=1)))
print('Position 40: {}'.format(rwo40[:, take_coll_up].mean(axis=1)))
print('Position 60: {}'.format(rwo60[:, take_coll_up].mean(axis=1)))
print('Position 80: {}'.format(rwo80[:, take_coll_up].mean(axis=1)))
print('Position 100: {}'.format(rwo100[:, take_coll_up].mean(axis=1)))
print('Position 120: {}'.format(rwo120[:, take_coll_up].mean(axis=1)))
print('Position 140: {}'.format(rwo140[:, take_coll_up].mean(axis=1)))
print('Position 180: {}'.format(rwo180[:, take_coll_up].mean(axis=1)))
print('Position 240: {}'.format(rwo240[:, take_coll_up].mean(axis=1)))
print('Position 320: {}'.format(rwo320[:, take_coll_up].mean(axis=1)))
print()
print('First Occurances from take_down_coll')
print('Seperate Occurances Found: {}'.format(len(take_coll_down)))
print('Position  5: {}'.format(rwo5[ :, take_coll_down].mean(axis=1)))
print('Position 10: {}'.format(rwo10[:, take_coll_down].mean(axis=1)))
print('Position 20: {}'.format(rwo20[:, take_coll_down].mean(axis=1)))
print('Position 40: {}'.format(rwo40[:, take_coll_down].mean(axis=1)))
print('Position 60: {}'.format(rwo60[:, take_coll_down].mean(axis=1)))
print('Position 80: {}'.format(rwo80[:, take_coll_down].mean(axis=1)))
print('Position 100: {}'.format(rwo100[:, take_coll_down].mean(axis=1)))
print('Position 120: {}'.format(rwo120[:, take_coll_down].mean(axis=1)))
print('Position 140: {}'.format(rwo140[:, take_coll_down].mean(axis=1)))
print('Position 180: {}'.format(rwo180[:, take_coll_down].mean(axis=1)))
print('Position 240: {}'.format(rwo240[:, take_coll_down].mean(axis=1)))
print('Position 320: {}'.format(rwo320[:, take_coll_down].mean(axis=1)))
'''
'''
# Plot Take Up Occurance 
results = pd.DataFrame(candles[['timestamp', 'midclose']])
results['take_up']  = candles.midclose.min()
results.loc[take_up,  'take_up'] = candles.loc[take_up, 'midclose']
results['take_down']  = candles.midclose.min()
results.loc[take_down,  'take_down'] = candles.loc[take_down, 'midclose']
results[['midclose', 'take_up', 'take_down']].plot(figsize=(16, 4))
results.loc[30000:35000, ['midclose', 'take_up', 'take_down']].plot(figsize=(16, 4))


'''
