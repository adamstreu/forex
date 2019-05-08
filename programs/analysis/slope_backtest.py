import numpy as np
import pandas as pd
from scipy.stats import binom_test
from matplotlib import pyplot as plt
from sklearn.metrics import precision_score as ps
import os; os.chdir('/sequences')
from libraries.oanda import get_candles_from
from libraries.sequences import get_ud_bars
from libraries.sequences import get_position_bars
from libraries.sequences import get_positions
from libraries.indicators import stochastic_oscillator
from libraries.indicators import bollinger_bands
from libraries.indicators import binary_slope
from libraries.indicators import min_max_slope


# Instrument
pair = 'EUR_AUD'
granularity = 'M1'
daily_alignment = 17
# Time
_from = '2018-04-01T00:00:00Z'
_to = '2019-01-01T00:00:00Z'
# Filters
bar_limit = 4000
# Slope Parameters
high_cutoff = .95
low_cutoff = -.95
direction = 1
shift = 13
length = 30
# Rwo and udo Parameters
position = (7, 7)
position_final = (20, 20)
# min_max_slope parameters
lengths = [30, 60, 120, 240, 480, 960, 1920, 3840, 7680]


if direction == 0:
    final_direction = 1
else:
    final_direction  = 0

# Aquire Candles
candles = get_candles_from(pair, granularity, _from, _to)

# Aquire binary outcomes. Assign shift to be avg bars.
rwo, rwmin = get_position_bars(candles, position, bar_limit)
rwo_final, rwmin_final = get_position_bars(candles, position_final, bar_limit)

# Set slope shift based on min bars average (not including unknonwn outcomes)
outcomes = rwo[direction] # or udo[0][0][direction]
shift = np.ceil(rwmin[direction][rwmin[direction] != bar_limit].mean()).astype(int)

# Aquire Slopes with varying windows
s30  = binary_slope(candles, outcomes, 30,  shift)
s60  = binary_slope(candles, outcomes, 60,  shift)
s120 = binary_slope(candles, outcomes, 120, shift)
s960 = binary_slope(candles, outcomes, 960, shift)

# Determine where all binary slope indicators show slope to be higher than param
take_long  = candles[(s30  >= high_cutoff) & \
                     (s60  >= high_cutoff) & \
                     (s120 >= high_cutoff)].index + 1
take_short = candles[(s30  <= low_cutoff) & \
                     (s60  <= low_cutoff) & \
                     (s120  <= low_cutoff)].index + 1
if candles.shape[0] in take_long:
    take_long = take_long[:-1]
if candles.shape[0] in take_short:
    take_short = take_short[:-1]                    
              
# Add take locations to candles.  Add Binary slopes
candles['take_long']  = candles.midclose.min()
candles.loc[take_long,  'take_long'] = candles.midclose.max()
candles['take_short'] = candles.midclose.min()
candles.loc[take_short, 'take_short'] = candles.midclose.max()
candles['s30'] = s30
candles['s60'] = s60
candles['s120'] = s120
candles['s960'] = s960

# Plot results
candles.take_long.plot(figsize=(12, 1))
candles.take_short.plot(figsize=(12, 1))
candles.midclose.plot(figsize=(12, 1))
plt.figure()
candles.loc[30000:31000, ['s30', 's60', 's120', 's960']].plot(figsize=(12, 3))
plt.figure()
candles.loc[30000:31000, 'midclose'].plot(figsize=(12, 3))

# Print Summary
print()
print('Take Long: ')
print('----------------------------------------------------------')
print('Conditions found: \t\t\t{}'.format(take_long.shape[0]))
print('Conditions found %: \t\t\t{:.2f}'.format(take_long.shape[0]/candles.shape[0]))
print()
print('Rwo Predict correct %: \t\t\t{:.2f}'.format(rwo[direction][take_long].mean()))
print('Rwo Final correct %: \t\t\t{:.2f}'.format(rwo_final[direction][take_long].mean()))
print('Original rwo win %: \t\t\t{:.2f}'.format(rwo[direction].mean()))
print()
print('Average Bars to outcome: \t\t{}'.format(shift))
print('----------------------------------------------------------')
print()
print('Take Short: ')
print('----------------------------------------------------------')
print('Conditions found: \t\t\t{}'.format(take_short.shape[0]))
print('Conditions found %: \t\t\t{:.2f}'.format(take_short.shape[0]/candles.shape[0]))
print()
print('Rwo Predict correct %: \t\t\t{:.2f}'.format(rwo[direction][take_short].mean()))
print('Rwo Final correct %: \t\t\t{:.2f}'.format(rwo_final[direction][take_short].mean()))
print('Original rwo win %: \t\t\t{:.2f}'.format(rwo[direction].mean()))
print()
print('Average Bars to outcome: \t\t{}'.format(shift))
print('----------------------------------------------------------')

'''
results = pd.DataFrame()
mms = min_max_slope(candles, lengths)
mms_take = mms[mms.min_long & mms.max_long == True].index + 1
rwo[:, mms_take].mean(axis=1)
results['take_long']  = candles.midclose.min()
results.loc[mms_take,  'take_long'] = candles.midclose.max()
results['midclose'] = candles.midclose
results[['midclose', 'take_long'], axis=1].plot()
'''



# Min Max Slope
###############################################################################
###############################################################################

'''
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
'''

# min_max_slope parameters
lengths = [45, 90, 180, 320, 640, 1280, 2560]
lengths = np.linspace(60, 1000, 7).astype(int)
current_window = 80

# Call min_max slope function.  
mms = min_max_slope(candles, lengths, current_window)

# Gather Takes
take_up   = mms[mms.slope_up   & mms.down_current].index + 1
take_down = mms[mms.slope_down & mms.up_current].index + 1
'''
take_up   = mms[mms.slope_up  ].index + 1
take_down = mms[mms.slope_down].index + 1
'''
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

# Plot Take Up Occurance 
results = pd.DataFrame(candles[['timestamp', 'midclose']])
results['take_up']  = candles.midclose.min()
results.loc[take_up,  'take_up'] = candles.loc[take_up, 'midclose']
results['take_down']  = candles.midclose.min()
results.loc[take_down,  'take_down'] = candles.loc[take_down, 'midclose']
results[['midclose', 'take_up', 'take_down']].plot(figsize=(16, 4))
results.loc[30000:35000, ['midclose', 'take_up', 'take_down']].plot(figsize=(16, 4))

###############################################################################
###############################################################################
