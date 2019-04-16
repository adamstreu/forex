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
from libraries.indicators import local_range

# Instrument
pair = 'EUR_AUD'
granularity = 'M1'
daily_alignment = 17
# Time
_from = '2018-04-01T00:00:00Z'
_to = '2019-01-01T00:00:00Z'

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




'''
Some decent Results

pair = 'EUR_AUD'
granularity = 'M5'
_from = '2016-01-01T00:00:00Z'
_to = '2019-01-01T00:00:00Z;

lengths = np.linspace(30, 600, 10).astype(int)
current_window = 50 ( 30, 15, 100 has same outcome (?))
mms_take = mms[mms.min_short & mms.max_short].index + 1

Position  5: [ 0.46440678  0.48474576]
Position 10: [ 0.5220339   0.46779661]
Position 20: [ 0.41694915  0.61016949]
Position 30: [ 0.56271186  0.49152542]
Position 40: [ 0.36610169  0.66779661]
Position 50: [ 0.2440678   0.73898305]




# Gather Takes
take_up   = mms[mms.slope_up   & mms.down_current].index + 1
take_down = mms[mms.slope_down & mms.up_current].index + 1
take_down = mms[mms.slope_down & mms.up_current].index + 1
occurances 60 steps apart to count.
First Occurances from take_down_coll
Seperate Occurances Found: 12
Position  5: [ 0.41666667  0.25      ]
Position 10: [ 0.75  0.25]
Position 20: [ 0.66666667  0.33333333]
Position 40: [ 0.66666667  0.33333333]
Position 60: [ 0.58333333  0.41666667]
Position 80: [ 0.83333333  0.16666667]
Position 100: [ 0.83333333  0.16666667]
Position 120: [ 0.83333333  0.16666667]
Position 140: [ 0.75  0.25]
lengths = np.linspace(20, 1000, 6).astype(int)
current_window = 15


'''
