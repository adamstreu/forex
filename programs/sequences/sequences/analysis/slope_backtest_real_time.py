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

'''
Slope backtest in real time.
For each new row in candle:
    get previous _backtest max length candles
    collect outcomes
    calculate slope.
    Add integer to placements if higher / lower than cutoff
    
    At end: evaluate placement outcomes
    
    side analysis:
        if already engaged, don't add another
'''

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
high_cutoff = .7
low_cutoff = -.7
direction = 0
shift = 0
length = 30

# Rwo and udo Parameters
position = (5, 5)
steps = np.arange(5, 16, 10)
position_final = (10, 10)

# Aquire Candles
candles = get_candles_from(pair, granularity, _from, _to)

# Aquire binary outcomes. Assign shift to be minimum bars.
up, down, udo, udnext, udmin = get_ud_bars(candles, steps)
rwo, rwmin = get_position_bars(candles, position, bar_limit)
rwo_final, rwmin_final = get_position_bars(candles, position_final, bar_limit)

# Set slope shift based on min bars average (not including unknonwn outcomes)
outcomes = rwo # or udo[0][0][direction]
shift = np.ceil(rwmin[direction][rwmin[direction] != bar_limit].mean()).astype(int)

# Aquire Slopes with varying windows
s30  = binary_slope(candles, outcomes, direction, 30,  shift)
s60  = binary_slope(candles, outcomes, direction, 60,  shift)
s120 = binary_slope(candles, outcomes, direction, 120, shift)

# Determine where all binary slope indicators show slope to be higher than param
take = candles[(s30 > high_cutoff) & \
               (s60 > high_cutoff) & \
               (s120 > high_cutoff)].index + 1
               
# Add take locations to candles.  Add Binary slopes
candles['positions'] = candles.midclose.min()
candles.loc[take, 'positions'] = candles.midclose.max()
candles['s30'] = s30
candles['s60'] = s60
candles['s120'] = s120

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
take = candles[(s30 < low_cutoff) & \
               (s60 < low_cutoff) & \
               (s120 < low_cutoff)].index + 1
'''
           

 
'''
# Take results from above index.  Print results for all steps
for u in range(udo.shape[0]):
    for d in range(udo.shape[1]):
        print('{} | {}:\t{:.2f}, {:.2f}'.format(u, 
                                                d, 
                                                udo[u][d][0][take].mean(),
                                                udo[u][d][1][take].mean()))
print(take.shape[0])
'''

