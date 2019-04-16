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


'''
When bollinger backtests from multiple timelines are condition (above upper),
    do we have a good correlation with an up / down move?
    
'''


# Instrument
pair = 'GBP_CAD'
granularity = 'M5'
daily_alignment = 17
# Time
_from = '2018-01-01T00:00:00Z'
_to = '2019-01-01T00:00:00Z'
# Filters
bar_limit = 4000
# Bollinger Parameters
length = 10
std = 3


candles = get_candles_from(pair, granularity, _from, _to)
steps = np.arange(5, 100, 5)
up, down, udo, udnext, udmin = get_ud_bars(candles, steps)

b60 = bollinger_bands(candles, 240, std)
b30 = bollinger_bands(candles, 120, std)
b15 = bollinger_bands(candles,  60, std)
b5  = bollinger_bands(candles, 20, std)

take = (candles.index[((candles.midhigh > b30.upper) & \
                       (candles.midhigh > b60.upper) & \
                       (candles.midhigh > b15.upper) & \
                       (candles.midhigh > b5.upper)) == True] + 1).tolist()

take = (candles.index[((candles.midlow < b30.lower) & \
                       (candles.midlow < b60.lower) & \
                       (candles.midlow < b15.lower) & \
                       (candles.midlow <  b5.lower)) == True] + 1).tolist()

results = udo[:, :, :, take].mean(axis=3)
for i in range(steps.shape[0]):
    print(udo[i, i, :,take].mean(axis=0))

