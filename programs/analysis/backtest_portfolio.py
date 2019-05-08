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


# Instrument
currency = 'EUR_AUD'
granularity = 'H1'
daily_alignment = 17

# Time
_from = '2010-01-01T00:00:00Z'
_to = '2019-01-01T00:00:00Z'

# Sequence
outcomes = (0, 0, 0, 1, 0)
sequence = ((375, 75), (75, 275), (175, 75), (375, 175), (75, 75))
position = (75, 75)
direction = 0

# Filters
bar_limit = 4000
placements_filter = 0
win_perc_filter = .9
binom_filter= .001  # 1 / ((len(steps) ** (2 * seq_len)) * (2 ** seq_len) * 2)
return_filter = 0

# Stochastic Indicator Parameters
so_k = 45
so_d = 15

length = 10
std = 2


if __name__ == '__main__': 
    

    # Normal Setup
    steps = np.array(sorted(list(set([25] + list(np.unique(sequence))))))
    seq_len = len(sequence)
    candles = get_candles_from(currency, granularity, _from, _to, ) 
    up, down, udo, udnext, udmin = get_ud_bars(candles, steps)
    rwo, rwmin = get_position_bars(candles, position)
    # Get Placements and calculate simple statistics
    take = get_positions(candles, steps, sequence, outcomes, udo, udnext)
    results = rwo[direction][take]
    win_perc = results.mean()
    binom = binom_test(results.sum(), results.shape[0], rwo[direction].mean())
    volume = candles.loc[take, 'volume']
    spreads = candles.loc[take, 'spread']
    # Stochastic Indicator
    so = stochastic_oscillator(candles, so_k, so_d)  # 10 5 seems descent so far
    if direction == 1:
        stochastic = so[take] > .7
        so10 = rwo[direction][take][so[take] <= 10]
        so20 = rwo[direction][take][so[take] <= 20]
        so30 = rwo[direction][take][so[take] <= 30] 
    else:
        stochastic = so[take] < .3
        so10 = rwo[direction][take][so[take] >= 90]
        so20 = rwo[direction][take][so[take] >= 80]
        so30 = rwo[direction][take][so[take] >= 70]
    # Bollinger Bands
    bb = bollinger_bands(candles, length, std)
    if direction == 1:
        bollinger = (bb.sma < bb.midclose)[take]
    else:
        bollinger = (bb.sma >= bb.midclose)[take]
    # Print Results for each position
    print('timestamp, outcome, bars, stoch osc, spread, volume')
    for each in list(zip(candles.loc[take,'timestamp'], 
                         results, 
                         rwmin[direction][take], 
                         so[take],
                         spreads,
                         volume, 
                         bollinger)): 
        print('{}\t{}\t{}\t{:.0f}\t{:.4f}\t{}\t{}'.format(each[0], each[1], 
                                                  each[2], each[3], 
                                                  each[4], each[5], each[6]))
    # Print Precision Scores
    print()
    print('Presision score: \t\t{:.02}\t from {}'.format(results.mean(), results.shape[0]))
    print('Bollinger Presision score: \t{:.2f}\t from {}'.format(ps(results, bollinger), bollinger.sum()))
    print('Stochastic Presision score: \t{:.2f}\t from {}'.format(ps(results, stochastic), stochastic.sum()))
    print('Combined Presision score: \t{:.2f}\t from {}'.format(ps(results, np.logical_and(stochastic, bollinger)), np.logical_and(stochastic, bollinger).sum()))
    # Print Binom
    print('Binom: \t{:.6f}'.format(binom))
    
    
    
    
    
    # Print stochastic Indicator Statsitcs
    print('so 10: {:.2f}, {}, {:.2f}'.format(so10.mean(),
                                         so10.shape[0],
                                         binom_test(so10.sum(),
                                                    so10.shape[0],
                                                    results.mean())))
    print('so 20: {:.2f}, {}, {:.2f}'.format(so20.mean(),
                                 so20.shape[0],
                                 binom_test(so20.sum(),
                                            so20.shape[0],
                                            results.mean())))
    print('so 30: {:.2f}, {}, {:.2f},'.format(so30.mean(),
                                         so30.shape[0],
                                         binom_test(so30.sum(),
                                                    so30.shape[0],
                                                    results.mean())))
    # Plot Graphs
    plt.plot(so[take], results, 'o')
    bb.loc[take,['sma', 'midclose']].plot(style='o', figsize=(16, 8))
    

        
        
    
