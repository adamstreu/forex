
# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from time import sleep
import os; os.chdir('/northbend')
from programs.slopes_position_manual.plot_currency_universe import plot_currency_universe    
from programs.slopes_position_manual.plot_currency_indicators import plot_currency_indicators 
from libraries.oanda import market
from libraries.oanda import get_time
from libraries.oanda import market
from libraries.oanda import get_candles
from libraries.oanda import get_multiple_candles_midclose
from libraries.oanda import get_multiple_candles_volume
from libraries.currency_universe import get_volume_universe_singular
from libraries.currency_universe import backfill_volume_with_singular

# Set Environment
pd.set_option('display.width', 1000)
pd.set_option('max.columns', 15) 
np.warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = [14, 4]





# General Parameters
_from = '2018-12-13T00:00:00Z'
_to   = '2019-12-01T00:00:00Z'
granularity = 'M15'
# Currencies to use
currencies = ['aud', 'cad', 'chf', 'eur', 'gbp', 'hkd', 'jpy', 'nzd', 'usd']     

# Get instrument List
instrument_list = []
for mark in market:
    if mark.split('_')[0] in [x.upper() for x in currencies]: 
        if mark.split('_')[1] in [x.upper() for x in currencies]:
            instrument_list.append(mark)    



volume, pair_volume = backfill_volume_with_singular(currencies, granularity, _from, _to)
volume.reset_index(inplace=True)
pair_volume.reset_index(inplace=True)














