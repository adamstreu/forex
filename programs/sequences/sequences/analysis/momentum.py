import numpy as np
import pandas as pd
import os; os.chdir('/sequences')
from libraries.oanda import get_candles_from
from libraries.sequences import get_ud_bars
from libraries.sequences import get_position_bars
from libraries.sequences import get_positions
from libraries.sequences import create_sequences
from libraries.indicators import stochastic_oscillator
from libraries.indicators import bollinger_bands

# Instrument
pair = 'EUR_AUD'
granularity = 'H1'
daily_alignment = 17

# Time
_from = '2018-01-01T00:00:00Z'
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


m5 = get_candles_from(pair, 'M5', _from, _to)
m5 = 