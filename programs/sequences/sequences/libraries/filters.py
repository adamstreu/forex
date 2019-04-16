import numpy as np
from scipy.stats import binom_test

binom_filter= .001  # 1 / ((len(steps) ** (2 * seq_len)) * (2 ** seq_len) * 2)

def binomial_filter(df, bf=binom_filter):
    binom_test_v = np.vectorize(binom_test)
    df['binom'] =  binom_test_v(df.win_perc * df.placements, 
                                           df.placements, df.p_win_perc)
    df = df[df.binom < bf]
    return df