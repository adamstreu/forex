import pandas as pd

def stochastic_oscillator(df, periods_k, periods_d):
    close = df.midclose.values
    high = pd.rolling_max(df.midhigh, periods_k, how='max')
    low = pd.rolling_min(df.midlow, periods_k, how='min') 
    k = ((close - low) / (high - low)) * 100
    d = pd.rolling_mean(k, periods_d)
    return d

if __name__ == '__main__':
    pass






    '''
    
    #from backtest_new import get_candles
    
    ## Instrument 
    #currency = 'GBP_USD'
    #granularity = 'M15'
    #
    ## Time
    #_from = '2018-01-01T00:00:00Z'
    #_to =   '2019-01-01T00:00:00Z'  
    #daily_alignment = 17
    #periods_k = 25
    #periods_d = 5
    
    
    def timed_all():
        o = (0, 1, 0)
        # s = ((50, 50), (325, 50), (50, 75))
        s = ((1, 1), (3, 1), (1, 2))
        return np.unique((udnext[s[2][0]][s[2][1]] * udo[s[2][0]][s[2][1]][o[3]])[(udnext[3][1] * udo[3][1][1]) \
                        [(udnext[1][1] * udo[1][1][0])[(udnext[1][1] * udo[1][1][0]) != 0]] \
                        [(udnext[3][1] * udo[3][1][1])[(udnext[1][1] * udo[1][1][0]) \
                        [(udnext[1][1] * udo[1][1][0]) != 0]] !=0 ]] \
                        [(udnext[1][2] * udo[1][2][0])[(udnext[3][1] * udo[3][1][1]) \
                        [(udnext[1][1] * udo[1][1][0])[(udnext[1][1] * udo[1][1][0]) != 0]] \
                        [(udnext[3][1] * udo[3][1][1])[(udnext[1][1] * udo[1][1][0]) \
                        [(udnext[1][1] * udo[1][1][0]) != 0]] !=0 ]] != 0])
       '''
    
        
        
        

