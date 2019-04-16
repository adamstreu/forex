import pandas as pd
import numpy as np
from scipy.linalg import lu
import os; os.chdir('/northbend')
from libraries.oanda import get_candles

    
class Currencies(granularity, _from, _to, column='midclose'):
    
    '''
    The instruments and ratios are set right now.
    This function removes the currencies from ratios.
    '''
    
    def init(self, pair_names, pair_value):
        pass
        # Not doing anything with this yet.....
    
    
    
    
    def currency_df(pair_names, pair_values):
    
        '''
        Inputs: 
                each c urrency should be input as name like gbp_usd.  Order matters
                Should be as array of all time values wanted.    
                Each currency needs to have a link - and only one - to every other
                currency names currnecy values have to have same location in lists
    
        WARNING:
                Def need to go through each step checking values
        '''
        
        # preliminary organization
        # -------------------------------------------------------------------------
        # set up pair dictionary
        pair_d = {}
        for i in range(len(pair_names)):
            pair_d[pair_names[i]] = pair_values[i]
        # Collect currency names to withdraw from ratios.  Alphabetize
        currency_collection = []
        for pn in pair_names:
            currency_collection += pn.split('_')
        currency_collection = list(set(currency_collection))
        currency_strings = pd.Series(currency_collection).sort_values().values    
        # Create currency dictionary.  This will be converted to df for return
        currency_d = {}
        for i in range(len(currency_collection)):
            currency_d[currency_collection[i]] = 0
    
        # Main iteration.
        # ---------------------------------------------------------------------
        for currency in currency_d.keys():
            pairs_list = []
            for k in pair_d.keys():
                if currency in str(k):
                    pairs_list.append(k)
            pairs_list = pd.Series(pairs_list).sort_values().values
    
            # Create Coefficient array.
            # ---------------------------------------------------------------------
            # Create array to say weather to use the coefficeint or its inverse.
            inverse_array = []
            for pair in pairs_list:
                if pair.index(currency) == 0:
                    inverse_array.append(-1)
                else:
                    inverse_array.append(1)
            inverse_array = np.array(inverse_array)
            # Get all pairs to use.  Sort them into alhebetical order.  
            pairs = np.empty((pairs_list.shape[0], pair_d[pairs_list[0]].shape[0]))
            for i in range(pairs_list.shape[0]):
                if inverse_array[i] == 1:
                    pairs[i] = pair_d[pairs_list[i]]
                else:
                    pairs[i] = 1 / pair_d[pairs_list[i]]
                    
            # Calculate currency.  Use that to calculate the rest
            # ---------------------------------------------------------------------
            # Calculate prime currency
            currency_value = 1 / (pairs.sum(axis=0) + 1)
            # Calculate all others based on above
            all_currencies = pairs * currency_value
            # insert currency back into matrix so all our togehter (alphabetically)
            insert_here = np.where(currency_strings == currency)[0][0]
            insert_here *= pair_d[pairs_list[0]].shape[0]
            all_currencies = np.insert(all_currencies, insert_here, currency_value)
            all_currencies = all_currencies.reshape(pairs_list.shape[0] + 1, -1)
            # Add to dictionary
            currency_d[currency] = all_currencies
        
        
        # Create DataFrame and return
        # -----------------------------------------------------------------------------
        # instantiate size
        all_values = np.empty((pair_d[pairs_list[0]].shape[0] \
                               * currency_d[currency_strings[0]].shape[0], 
                              currency_d[currency_strings[0]].shape[0] ))
        # For each currency, reshape by column.  Put into alphabaetical order
        for i in range(currency_strings.shape[0]): 
            all_values[:, i] = currency_d[currency_strings[i]]\
                               .reshape(-1, 1, order='F').ravel()
        # Create df indexes.  Location might create some difficuly if not enter ok
        locations = np.repeat(np.arange(pair_d[pair_names[0]].shape[0]), 
                              currency_strings.shape[0])
        currency_index = np.tile(currency_strings, pair_d[pairs_list[0]].shape[0] )
        # Create Dataframe
        values = pd.DataFrame(all_values, 
                              columns=currency_strings,
                              index = [locations, currency_index ])
        
        return values
    
    
    instruments = {
                   'EUR_USD': [], 
                   'EUR_AUD': [], 
                   'EUR_CAD': [],
                   'AUD_USD': [], 
                   'AUD_CAD': [],
                   'USD_CAD': [],
                   'GBP_USD': [],
                   'EUR_GBP': [],
                   'GBP_AUD': [],
                   'GBP_CAD': [],
                   'GBP_NZD': [],
                   'AUD_NZD': [],
                   'EUR_NZD': [],
                   'NZD_CAD': [],
                   'NZD_USD': [],
                   }
    instrument_list = list(instruments.keys())
    # Call all instruments.  get column and timestamp as index
    for instrument in instrument_list:
         df = get_candles(instrument, granularity, _from, _to)[['timestamp', column]]
         df.set_index('timestamp', inplace=True, drop=True)
         instruments[instrument] = df
    # join all to one index to make sure times align.  Backfill missing
    for instrument in instrument_list[1:]:
        instruments[instrument_list[0]] = instruments[instrument_list[0]].join(instruments[instrument], lsuffix='.')
    instruments[instrument_list[0]].fillna(method='backfill', inplace=True)
    # Convert completed value set to unlabeled array in order of instrument_list
    pair_values = instruments[instrument_list[0]].T.values
    pair_names  = [x.lower() for x in instrument_list]   
    # Get names of currencies
    currencies = []
    for pn in pair_names:
        currencies += pn.split('_')
    currencies = list(set(currencies))
    currencies = pd.Series(currencies).sort_values().values 
    # Call Currency Matrix.
    currency_matrix  = currency_df(pair_names, pair_values)
    cur = currency_matrix.copy()    
    cur.index = cur.index.swaplevel(0,1)
    currency_set = cur.copy()
    # Just work with the Primaries
    primaries = pd.DataFrame()
    for col in cur.columns:
        primaries[col] = cur.loc[col, col]
    cur = primaries.copy()    
    # Create Ratios DataFrame (from pair_values and pair_names)
    ratios = pd.DataFrame(pair_values.T, columns=np.array(pair_names))
    # Create currency differernce dataframe
    curdiff = cur.rolling(window=2).apply(lambda x: x[1] - x[0])
    # Get index of values to drop
    filter_nas = cur[(pd.isna(curdiff) == True).sum(axis=1) > 0].index.values
    # Drop values and reindex
    for df in [ratios, cur, curdiff]:
        df.drop(filter_nas, inplace=True)
        df.reset_index(drop=True, inplace=True)
    
    
    return {'currencies': cur,
           'currency_set': currency_set,
           'ratios': ratios,
           'currencies_delta': curdiff}


# This would be better as a method of a currency class
def get_currency_rolling_correlations(dataframe, pair_names, window):
    df = dataframe.rolling(window).corr()
    df.index = df.index.swaplevel(0,1)
    corr = pd.DataFrame()
    for currencies in pair_names:
        cur1, cur2 = currencies.split('_')
        corr[currencies] = df.loc[cur1, cur2]
    return corr


def get_currency_rolling_correlation_waves(dataframe, pair_names, windows):
    correlations = pd.DataFrame(np.zeros((dataframe.shape[0], pair_names.shape[0])))
    correlations.columns = pair_names
    for window in windows:    
        df = dataframe.rolling(window).corr()
        df.index = df.index.swaplevel(0,1)
        corr = pd.DataFrame()
        for currencies in pair_names:
            cur1, cur2 = currencies.split('_')
            corr[currencies] = df.loc[cur1, cur2]
        correlations = correlations + corr
    correlations = correlations / windows.shape[0]
    return correlations
        

    
    
    

    
    
    
    


if __name__ == '__main__':
    
    pass



