import pandas as pd
import numpy as np
import os; os.chdir('/northbend')
from libraries.oanda import market
from libraries.oanda import get_candles
from libraries.oanda import get_multiple_candles_midclose
from libraries.oanda import get_multiple_candles_volume




###############################################################################
# Get Currencies Midclose Values From One Timestamp
###############################################################################
def get_universe_singular(currencies, granularity, market=market):
    
    # Prepare Currencies and Instrument collections
    # ---------------------------------------------------------------------
    # Get instrument List
    currencies = list(map(lambda x: x.upper(), currencies))
    instrument_list = []
    for mark in market:
        if mark.split('_')[0] in currencies: 
            if mark.split('_')[1] in currencies:
                instrument_list.append(mark)    
    # Get Prices for all instrument at timestamp
    instrument_values = get_multiple_candles_midclose(instrument_list, granularity)
    
    # Get Currency Universe
    # ---------------------------------------------------------------------
    currency_values = dict.fromkeys(currencies) 
    for curr in currencies:    
        # Get a list of all instruments that contain the currency
        pairs_list = []
        for instrument in instrument_list:
            if curr in instrument.split('_'):
                pairs_list.append(instrument)
        pairs_list = np.array(pairs_list)
            
        # Create array to say whether to use the coefficeint or its inverse.
        inverse_array = []
        for pair in pairs_list:
            if pair.index(curr) == 0:
                inverse_array.append(-1)
            else:
                inverse_array.append(1)
        inverse_array = np.array(inverse_array)
        
        # Get all pairs to use.  Sort them into alhebetical order.  
        pairs = [] #np.empty((pairs_list.shape[0], pair_d[pairs_list[0]].shape[0]))
        for i in range(pairs_list.shape[0]):
            if inverse_array[i] == 1:
                pairs.append(float(instrument_values[pairs_list[i]]))
            else:
                pairs.append(1 / float(instrument_values[pairs_list[i]]))
        pairs = np.array(pairs)
        # Calculate prime currency and set to currency_values
        currency_values[curr] = 1 / (pairs.sum(axis=0) + 1)
    return dict((k.lower(), v) for k,v in currency_values.items()), instrument_values




###############################################################################
# Get Currencies Midclsoe Values Between Two Timestamps
###############################################################################
def backfill_with_singular(currencies, granularity, _from, _to):
    
    def new_matrix(currencies, instrument_list, instrument_values):
        # Get Currency Universe
        # ---------------------------------------------------------------------
        currency_values = dict.fromkeys(currencies) 
        for curr in currencies:    
            # Get a list of all instruments that contain the currency
            pairs_list = []
            for instrument in instrument_list:
                if curr in instrument.split('_'):
                    pairs_list.append(instrument)
            pairs_list = np.array(pairs_list)
                
            # Create array to say whether to use the coefficeint or its inverse.
            inverse_array = []
            for pair in pairs_list:
                if pair.index(curr) == 0:
                    inverse_array.append(-1)
                else:
                    inverse_array.append(1)
            inverse_array = np.array(inverse_array)
            
            # Get all pairs to use.  Sort them into alhebetical order.  
            pairs = [] #np.empty((pairs_list.shape[0], pair_d[pairs_list[0]].shape[0]))
            for i in range(pairs_list.shape[0]):
                if inverse_array[i] == 1:
                    pairs.append(float(instrument_values[pairs_list[i]]))
                else:
                    pairs.append(1 / float(instrument_values[pairs_list[i]]))
            pairs = np.array(pairs)
            # Calculate prime currency and set to currency_values
            currency_values[curr] = 1 / (pairs.sum(axis=0) + 1)
        return dict((k.lower(), v) for k,v in currency_values.items())
    
    
    # Prepare Currencies and Instrument collections
    # ---------------------------------------------------------------------
    # Get instrument List
    currencies = list(map(lambda x: x.upper(), currencies))
    instrument_list = []
    for mark in market:
        if mark.split('_')[0] in currencies: 
            if mark.split('_')[1] in currencies:
                instrument_list.append(mark)       
    
    
    instruments = {}
    for instrument in instrument_list:
         df = get_candles(instrument, granularity, _from, _to)[['timestamp', 'midclose']]
         df.set_index('timestamp', inplace=True, drop=True)
         instruments[instrument] = df
    
    
    # join all to one index to make sure times align.  Frontfill missing
    for instrument in instrument_list[1:]:
        instruments[instrument_list[0]] = instruments[instrument_list[0]]\
                                          .join(instruments[instrument], 
                                                how='outer', lsuffix='.')
    
    pair_values = instruments[instrument_list[0]]
    pair_values.fillna(method='ffill', inplace=True)
    pair_values.fillna(method='bfill', inplace=True)
    pair_values.columns = instrument_list
    
    # Convert completed value set to unlabeled array in order of instrument_list
    close = pd.DataFrame(columns = list(map(lambda x: x.lower(), currencies)))
    for row in pair_values.index:
        time_pairs = pair_values.loc[row].to_dict()
        cu = new_matrix(currencies, instrument_list, time_pairs)
        close.loc[row] = cu
        
    return close, pair_values
    





###############################################################################
# Get Currencies Midclsoe Values Between Two Timestamps
###############################################################################
def backfill_volume_with_singular(currencies, granularity, _from, _to):
    
    def new_matrix(currencies, instrument_list, instrument_values):
        # Get Currency Universe
        # ---------------------------------------------------------------------
        currency_values = dict.fromkeys(currencies) 
        for curr in currencies:    
            # Get a list of all instruments that contain the currency
            pairs_list = []
            for instrument in instrument_list:
                if curr in instrument.split('_'):
                    pairs_list.append(instrument)
            pairs_list = np.array(pairs_list)
                
            # Create array to say whether to use the coefficeint or its inverse.
            inverse_array = []
            for pair in pairs_list:
                if pair.index(curr) == 0:
                    inverse_array.append(-1)
                else:
                    inverse_array.append(1)
            inverse_array = np.array(inverse_array)
            
            # Get all pairs to use.  Sort them into alhebetical order.  
            pairs = [] #np.empty((pairs_list.shape[0], pair_d[pairs_list[0]].shape[0]))
            for i in range(pairs_list.shape[0]):
                if inverse_array[i] == 1:
                    pairs.append(float(instrument_values[pairs_list[i]]))
                else:
                    pairs.append(1 / float(instrument_values[pairs_list[i]]))
            pairs = np.array(pairs)
            # Calculate prime currency and set to currency_values
            currency_values[curr] = 1 / (pairs.sum(axis=0) + 1)
        return dict((k.lower(), v) for k,v in currency_values.items())
    
    
    # Prepare Currencies and Instrument collections
    # ---------------------------------------------------------------------
    # Get instrument List
    currencies = list(map(lambda x: x.upper(), currencies))
    instrument_list = []
    for mark in market:
        if mark.split('_')[0] in currencies: 
            if mark.split('_')[1] in currencies:
                instrument_list.append(mark)       
    
    
    instruments = {}
    for instrument in instrument_list:
         df = get_candles(instrument, granularity, _from, _to)[['timestamp', 'volume']]
         df.set_index('timestamp', inplace=True, drop=True)
         instruments[instrument] = df
    
    
    # join all to one index to make sure times align.  Frontfill missing
    for instrument in instrument_list[1:]:
        instruments[instrument_list[0]] = instruments[instrument_list[0]]\
                                          .join(instruments[instrument], 
                                                how='outer', lsuffix='.')
    
    pair_values = instruments[instrument_list[0]]
    pair_values.fillna(method='ffill', inplace=True)
    pair_values.fillna(method='bfill', inplace=True)
    pair_values.columns = instrument_list
    
    # Convert completed value set to unlabeled array in order of instrument_list
    close = pd.DataFrame(columns = list(map(lambda x: x.lower(), currencies)))
    for row in pair_values.index:
        time_pairs = pair_values.loc[row].to_dict()
        cu = new_matrix(currencies, instrument_list, time_pairs)
        close.loc[row] = cu
        
    return close, pair_values
    








###############################################################################
# Get Currencies Midclose Values From One Timestamp
###############################################################################
def get_volume_universe_singular(currencies, granularity, market=market):
    
    # Prepare Currencies and Instrument collections
    # ---------------------------------------------------------------------
    # Get instrument List
    currencies = list(map(lambda x: x.upper(), currencies))
    instrument_list = []
    for mark in market:
        if mark.split('_')[0] in currencies: 
            if mark.split('_')[1] in currencies:
                instrument_list.append(mark)    
    # Get Prices for all instrument at timestamp
    instrument_values = get_multiple_candles_volume(instrument_list, granularity)
    
    # Get Currency Universe
    # ---------------------------------------------------------------------
    currency_values = dict.fromkeys(currencies) 
    for curr in currencies:    
        # Get a list of all instruments that contain the currency
        pairs_list = []
        for instrument in instrument_list:
            if curr in instrument.split('_'):
                pairs_list.append(instrument)
        pairs_list = np.array(pairs_list)
            
        # Create array to say whether to use the coefficeint or its inverse.
        inverse_array = []
        for pair in pairs_list:
            if pair.index(curr) == 0:
                inverse_array.append(-1)
            else:
                inverse_array.append(1)
        inverse_array = np.array(inverse_array)
        
        # Get all pairs to use.  Sort them into alhebetical order.  
        pairs = [] #np.empty((pairs_list.shape[0], pair_d[pairs_list[0]].shape[0]))
        for i in range(pairs_list.shape[0]):
            if inverse_array[i] == 1:
                pairs.append(float(instrument_values[pairs_list[i]]))
            else:
                pairs.append(1 / float(instrument_values[pairs_list[i]]))
        pairs = np.array(pairs)
        # Calculate prime currency and set to currency_values
        currency_values[curr] = 1 / (pairs.sum(axis=0) + 1)
    return dict((k.lower(), v) for k,v in currency_values.items()), instrument_values
    
    










'''
Call this function 4 times too many
'''
    
def get_currencies(granularity, _from, _to, column='midclose'):
    
    '''
    This function removes the currencies from the ratios.
    The instruments and ratios are hardcoded right now.
        aud, cad, eur, gbp, nzd, usd
    '''
    
    def currency_df(pair_names, pair_values, primaries_only=True):
    
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
         
    ''' Well - what if the first is missing.  Should have used np.arange '''
    # join all to one index to make sure times align.  Backfill missing
    for instrument in instrument_list[1:]:
        instruments[instrument_list[0]] = instruments[instrument_list[0]].join(instruments[instrument], lsuffix='.')
    instruments[instrument_list[0]].fillna(method='backfill', inplace=True)
    
    # Get timestamps array - Create Series to return
    timestamps = pd.Series(instruments[instrument_list[0]].index.values)
    
    ''' This appears to br wrong '''
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
    curdiff = cur.rolling(window=2).apply(lambda x: x[1] - x[0], raw=True)
    
    # Get index of values to drop
    filter_nas = cur[(pd.isna(curdiff) == True).sum(axis=1) > 0].index.values
    
    # Drop values and reindex
    for df in [ratios, cur, curdiff]:
        df.drop(filter_nas, inplace=True)
        df.reset_index(drop=True, inplace=True)
    

    
    return {'currencies': cur,
           'ratios': ratios,
           'currencies_delta': curdiff,
           'currency_set': currency_set,
           'timestamps': timestamps}















if __name__ == '__main__':
    
    pass



