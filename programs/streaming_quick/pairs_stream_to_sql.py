import numpy as np
import pandas as pd
import oandapyV20
import oandapyV20.endpoints.pricing as pricing
import sqlite3
import os
os.chdir('/northbend')


def db_universe_init(db):
    ''' Create db in env location if not exiting.  Do nothing otherwise '''
    conn = sqlite3.connect(db)
    cur = conn.cursor()  
    cur.execute('create table if not exists universe ('
                'id integer primary key, '
                'timestamp text not null, '
                'aud real not null, '
                'cad real not null, '
                'chf real not null, '
                'eur real not null, '
                'gbp real not null, '
                'hkd real not null, '
                'jpy real not null, '
                'nzd real not null, '
                'usd real not null);')
    conn.commit()
    cur.close()
    conn.close()
    
    
def db_stream_init(db):
    ''' Create db in env location if not exiting.  Do nothing otherwise '''
    conn = sqlite3.connect(db)
    cur = conn.cursor()  
    cur.execute('create table if not exists stream ('
                'id integer primary key, '
                'timestamp text not null, '
                'instrument real not null, '
                'bid real not null, '
                'ask real not null);')
    conn.commit()
    cur.close()
    conn.close()
    

def db_instruments_init(db):
    ''' Create db in env location if not exiting.  Do nothing otherwise '''
    conn = sqlite3.connect(db)
    cur = conn.cursor()  
    cur.execute('create table if not exists instruments ('
                'id integer primary key, '
                'timestamp text not null, '
                'AUD_CAD real not null,'
                'AUD_CHF real not null,'
                'AUD_HKD real not null,'
                'AUD_JPY real not null,'
                'AUD_NZD real not null,'
                'AUD_USD real not null,'
                'CAD_CHF real not null,'
                'CAD_HKD real not null,'
                'CAD_JPY real not null,'
                'CHF_HKD real not null,'
                'CHF_JPY real not null,'
                'EUR_AUD real not null,'
                'EUR_CAD real not null,'
                'EUR_CHF real not null,'
                'EUR_GBP real not null,'
                'EUR_HKD real not null,'
                'EUR_JPY real not null,'
                'EUR_NZD real not null,'
                'EUR_USD real not null,'
                'GBP_AUD real not null,'
                'GBP_CAD real not null,'
                'GBP_CHF real not null,'
                'GBP_HKD real not null,'
                'GBP_JPY real not null,'
                'GBP_NZD real not null,'
                'GBP_USD real not null,'
                'HKD_JPY real not null,'
                'NZD_CAD real not null,'
                'NZD_CHF real not null,'
                'NZD_HKD real not null,'
                'NZD_JPY real not null,'
                'NZD_USD real not null,'
                'USD_CAD real not null,'
                'USD_CHF real not null,'
                'USD_HKD real not null,'
                'USD_JPY real not null);')         
    conn.commit()
    cur.close()
    conn.close()
    

def calculate_currencies(pairs, currencies, instrument_values):
    global universe
    
    for curr in currencies:    
        # Get a list of all instruments that contain the currency
        pair_set = []
        for instrument in pairs:
            if curr.upper() in instrument.split('_'):
                pair_set.append(instrument)
        pair_set = np.array(pair_set)
            
        # Create array to say whether to use the coefficeint or its inverse.
        inverse_array = []
        for pair in pair_set:
            if pair.index(curr.upper()) == 0:
                inverse_array.append(1)
            else:
                inverse_array.append(-1)
        inverse_array = np.array(inverse_array)
        
        # Get all ratios_list to use.  Sort them into alhebetical order.  
        ratios_list = [] 
        for i in range(pair_set.shape[0]):
            if inverse_array[i] == -1:
                ratios_list.append(instrument_values[pair_set[i]])
            else:
                ratios_list.append(1 / instrument_values[pair_set[i]])
        ratios_list = np.array(ratios_list)
        # Calculate prime currency and set to currency_values
        universe[curr] = 1 / (ratios_list.sum(axis=0) + 1)
    

def db_insert(db, table, dictionary):
    # Open Database Connection
    conn = sqlite3.connect(db)  
    cursor = conn.cursor()
    columns = ', '.join(dictionary.keys())
    qmarks = ', '.join('?' * len(dictionary))
    sql = "INSERT INTO %s ( %s ) VALUES ( %s )" % (table, columns, qmarks)
    cursor.execute(sql, tuple(dictionary.values()))
    conn.commit()
    cursor.close()
    conn.close()


def price_stream(pairs, currencies, bd):
    '''
    Request a price stream from oanda for every instrument in pairs.
    With every updated price, update currency prices
    Store values in db. not figured out how yet)
    '''
    global pairs_bid
    global pairs_ask
    global universe
    
    
    # Stream Parameters
    pairs_list = ','.join(pairs)
    account='101-001-7518331-001'
    client = 'f01b219340f61ffa887944e7673d85a5-'
    client += '6bcb8a840148b5c366e17285c984799e'
    api = oandapyV20.API(access_token=client)
    params ={'instruments': pairs_list} 
    r = pricing.PricingStream(accountID=account, params=params)
    
    # Begin Stream
    rv = api.request(r)
    for ticks in rv:
        if ticks['type'] == 'PRICE':
            mid = (float(ticks['asks'][0]['price']) \
                + float(ticks['bids'][0]['price'])) / 2
            pairs_mid[ticks['instrument']] = mid
            pairs_mid['timestamp'] = ticks['time']
            universe['timestamp'] = ticks['time']
            
            # Reduce for streaming insertion
            stream = {'timestamp': ticks['time'],
                      'instrument': ticks['instrument'],
                      'ask': ticks['asks'][0]['price'],
                      'bid': ticks['bids'][0]['price']}
            
            # Calculate Currencies
            calculate_currencies(pairs, currencies, pairs_mid)
            
            # Insert instrument an duniverse pricing into db
            db_insert(db, 'universe', universe)
            db_insert(db, 'instruments', pairs_mid)
            db_insert(db, 'stream', stream)            
            







if __name__ == '__main__':
    
    # Database Params
    db = '/northbend/db/streaming.db'

    # Currency Universe Params
    currencies = ['aud', 'cad', 'chf', 'eur', 'gbp', 'hkd', 'jpy', 'nzd', 'usd']
    pairs = ['AUD_CAD', 'AUD_CHF', 'AUD_HKD', 'AUD_JPY', 'AUD_NZD', 'AUD_USD',
             'CAD_CHF', 'CAD_HKD', 'CAD_JPY', 'CHF_HKD', 'CHF_JPY', 'EUR_AUD',
             'EUR_CAD', 'EUR_CHF', 'EUR_GBP', 'EUR_HKD', 'EUR_JPY', 'EUR_NZD',
             'EUR_USD', 'GBP_AUD', 'GBP_CAD', 'GBP_CHF', 'GBP_HKD', 'GBP_JPY',
             'GBP_NZD', 'GBP_USD', 'HKD_JPY', 'NZD_CAD', 'NZD_CHF', 'NZD_HKD',
             'NZD_JPY', 'NZD_USD', 'USD_CAD', 'USD_CHF', 'USD_HKD', 'USD_JPY']
    
    # Global Params
    pairs_mid = dict.fromkeys(pairs, 1) 
    universe = dict.fromkeys(currencies, 1) 
    pairs_mid['timestamp'] = ''
    universe['timestamp'] = ''
    
    # Initialize Databases
    db_universe_init(db)
    db_stream_init(db)
    db_instruments_init(db)
    
    # Call Data Stream
    print('Streaming Price Data')
    price_stream(pairs, currencies, db)








