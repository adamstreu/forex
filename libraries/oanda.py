import pandas as pd
import json
import pprint
import oandapyV20
from oandapyV20.contrib.factories import InstrumentsCandlesFactory
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.positions as positions
import oandapyV20.endpoints.transactions as trans
import oandapyV20.endpoints.trades as trades
import oandapyV20.endpoints.accounts as accounts
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.pricing as pricing
import oandapyV20.endpoints.forexlabs as labs



daily_alignment = 0


def get_candles(instrument, granularity, _from, _to, da=daily_alignment):
    #print('Fetching Candles.')
    client = 'f01b219340f61ffa887944e7673d85a5-6bcb8a840148b5c366e17285c984799e'
    client = oandapyV20.API(access_token=client)
    params = {'from': _from,
              'to': _to,
              'granularity': granularity,
              'price': 'BAM',
              'count': 5000,
              'alignmentTimezone': 'UTC',
              'dailyAlignment': da}
    # Request Data
    coll = []
    for r in InstrumentsCandlesFactory(instrument = instrument, 
                                       params = params):
        try:
            client.request(r)
            coll.append(r.response)
        except Exception as e:
            print(e)
    # collect Returned Data into list.  Cast to floats.
    bidlow = []
    bidhigh = []
    bidclose = []
    asklow = []
    askhigh = []
    askclose = []
    midopen = []
    midlow = []
    midhigh = []
    midclose = []
    timestamp = []
    volume = []
    for i in range(len(coll)):
        for j in range(len(coll[i]['candles'])):
            bidhigh.append(float(coll[i]['candles'][j]['bid']['h']))
            bidlow.append(float(coll[i]['candles'][j]['bid']['l']))
            bidclose.append(float(coll[i]['candles'][j]['bid']['c']))
            askhigh.append(float(coll[i]['candles'][j]['ask']['h']))
            asklow.append(float(coll[i]['candles'][j]['ask']['l']))
            askclose.append(float(coll[i]['candles'][j]['ask']['c']))               
            midopen.append(float(coll[i]['candles'][j]['mid']['o']))
            midhigh.append(float(coll[i]['candles'][j]['mid']['h']))
            midlow.append(float(coll[i]['candles'][j]['mid']['l']))
            midclose.append(float(coll[i]['candles'][j]['mid']['c']))               
            timestamp.append(coll[i]['candles'][j]['time'])
            volume.append(float(coll[i]['candles'][j]['volume']))
    # Assemble DataFrame.  Cast Values.
    df = pd.DataFrame(pd.to_datetime(timestamp))
    df.columns = ['timestamp']
    df['bidhigh'] = pd.to_numeric(bidhigh)
    df['bidlow'] = pd.to_numeric(bidlow)
    df['bidclose'] = pd.to_numeric(bidclose)
    df['askhigh'] = pd.to_numeric(askhigh)
    df['asklow'] = pd.to_numeric(asklow)
    df['askclose'] = pd.to_numeric(askclose)
    df['midopen'] = pd.to_numeric(midopen)
    df['midhigh'] = pd.to_numeric(midhigh)
    df['midlow'] = pd.to_numeric(midlow)
    df['midclose'] = pd.to_numeric(midclose)
    df['spread'] = df.askclose - df.bidclose
    df['volume'] = pd.to_numeric(volume)
    
    if not coll[i]['candles'][-1]['complete']:
        df.drop(df.last_valid_index(), inplace=True)

    return df


def get_candles_by_count(instrument, granularity, count, da = daily_alignment):
    client = 'f01b219340f61ffa887944e7673d85a5-6bcb8a840148b5c366e17285c984799e'
    client = oandapyV20.API(access_token=client)
    params = {'count': count,
              'granularity': granularity,
              'price': 'BAM',
              'alignmentTimezone': 'UTC',
              'dailyAlignment': da}

    r = instruments.InstrumentsCandles(instrument=instrument, params=params)
    coll = client.request(r)
    # Remove last candle if not complete.
    if coll['candles'][-1]['complete'] == False:
        coll['candles'].pop()
    # Assemble Dataframe
    askclose = []
    bidclose = []
    high = []
    low = []
    midhigh = []
    midlow = []    
    midclose = []
    volume = []
    close = []
    timestamp = []
    for i in range(len(coll['candles'])):
        high.append(float(coll['candles'][i]['mid']['h']))
        low.append(float(coll['candles'][i]['mid']['l']))
        midhigh.append(float(coll['candles'][i]['mid']['h']))
        midlow.append(float(coll['candles'][i]['mid']['l']))
        close.append(float(coll['candles'][i]['mid']['c']))
        midclose.append(float(coll['candles'][i]['mid']['c']))
        askclose.append(float(coll['candles'][i]['ask']['c']))
        bidclose.append(float(coll['candles'][i]['bid']['c']))
        volume.append(float(coll['candles'][i]['volume']))
        timestamp.append(coll['candles'][i]['time'])
    # Assemble DataFrame.  Cast Values.
    df = pd.DataFrame(pd.to_datetime(timestamp))
    df.columns = ['timestamp']
    df['midhigh'] = pd.to_numeric(midhigh)
    df['midlow'] = pd.to_numeric(midlow)
    df['high'] = pd.to_numeric(high)
    df['low'] = pd.to_numeric(low)
    df['close'] = pd.to_numeric(close)
    df['askclose'] = pd.to_numeric(askclose)
    df['midclose'] = pd.to_numeric(midclose)
    df['bidclose'] = pd.to_numeric(bidclose)
    df['spread'] = df.askclose - df.bidclose
    df['volume'] = pd.to_numeric(volume)
    return df



def get_multiple_candles_midclose(instrument_list, granularity):
    client = 'f01b219340f61ffa887944e7673d85a5-6bcb8a840148b5c366e17285c984799e'
    client = oandapyV20.API(access_token=client)
    params = {'count': 2,
              'granularity': granularity,
              'price': 'M',
              'alignmentTimezone': 'UTC',
              }    
    instrument_dict = {}
    for instrument in instrument_list:
        r = instruments.InstrumentsCandles(instrument=instrument, params=params)
        coll = client.request(r)
        # Remove last candle if not complete.
        if coll['candles'][-1]['complete'] == False:
            coll['candles'].pop()
        instrument_dict[instrument] = float(coll['candles'][-1]['mid']['c'])                
    return instrument_dict


def get_multiple_candles_volume(instrument_list, granularity):
    client = 'f01b219340f61ffa887944e7673d85a5-6bcb8a840148b5c366e17285c984799e'
    client = oandapyV20.API(access_token=client)
    params = {'count': 2,
              'granularity': granularity,
              'price': 'AB',
              'alignmentTimezone': 'UTC',
              }    
    instrument_dict = {}
    for instrument in instrument_list:
        r = instruments.InstrumentsCandles(instrument=instrument, params=params)
        coll = client.request(r)
        # Remove last candle if not complete.
        if coll['candles'][-1]['complete'] == False:
            coll['candles'].pop()
        instrument_dict[instrument] = float(coll['candles'][-1]['volume'])                  
    return instrument_dict



def get_multiple_candles_spread(instrument_list, granularity):
    client = 'f01b219340f61ffa887944e7673d85a5-6bcb8a840148b5c366e17285c984799e'
    client = oandapyV20.API(access_token=client)
    params = {'count': 2,
              'granularity': granularity,
              'price': 'AB',
              'alignmentTimezone': 'UTC',
              }    
    instrument_dict = {}
    for instrument in instrument_list:
        r = instruments.InstrumentsCandles(instrument=instrument, params=params)
        coll = client.request(r)
        # Remove last candle if not complete.
        if coll['candles'][-1]['complete'] == False:
            coll['candles'].pop()
        instrument_dict[instrument] = float(coll['candles'][-1]['ask']['c']) \
                                    - float(coll['candles'][-1]['bid']['c'])                  
    return instrument_dict


def get_spreads():
    client = 'f01b219340f61ffa887944e7673d85a5-6bcb8a840148b5c366e17285c984799e'
    client = oandapyV20.API(access_token=client)
    params = {
              "instrument": "EUR_USD",
              "period": 57600
              }
    r = labs.Spreads(params=params)
    client.request(r)
    print(r.response)
    

def get_time(granularity):
    client = 'f01b219340f61ffa887944e7673d85a5-6bcb8a840148b5c366e17285c984799e'
    client = oandapyV20.API(access_token=client)
    params = {'count': 2,
              'granularity': granularity,
              'price': 'M',
              'alignmentTimezone': 'UTC',
              }    
    r = instruments.InstrumentsCandles(instrument='AUD_JPY', params=params)
    coll = client.request(r)
    # Remove last candle if not complete.
    if coll['candles'][-1]['complete'] == False:
        coll['candles'].pop()
    return pd.to_datetime(coll['candles'][-1]['time'])
    


def get_orderbook(instrument, time):
    client = 'f01b219340f61ffa887944e7673d85a5-6bcb8a840148b5c366e17285c984799e'
    client = oandapyV20.API(access_token=client)
    params = {'time': time, 'Accept-Datetime-Format': 'RFC3339'}
    r = instruments.InstrumentsOrderBook(instrument=instrument,
                                          params=params)
    a = client.request(r)
    #print(a)#(r.response)
    return a          
 

def create_order(instrument, direction):#, quantity, target, stop, account):
    client = 'f01b219340f61ffa887944e7673d85a5-6bcb8a840148b5c366e17285c984799e'
    account_id = '101-001-7518331-001'
    client = oandapyV20.API(access_token=client)
    if direction.lower() == 'long':
        units = 35000
    else:
        units = -35000
    data   = {'order' : {"units": units, #quantity, 
                         "instrument": instrument, 
                         "timeInForce": "FOK", 
                         "type": "MARKET", 
                         "positionFill": "DEFAULT"}}
    '''
    'takeProfitOnFill' : {'price': str(round(target, 5)), 
                              'timeInForce' : 'GTC'},
    'stopLossOnFill':    {'price': str(round(stop, 5)), 
                              'timeInForce' : 'GTC'}}}
    '''
    r = orders.OrderCreate(account_id, data=data)
    client.request(r)    
    return int(r.response['orderCreateTransaction']['id'])


#>>> import json
#>>> from oandapyV20 import API
#>>> import oandapyV20.endpoints.trades as trades
#>>> from oandapyV20.contrib.requests import TradeCloseRequest
#>>>
#>>> accountID = "..."
#>>> client = API(access_token=...)
#>>> ordr = TradeCloseRequest(units=10000)
#>>> print(json.dumps(ordr.data, indent=4))
#{
#   "units": "10000"
#}
#>>> # now we have the order specification, create the order request
#>>> r = trades.TradeClose(accountID, tradeID=1234,
#>>>                       data=ordr.data)
#>>> # perform the request
#>>> rv = client.request(r)
#>>> print(rv)
#>>> ...




def close_position(instrument):
    client = 'f01b219340f61ffa887944e7673d85a5-6bcb8a840148b5c366e17285c984799e'
    account_id = '101-001-7518331-001'
    client = oandapyV20.API(access_token=client)
    data =  {
              "shortUnits": "ALL"
            }
    r = positions.PositionClose(accountID=account_id,
                                 instrument=instrument, 
                                 data = data)
    client.request(r)
    print(r.response)




def price_stream(currency_pairs):    
    # Just a little titillation (can pass shit along based on, you know, stuff)
    def go():
        print('YES')
        print()
        return
    # Main Body
    account='101-001-7518331-001'
    client = 'f01b219340f61ffa887944e7673d85a5-6bcb8a840148b5c366e17285c984799e'
    api = oandapyV20.API(access_token=client)
    params ={'instruments': 'EUR_USD, EUR_JPY'}
    params ={'instruments': currency_pairs}
    r = pricing.PricingStream(accountID=account, params=params)
    rv = api.request(r)
    maxrecs = 10
    for ticks in rv:
        if ticks['type'] == 'PRICE':
            print(json.dumps(ticks['instrument'], indent=4),",")
            print(json.dumps(ticks['asks'][0]['price'], indent=4),",")
            if float(ticks['asks'][0]['price']) < 1000:
                go()
            if maxrecs == 0:
                r.terminate("maxrecs records received")


def get_open_positions(account):
    client = 'f01b219340f61ffa887944e7673d85a5-6bcb8a840148b5c366e17285c984799e'
    client = oandapyV20.API(access_token=client)
    r = positions.OpenPositions(accountID=account)
    client.request(r)
    p = r.response
    instruments = []
    for position in p['positions']:
        instruments.append(position['instrument'])
    return instruments  


def get_transactions_range(_from, account):
    client = 'f01b219340f61ffa887944e7673d85a5-6bcb8a840148b5c366e17285c984799e'
    client = oandapyV20.API(access_token=client)
    params = {"from": _from,
          "to": _from + 500,
          'type': 'ORDER_FILL'}
    r = trans.TransactionIDRange(accountID=account, params=params)
    client.request(r)
    return r.response

    
def get_most_recent_transaction(account):
    client = 'f01b219340f61ffa887944e7673d85a5-6bcb8a840148b5c366e17285c984799e'
    client = oandapyV20.API(access_token=client)
    r = trades.OpenTrades(accountID=account)
    client.request(r)
    _id = int(r.response['lastTransactionID'])
    return _id


def get_accounts():
    client = 'f01b219340f61ffa887944e7673d85a5-6bcb8a840148b5c366e17285c984799e'
    client = oandapyV20.API(access_token=client)
    r = accounts.AccountList()
    client.request(r)
    pprint(r.response['accounts'])
    accounts_collection = []
    for each in r.response['accounts']:
        accounts_collection.append(each['id'])
    return accounts_collection


market = ['AUD_CAD',
        'AUD_CHF',
        'AUD_HKD',
        'AUD_JPY',
        'AUD_NZD',
        'AUD_SGD',
        'AUD_USD',
        'CAD_CHF',
        'CAD_HKD',
        'CAD_JPY',
        'CAD_SGD',
        'CHF_HKD',
        'CHF_JPY',
        'CHF_ZAR',
        'EUR_AUD',
        'EUR_CAD',
        'EUR_CHF',
        'EUR_CZK',
        'EUR_DKK',
        'EUR_GBP',
        'EUR_HKD',
        'EUR_HUF',
        'EUR_JPY',
        'EUR_NOK',
        'EUR_NZD',
        'EUR_PLN',
        'EUR_SEK',
        'EUR_SGD',
        'EUR_TRY',
        'EUR_USD',
        'EUR_ZAR',
        'GBP_AUD',
        'GBP_CAD',
        'GBP_CHF',
        'GBP_HKD',
        'GBP_JPY',
        'GBP_NZD',
        'GBP_PLN',
        'GBP_SGD',
        'GBP_USD',
        'GBP_ZAR',
        'HKD_JPY',
        'NZD_CAD',
        'NZD_CHF',
        'NZD_HKD',
        'NZD_JPY',
        'NZD_SGD',
        'NZD_USD',
        'SGD_CHF',
        'SGD_HKD',
        'SGD_JPY',
        'TRY_JPY',
        'USD_CAD',
        'USD_CHF',
        'USD_CNH',
        'USD_CZK',
        'USD_DKK',
        'USD_HKD',
        'USD_HUF',
        'USD_INR',
        'USD_JPY',
        'USD_MXN',
        'USD_NOK',
        'USD_PLN',
        'USD_SAR',
        'USD_SEK',
        'USD_SGD',
        'USD_THB',
        'USD_TRY',
        'USD_ZAR',
        'ZAR_JPY']


if __name__ == '__main__':
    pass