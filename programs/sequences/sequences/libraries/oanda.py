import pandas as pd
import oandapyV20
from oandapyV20.contrib.factories import InstrumentsCandlesFactory
import oandapyV20.endpoints.instruments as instruments

# Instrument
currency = 'EUR_AUD'
granularity = 'H1'
daily_alignment = 17

# Time
_from = '2010-04-01T00:00:00Z'
_to = '2019-00-01T00:00:00Z'


def get_candles(instrument, granularity, _from, _to, da=daily_alignment):
    print('Fetching Candles.')
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
        client.request(r)
        coll.append(r.response)
    # collect Returned Data into list.  Cast to floats.
    bidlow = []
    bidhigh = []
    bidclose = []
    asklow = []
    askhigh = []
    askclose = []
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
    df['midhigh'] = pd.to_numeric(midhigh)
    df['midlow'] = pd.to_numeric(midlow)
    df['midclose'] = pd.to_numeric(midclose)
    df['spread'] = df.askclose - df.bidclose
    df['volume'] = pd.to_numeric(volume)
    return df


def get_candles_by_count(instrument, granularity, count, da = daily_alignment):
    client = 'f01b219340f61ffa887944e7673d85a5-6bcb8a840148b5c366e17285c984799e'
    client = oandapyV20.API(access_token=client)
    params = {'count': count,
              'granularity': granularity,
              'price': 'M',
              'alignmentTimezone': 'UTC', #'America/Los_Angeles',
              'dailyAlignment': da}

    r = instruments.InstrumentsCandles(instrument=instrument, params=params)
    coll = client.request(r)
    # Remove last candle if not complete.
    if coll['candles'][-1]['complete'] == False:
        coll['candles'].pop()
    # Assemble Dataframe
    high = []
    low = []
    close = []
    timestamp = []
    for i in range(len(coll['candles'])):
        high.append(float(coll['candles'][i]['mid']['h']))
        low.append(float(coll['candles'][i]['mid']['l']))
        close.append(float(coll['candles'][i]['mid']['c']))
        timestamp.append(coll['candles'][i]['time'])
    # Assemble DataFrame.  Cast Values.
    df = pd.DataFrame(pd.to_datetime(timestamp))
    df.columns = ['timestamp']
    df['high'] = pd.to_numeric(high)
    df['low'] = pd.to_numeric(low)
    df['close'] = pd.to_numeric(close)
    return df
            
