#!python3
import yaml
import time
import oandapyV20
import oandapyV20.endpoints.transactions as trans
import oandapyV20.endpoints.trades as trades
from logs import log_ping, log_outcomes_long


def get_transactions_range(_from):
    env = yaml.safe_load(open('/forex/programs/channel/configs/env.yaml','r'))
    accountid=env['accountid_long']
    client=oandapyV20.API(access_token=env['client'])
    params = {"from": _from,
          "to": _from + 500,
          'type': 'ORDER_FILL'}
    r = trans.TransactionIDRange(accountID=accountid, params=params)
    client.request(r)
    return r.response

    
def get_most_recent_transaction():
    env = yaml.safe_load(open('/forex/programs/channel/configs/env.yaml','r'))
    accountid=env['accountid_long']
    client=oandapyV20.API(access_token=env['client'])
    r = trades.OpenTrades(accountID=accountid)
    client.request(r)
    _id = int(r.response['lastTransactionID'])
    return _id


def watch_outcomes_long():
    most_recent_transaction = get_most_recent_transaction()
    while True:
        try:
            log_ping('Pinging outcomes long.')
            transactions = get_transactions_range(most_recent_transaction)
            for each in transactions['transactions']:
                log_outcomes_long(each)
            most_recent_transaction = int(transactions['lastTransactionID'])
        except Exception as e:
            log_ping('Can not establish long connection.' + str(e))   
        
        env = yaml.safe_load(open('/forex/programs/channel/configs/env.yaml','r'))
        time.sleep(env['ping_oanda_outcomes_interval'])
       

if __name__ == '__main__':
    watch_outcomes_long()
        
        
    


              


