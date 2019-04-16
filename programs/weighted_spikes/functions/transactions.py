import yaml
import time
import importlib
import os; os.chdir('/northbend')
from programs.channel_breakout.functions.logs import log_ping
from programs.channel_breakout.functions.logs import log_eval
from programs.channel_breakout.functions.logs import log_transactions
from libraries.oanda import get_transactions_range
from libraries.oanda import get_most_recent_transaction
import programs.channel_breakout.configs.portfolio as ports

    
def watch_transactions():
    # To start, Collect most recent transactions for all accounts in porfolio.
    importlib.reload(ports)  
    accounts = ports.portfolio()
    transactions_record = {}
    for account in accounts['accounts_list']:
        transactions_record[account] = get_most_recent_transaction(account)
    while True:
        log_ping('Requesting recent transactions from Oanda for all active accounts.')
        # Reimport portfolio (change while running)
        importlib.reload(ports)  
        accounts = ports.portfolio()
        # Between sleeps, run for each account
        for account in accounts['accounts_list']:
            # Get direction of account
            if account in accounts['accounts_direction']['long']:
                direction = 'long'
            elif account in accounts['accounts_direction']['short']:
                direction = 'short'
            else:
                direction = 'No direction Found.'
                log_eval('Direction Not found for account: {}'.format(account))
            # Check for new transactions
            try:
                transactions = get_transactions_range(transactions_record[account], account)
            except Exception as e:
                log_ping('Can not get_transactions_range.' + str(e)) 
            try:
                for each in transactions['transactions']:
                    pass # log_transactions(each, account, direction)   # Do not log anything
                transactions_record[account] = int(transactions['lastTransactionID'])
            except Exception as e:
                log_ping('Can not establish connection to Oanda.' + str(e))   
            env = yaml.safe_load(open('/northbend/programs/weighted_spikes/configs/env.yaml','r'))
            
        print('sleeping')
        time.sleep(env['ping_oanda_outcomes_interval'])
           

if __name__ == '__main__':
    watch_transactions()
        
        
    


              


