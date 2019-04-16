import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import os; os.chdir('/sequences')
from libraries.oanda import get_candles
from libraries.transformations import horizontal_transform
from libraries.transformations import create_channels
from libraries.outcomes import up_down_simple

'''


RUNNING SUPER LONG ALL NIGHT PROGRAM.
MAKE SURE TO SAVE ALL RESULTS TO FILE.
THEN CAN ANALYZE ON LEISURE.

TOMORROW:
    ADD STATISTICS.
    
    
    

   
    
'''


'''
Proiblem.
I have an %89 on outcome=0  success on NN with M1 with i= 3.
but that is with all of the -1 options . (However, it was not excellent otherwise)

I got the same outcome with 3 outcomes.
i=3 we have 89%.



'''
'''
-------------------------
Do Iteration.
Track results.  
Throw in Tableau.  
Machine Learn Experiment. 
-------------------------

NEXT:
    Check both 3 outcome and two results
    Try hour granularity at comperable window length.
    Check grouping of decent results.
    Add in some new stats and go again!


Statistics to add:
    where is final closing in relation to distribution (and channel to dist, etc.)
    relation of regression line to channel lines to peak lines
    position of peaks - transformed and closing
    if the breakout is 'busted' - a turn around of some sort
    need a 'how well does channel work statistic'
    add channels degree (why would it matter who knows but we can try it - do after 10000 and new granularity)
    
Outcomes to analyze:
    can try for a number of different outcome size ? 
    can just try for when positions are greater than ? 
    try with different window sizes
    
    consider doing a third category for outcomes over _x_ bars.
    
    PROBLEM WITH RESULTS GROUPING (KEEP ANALYSIS)
'''
'''



Columns to use:
    
    closing statistics:
        
    channel statistics:    
        channel position
        channel slope
        closings slope
        channel slope / closings slope
        channel range
        portion outside channel
        
    channel distribution statistics:
        

'''




# Parameters
###############################################################################
# Instrument
instrument  = 'EUR_USD'
granularity = 'M15'
_from       = '2006-01-01T00:00:00Z'
_to         = '2019-01-01T00:00:00Z'
# Window Parameters
window = 500
search_outcomes = window
# Import Candles
candles = get_candles(instrument, granularity, _from, _to)


# Main iteration sequence
###############################################################################
# Call each window.  Transform and collect all results
results = []
outcomes = []
bars = []
for i in range(window, candles.shape[0] - search_outcomes):
    # Prepare Data Slice for linearg regression and outcome
    closings = candles.loc[i - window: i, 'midclose'].values
    closings_outcomes = candles.loc[i:  min(i + search_outcomes, candles.shape[0]), 'midclose'].values
    # Flatten midclose values
    closings_flat = horizontal_transform(closings)
    # Create channel on flat midclose (c1 and c5 are nothing right now)
    channels = create_channels(closings_flat['closing'])
    c2 = (channels['c2'] + closings_flat['linregress']) + closings[0]
    c3 = (channels['c3'] + closings_flat['linregress']) + closings[0]
    c4 = (channels['c4'] + closings_flat['linregress']) + closings[0]
    # Calculate up down outcome on final value
    tmp = []
    tmp_bars = []
    for tar in [.25, .5, .75, 1, 1.25, 1.5, 2, 2.5]:
        distance = (channels['c4'][-1] - channels['c3'][-1]) * tar
        target_up = closings[-1] + distance
        target_down = closings[-1] - distance
        up_or_down = up_down_simple(closings_outcomes, target_up, target_down)
        tmp.append(up_or_down[0])
        tmp_bars.append(up_or_down[1])
    outcomes.append(tmp)
    bars.append(tmp_bars)
    # Where in channel is closing value
    channel_end_c4 = ((channels['c4'] + closings_flat['linregress']) + closings[0])[-1]
    channel_end_c2 = ((channels['c2'] + closings_flat['linregress']) + closings[0])[-1]
    channel_position = (closings[-1] - channel_end_c2) / (channel_end_c4 - channel_end_c2) 
    # Percentage in intra-channel ranges
    in1 = (closings < c2).sum() / closings.shape[0]
    in2 = ((closings >= c2) & (closings < c3)).sum() / closings.shape[0]
    in3 = ((closings >= c3) & (closings < c4)).sum() / closings.shape[0]
    in4 = (closings >= c4).sum() / closings.shape[0]
    '''
    # Position (in channel percentage) of transformed peaks
    trans_hist = np.histogram(closings_flat['closing'], 50) 
    x_dist = ((trans_hist[1][:-1] + trans_hist[1][1:]) / 2)
    y_dist = trans_hist[0]
    trans_dist_peaks = signal.find_peaks_cwt(y_dist, np.arange(1, 10))
    '''
    
    # Collect Statistics for each run
    ###########################################################################
    results.append([i,
                    closings_flat['slope'],
                    channels['slope'],
                    channels['range'], 
                    channel_position,
                    in1, 
                    in2,
                    in3, 
                    in4])
    
    # Track program run.  Print verification graphs.
    ###########################################################################
    # Print to follow alg run
    if i % 1000 == 0:
        print(i, 
              closings_flat['slope'],
              channels['slope'],
              channel_position)
        '''
        # plot verification
        plt.figure(figsize=(14,4))
        x1 = np.arange(closings.shape[0])
        x2 = np.arange(closings_outcomes.shape[0]) + x1[-1]
        plt.plot(x1, closings)
        plt.plot(x1, (channels['c2'] + closings_flat['linregress']) + closings[0])
        plt.plot(x1, (channels['c3'] + closings_flat['linregress']) + closings[0])
        plt.plot(x1, (channels['c4'] + closings_flat['linregress']) + closings[0])
        plt.plot(x2, closings_outcomes)
        plt.plot(x2, np.ones(closings_outcomes.shape[0]) *  target_up)
        plt.plot(x2, np.ones(closings_outcomes.shape[0]) *  target_down)
        plt.plot(x1, closings_flat['linregress'] + closings[0])
        plt.show()
        '''
        
        
# Results Configuration
###############################################################################
# Assemble columns for results collected
columns = ['location',
           'closings_slope',
           'channel_slope',
           'channel_range',
           'channel_position',
           'in1', 
           'in2',
           'in3',
           'in4']
# Put together dataframe of results
results = pd.DataFrame(results, columns=columns)
results = results.set_index('location', drop=True)
results.to_csv('/Users/user/Desktop/channel.csv')


# Outcomes Stuff
##############################################################################
outcomes = np.array(outcomes)
'''
# If we want to convert to 2 outcomes and drop all outcomes not found.
outs = outcomes.copy().astype(float)
outs[outs == -1] = np.nan
# Then use outs below instead of outcomes
'''

# ML Stuff
###############################################################################

for i in range(outcomes.shape[1]):
    print()
    print(i)
    print('-------------------------------------')
    # Copy results and reindex
    df = results.copy()
    df.reset_index(inplace=True, drop=True)
    # Set Outcomes.  Remove poor Data
    outcomes_column = i
    target = outcomes[:, outcomes_column]
    df['target'] = target
    df.dropna(inplace = True)
    target = df.pop('target')
    # df.drop(['breakout_c4', 'breakout_c2', 'breakout' ], axis=1, inplace=True)
    # Add any additional columns
    df['both_slopes'] = df.channel_slope / df.closings_slope
    # Arange results for modeling 
    row = int(df.shape[0] * .8)
    train_x = df.iloc[:row, :]
    train_y = target[:row].astype(int)
    test_x  = df.iloc[row:, :]
    test_y  = target[row:].astype(int)
    # Standardize values
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.fit_transform(test_x)
    # Fit model on Logistic Regression
    logreg = LogisticRegression(solver='saga',multi_class='multinomial')
    logreg.fit(train_x, train_y)
    predictions = logreg.predict(test_x)
    score = logreg.score(test_x, test_y)
    cr = classification_report(test_y, predictions)
    print('Logistic Regression Score: {}'.format(score))
    print(cr)
    print()
    # Fit model on Niueral Network
    nn = MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
           beta_1=0.9, beta_2=0.999, early_stopping=False,
           epsilon=1e-08, hidden_layer_sizes=(5, 2), learning_rate='constant',
           learning_rate_init=0.001, max_iter=200, momentum=0.9,
           nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
           solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
           warm_start=False)
    nn.fit(train_x, train_y)
    predictions = nn.predict(test_x)
    score = nn.score(test_x, test_y)
    cr = classification_report(test_y, predictions)
    print('NN Score: {}'.format(score))
    print(cr)
    '''
    # Fit model on Support Vecotr Machine
    svc_model = SVC()
    svc_model.fit(train_x, train_y)
    predictions = svc_model.predict(test_x)
    score = svc_model.score(test_x, test_y)
    cr = classification_report(test_y, predictions)
    print('SVM Score: {}'.format(score))
    print(cr)
    '''
    
# Plot some stuff
###############################################################################
plt.figure(figsize=(14,8))
plt.plot(np.arange(results.shape[0]),(results.closings_slope - results.closings_slope.mean()) / results.closings_slope.std(), label='closings slope')
plt.plot(np.arange(results.shape[0]),(results.channel_slope - results.closings_slope.mean()) / results.closings_slope.std(), label='channel slope')
plt.plot((candles.midclose.values[window:-search_outcomes] - candles.midclose.values[window:-search_outcomes].mean()) / candles.midclose.values[window:-search_outcomes].std() * 2, label='closing values')
plt.plot(np.zeros(results.shape[0]), color='black')
plt.legend()


'''
# Save model
###############################################################################
pkl_filename = "/channel/model/pickle_nn_model_3outcome_m1_tar2.5_89_perc_on_0.pkl"  
with open(pkl_filename, 'wb') as file:  
    pickle.dump(logreg, file)
'''










'''


# Parameters
###############################################################################
# Instrument
instrument  = 'EUR_USD'
granularity = 'M1'
_from       = '2017-06-01T00:00:00Z'
_to         = '2018-01-01T00:00:00Z'
# Window Parameters
window = 5000
search_outcomes = window
# Import Candles
candles = get_candles(instrument, granularity, _from, _to)


Really Terrbile.
Al at 50%.
Now we aree trying to go back to dropping.
It might be the granularity though.
We shall see.

Very Good results without the three thing


0
-------------------------------------
Logistic Regression Score: 0.5146685615673535
             precision    recall  f1-score   support

          0       0.44      0.26      0.32     17806
          1       0.54      0.73      0.62     21598

avg / total       0.50      0.51      0.49     39404


/anaconda3/lib/python3.6/site-packages/sklearn/metrics/classification.py:1135: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
NN Score: 0.5481169424423916
             precision    recall  f1-score   support

          0       0.00      0.00      0.00     17806
          1       0.55      1.00      0.71     21598

avg / total       0.30      0.55      0.39     39404


1
-------------------------------------
Logistic Regression Score: 0.49209370796213103
             precision    recall  f1-score   support

          0       0.40      0.45      0.43     16526
          1       0.57      0.52      0.54     22873

avg / total       0.50      0.49      0.49     39399


NN Score: 0.4288941343688926
             precision    recall  f1-score   support

          0       0.39      0.67      0.50     16526
          1       0.52      0.25      0.34     22873

avg / total       0.47      0.43      0.41     39399


2
-------------------------------------
Logistic Regression Score: 0.4886597407940753
             precision    recall  f1-score   support

          0       0.47      0.33      0.39     19091
          1       0.50      0.64      0.56     19797

avg / total       0.48      0.49      0.48     38888


NN Score: 0.5003085784817939
             precision    recall  f1-score   support

          0       0.26      0.01      0.02     19091
          1       0.50      0.97      0.66     19797

avg / total       0.39      0.50      0.35     38888


3
-------------------------------------
Logistic Regression Score: 0.5004712782112374
             precision    recall  f1-score   support

          0       0.50      0.17      0.26     19116
          1       0.50      0.83      0.62     19078

avg / total       0.50      0.50      0.44     38194


NN Score: 0.5020422055820286
             precision    recall  f1-score   support

          0       0.89      0.01      0.01     19116
          1       0.50      1.00      0.67     19078

avg / total       0.69      0.50      0.34     38194


4
-------------------------------------
Logistic Regression Score: 0.55336835264811
             precision    recall  f1-score   support

          0       0.74      0.15      0.25     18298
          1       0.53      0.95      0.68     18634

avg / total       0.63      0.55      0.47     36932


NN Score: 0.5087999566771364
             precision    recall  f1-score   support

          0       0.58      0.03      0.06     18298
          1       0.51      0.98      0.67     18634

avg / total       0.54      0.51      0.37     36932


5
-------------------------------------
Logistic Regression Score: 0.5424862649093342
             precision    recall  f1-score   support

          0       0.66      0.13      0.21     17126
          1       0.53      0.94      0.68     18003

avg / total       0.59      0.54      0.45     35129


NN Score: 0.5698425802043895
             precision    recall  f1-score   support

          0       0.67      0.23      0.35     17126
          1       0.55      0.89      0.68     18003

avg / total       0.61      0.57      0.52     35129


6
-------------------------------------
Logistic Regression Score: 0.5465744687993018
             precision    recall  f1-score   support

          0       0.51      0.22      0.31     13597
          1       0.56      0.82      0.66     16194

avg / total       0.53      0.55      0.50     29791


NN Score: 0.5901111073814239
             precision    recall  f1-score   support

          0       0.63      0.25      0.35     13597
          1       0.58      0.88      0.70     16194

avg / total       0.60      0.59      0.54     29791


7
-------------------------------------
Logistic Regression Score: 0.6025582646851844
             precision    recall  f1-score   support

          0       0.52      0.28      0.36      9823
          1       0.63      0.82      0.71     14334

avg / total       0.58      0.60      0.57     24157


NN Score: 0.6280167239309516
             precision    recall  f1-score   support

          0       0.62      0.23      0.33      9823
          1       0.63      0.90      0.74     14334

avg / total       0.62      0.63      0.58     24157

0
-------------------------------------
Logistic Regression Score: 0.5146685615673535
             precision    recall  f1-score   support

          0       0.44      0.26      0.32     17806
          1       0.54      0.73      0.62     21598

avg / total       0.50      0.51      0.49     39404


/anaconda3/lib/python3.6/site-packages/sklearn/metrics/classification.py:1135: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
NN Score: 0.5481169424423916
             precision    recall  f1-score   support

          0       0.00      0.00      0.00     17806
          1       0.55      1.00      0.71     21598

avg / total       0.30      0.55      0.39     39404


1
-------------------------------------
/anaconda3/lib/python3.6/site-packages/sklearn/linear_model/sag.py:326: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
  "the coef_ did not converge", ConvergenceWarning)
Logistic Regression Score: 0.4922850472033296
             precision    recall  f1-score   support

          0       0.41      0.45      0.43     16526
          1       0.57      0.52      0.54     22878

avg / total       0.50      0.49      0.50     39404


NN Score: 0.4243985382194701
             precision    recall  f1-score   support

          0       0.37      0.54      0.44     16526
          1       0.51      0.34      0.41     22878

avg / total       0.45      0.42      0.42     39404


2
-------------------------------------
/anaconda3/lib/python3.6/site-packages/sklearn/metrics/classification.py:1137: UndefinedMetricWarning: Recall and F-score are ill-defined and being set to 0.0 in labels with no true samples.
  'recall', 'true', average, warn_for)
Logistic Regression Score: 0.49360471018170743
             precision    recall  f1-score   support

         -1       0.00      0.00      0.00         0
          0       0.47      0.33      0.39     19092
          1       0.51      0.65      0.57     20312

avg / total       0.49      0.49      0.48     39404


NN Score: 0.5047203329611207
             precision    recall  f1-score   support

         -1       0.00      0.00      0.00         0
          0       0.48      0.35      0.41     19092
          1       0.52      0.65      0.58     20312

avg / total       0.50      0.50      0.49     39404


3
-------------------------------------
Logistic Regression Score: 0.5024870571515582
             precision    recall  f1-score   support

         -1       0.09      0.03      0.05       492
          0       0.48      0.18      0.26     19116
          1       0.51      0.83      0.63     19796

avg / total       0.49      0.50      0.44     39404


NN Score: 0.5091615064460461
             precision    recall  f1-score   support

         -1       0.00      0.00      0.00       492
          0       0.50      0.23      0.31     19116
          1       0.52      0.79      0.62     19796

avg / total       0.50      0.51      0.47     39404


4
-------------------------------------
Logistic Regression Score: 0.5395898893513349
             precision    recall  f1-score   support

         -1       0.46      0.62      0.53      1435
          0       0.58      0.15      0.23     18315
          1       0.54      0.90      0.67     19654

avg / total       0.56      0.54      0.46     39404


NN Score: 0.520530910567455
             precision    recall  f1-score   support

         -1       0.25      0.31      0.27      1435
          0       0.52      0.23      0.32     18315
          1       0.54      0.81      0.65     19654

avg / total       0.52      0.52      0.48     39404


5
-------------------------------------
Logistic Regression Score: 0.5143132676885596
             precision    recall  f1-score   support

         -1       0.45      0.46      0.45      3514
          0       0.51      0.13      0.20     17140
          1       0.52      0.88      0.66     18750

avg / total       0.51      0.51      0.44     39404


NN Score: 0.4557659120901431
             precision    recall  f1-score   support

         -1       0.28      0.17      0.21      3514
          0       0.37      0.26      0.31     17140
          1       0.51      0.69      0.58     18750

avg / total       0.43      0.46      0.43     39404


6
-------------------------------------
Logistic Regression Score: 0.4902040401989646
             precision    recall  f1-score   support

         -1       0.51      0.61      0.55      8829
          0       0.45      0.19      0.27     13814
          1       0.49      0.67      0.57     16761

avg / total       0.48      0.49      0.46     39404


NN Score: 0.44939600040605016
             precision    recall  f1-score   support

         -1       0.48      0.49      0.48      8829
          0       0.32      0.12      0.17     13814
          1       0.47      0.70      0.56     16761

avg / total       0.42      0.45      0.41     39404


7
-------------------------------------
Logistic Regression Score: 0.5799665008628566
             precision    recall  f1-score   support

         -1       0.67      0.76      0.71     15299
          0       0.50      0.23      0.32      9823
          1       0.51      0.62      0.56     14282

avg / total       0.57      0.58      0.56     39404


NN Score: 0.5191858694548777
             precision    recall  f1-score   support

         -1       0.62      0.72      0.67     15299
          0       0.89      0.01      0.02      9823
          1       0.43      0.65      0.52     14282

avg / total       0.62      0.52      0.45     39404



'''






