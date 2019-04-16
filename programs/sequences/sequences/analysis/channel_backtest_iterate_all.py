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
_from       = '2010-01-01T00:00:00Z'
_to         = '2018-01-01T00:00:00Z'
# Window Parameters
window = 10000
search_outcomes = window
# Import Candles
candles = get_candles(instrument, granularity, _from, _to)


# Main iteration sequence
###############################################################################
# Call each window.  Transform and collect all results
results = []
outcomes = []
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
    for tar in [.25, .5, .75, 1, 1.25, 1.5, 2, 2.5]:
        distance = (channels['c4'][-1] - channels['c3'][-1]) * tar
        target_up = closings[-1] + distance
        target_down = closings[-1] - distance
        up_or_down = up_down_simple(closings_outcomes, target_up, target_down)
        tmp.append(up_or_down[0])
    outcomes.append(tmp)
    # Where in channel is closing value
    channel_end_c4 = ((channels['c4'] + closings_flat['linregress']) + closings[0])[-1]
    channel_end_c2 = ((channels['c2'] + closings_flat['linregress']) + closings[0])[-1]
    channel_position = (closings[-1] - channel_end_c2) / (channel_end_c4 - channel_end_c2) 
    # Percentage in intra-channel ranges
    in1 = (closings < c2).sum() / closings.shape[0]
    in2 = ((closings >= c2) & (closings < c3)).sum() / closings.shape[0]
    in3 = ((closings >= c3) & (closings < c4)).sum() / closings.shape[0]
    in4 = (closings >= c4).sum() / closings.shape[0]
    # Position (in channel percentage) of transformed peaks
    trans_hist = np.histogram(closings_flat['closing'], 50) 
    x_dist = ((trans_hist[1][:-1] + trans_hist[1][1:]) / 2)
    y_dist = trans_hist[0]
    trans_dist_peaks = signal.find_peaks_cwt(y_dist, np.arange(1, 10))
   

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


# Filter Results
###############################################################################
# for instance - remove the folowing hits till first is resolved
#               - easier - if next number less than current + _x_, don't use
'''
a = results.channel_slope > 0
b = results.channel_slope < 0
b4 = results.

# Get index values where desired breakout and slope occured
ind = results.loc[results.breakout_c4 & (results.closings_slope > 0)].index.values
keep = ind[0]
for i in range(1, ind.shape[0]):
    if ind[i] > ind[i-1] + 25:
        keep = np.append(keep, ind[i])
# filter Results on new index and desired slope
results.loc[keep][results.channel_slope > 0]
'''


# Outcomes Stuff
##############################################################################
outcomes = np.array(outcomes)


# ML Stuff
###############################################################################
for i in range(outcomes.shape[1]):
    print()
    print(i)
    print('-------------------------------------')
    # Machine Learning Portion
    ###############################################################################
    # Copy results and reindex
    df = results.copy()
    df.reset_index(inplace=True, drop=True)
    # Set Outcomes.  Remove poor Data
    outcomes_column = i
    target = outcomes[:, outcomes_column]
    df['target'] = target
    df.dropna(inplace = True)
    target = df.pop('target')
    df.drop(['bars'], axis=1, inplace=True)
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
    logreg = LogisticRegression(C=1, solver='newton-cg')
    logreg.fit(train_x, train_y)
    predictions = logreg.predict(test_x)
    score = logreg.score(test_x, test_y)
    cr = classification_report(test_y, predictions)
    print('Logistic Regression Score: {}'.format(score))
    print(cr)
    print()
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
    

plt.figure(figsize=(14,8))
plt.plot(np.arange(results.shape[0]),(results.closings_slope - results.closings_slope.mean()) / results.closings_slope.std(), label='closings slope')
plt.plot(np.arange(results.shape[0]),(results.channel_slope - results.closings_slope.mean()) / results.closings_slope.std(), label='channel slope')
plt.plot((candles.midclose.values[window:-search_outcomes] - candles.midclose.values[window:-search_outcomes].mean()) / candles.midclose.values[window:-search_outcomes].std() * 2, label='closing values')
plt.plot(np.zeros(results.shape[0]), color='black')
plt.legend()



'''


    # Is Final closing Value outside of c4 or c2 channel
    channel_end_c4 = ((channels['c4'] + closings_flat['linregress']) + closings[0])[-1]
    channel_end_c2 = ((channels['c2'] + closings_flat['linregress']) + closings[0])[-1]
    breakout_c4 = False
    breakout_c2 = False
    breakout = False
    if closings[-1] > channel_end_c4:
        breakout_c4 = True
    if closings[-1] < channel_end_c2:
        breakout_c2 = True
    if breakout_c2 or breakout_c4:
        breakout = True
        
        
# Analysis
###############################################################################
# What do the slope and a breakout in the oall directions say?
slope_up_breakout_c4  = results.loc[(results.breakout_c4 & (results.closings_slope > 0)), 'outcome'].shape[0]
slope_up_breakout_c2  = results.loc[(results.breakout_c2 & (results.closings_slope > 0)), 'outcome'].shape[0]
slope_down_breakout_c4 = results.loc[(results.breakout_c4 & (results.closings_slope < 0)), 'outcome'].shape[0]
slope_down_breakout_c2 = results.loc[(results.breakout_c2 & (results.closings_slope < 0)), 'outcome'].shape[0]
slope_up_breakout_c4_outcome_up     = results.loc[(results.breakout_c4 & (results.closings_slope > 0)), 'outcome'].value_counts()['up']
slope_up_breakout_c4_outcome_down   = results.loc[(results.breakout_c4 & (results.closings_slope > 0)), 'outcome'].value_counts()['down']
slope_up_breakout_c2_outcome_up     = results.loc[(results.breakout_c2 & (results.closings_slope > 0)), 'outcome'].value_counts()['up']
slope_up_breakout_c2_outcome_down   = results.loc[(results.breakout_c2 & (results.closings_slope > 0)), 'outcome'].value_counts()['down']
slope_down_breakout_c4_outcome_up   = results.loc[(results.breakout_c4 & (results.closings_slope < 0)), 'outcome'].value_counts()['up']
slope_down_breakout_c4_outcome_down = results.loc[(results.breakout_c4 & (results.closings_slope < 0)), 'outcome'].value_counts()['down']
slope_down_breakout_c2_outcome_up   = results.loc[(results.breakout_c2 & (results.closings_slope < 0)), 'outcome'].value_counts()['up']
slope_down_breakout_c2_outcome_down = results.loc[(results.breakout_c2 & (results.closings_slope < 0)), 'outcome'].value_counts()['down']
a = slope_up_breakout_c4_outcome_up      / slope_up_breakout_c4
b = slope_up_breakout_c4_outcome_down    / slope_up_breakout_c4
c = slope_up_breakout_c2_outcome_up      / slope_up_breakout_c2
d = slope_up_breakout_c2_outcome_down    / slope_up_breakout_c2
e = slope_down_breakout_c4_outcome_up    / slope_down_breakout_c4
f = slope_down_breakout_c4_outcome_down  / slope_down_breakout_c4
g = slope_down_breakout_c2_outcome_up    / slope_down_breakout_c2
h = slope_down_breakout_c2_outcome_down  / slope_down_breakout_c2
print('slope_up_breakout_c4:    {}, or {:.2f}%'.format(slope_up_breakout_c4,  slope_up_breakout_c4   / candles.shape[0]))
print('slope_up_breakout_c2:    {}, or {:.2f}%'.format(slope_up_breakout_c2,  slope_up_breakout_c2   / candles.shape[0]))
print('slope_down_breakout_c4:   {}, or {:.2f}%'.format(slope_down_breakout_c4, slope_down_breakout_c4  / candles.shape[0]))
print('slope_down_breakout_c2:  {}, or {:.2f}%'.format(slope_down_breakout_c2, slope_down_breakout_c2  / candles.shape[0]))
print()
print('slope_up_breakout_c4     up/down: {:.2f} / {:.2f}'.format(a, b))
print('slope_up_breakout_c2     up/down: {:.2f} / {:.2f}'.format(c, d))
print('slope_down_breakout_c4   up/down: {:.2f} / {:.2f}'.format(e, f))     # Check percentages by printing counts
print('slope_down_breakout_c2   up/down: {:.2f} / {:.2f}'.format(g, h))
'''




    
"""


# Also, k mean was interesting on  trns_closing
clusters = 2
data = np.c_[closings_x.ravel(), trans_closings.ravel()]
k = scipy.cluster.vq.kmeans(data, clusters)

# plot closing values with linear regression line
plt.figure(figsize=(14, 3))
plt.plot(np.arange(closings.shape[0]), closings.values,  color='black', marker='.')
plt.plot(np.arange(closings.shape[0]), line, color='blue')
plt.plot(np.arange(closings_long.shape[0]), candles_after.midclose.values - candles_after.midclose.values[0])
# plot horizontal transformation with linear regression, top and bottom channels
plt.figure(figsize=(14, 3))
plt.plot(np.arange(closings.shape[0]), trans_closings, color='black', marker='.')
plt.plot(np.arange(closings.shape[0]), line-line , color='blue')
plt.bar(top, trans_closings[top])
plt.bar(bottom, trans_closings[bottom])
plt.plot(closings_x, top_line)
plt.plot(closings_x, bottom_line)
plt.plot(closings_x, bottom_raised)
plt.plot(closings_x, top_lowered)
plt.plot(closings_x, channel_top)
plt.plot(closings_x, channel_bottom)
plt.plot(closings_x, channel_top_outer)
plt.plot(closings_x, channel_bottom_outer)
plt.plot(closings_long, trans_closings_long)
# Plot k means
for i in range(clusters):
    plt.plot(k[0][i][0], k[0][i][1], 'o', color='black', markersize=12)
# plot histograms with kernels for both original closing values and transformed
plt.figure(figsize=(14, 3))
sns.distplot(closings.values, color='orange',  kde_kws={"label": "Closing"}, bins=50)
plt.figure(figsize=(14, 3))
sns.distplot(trans_closings, kde_kws={"label": "Transformed"}, bins=50)
plt.figure(figsize = (14, 3))
plt.plot(x_dist, y_dist)
plt.plot(x_dist[trans_dist_peaks], y_dist[trans_dist_peaks], 'o')
plt.show()

# Print Summary of regression
print()
print('Pips Range: \t\t{}'.format(pips))
print('Channels: \t{}\t\t{}'.format(channel_outer_width, channel_width))
print('Slope:\t\t{}'.format(slope))
print('Channel Slope:\t\t{}'.format(channel_slope))
print('Slope Ratio:\t\t{}'.format(channel_slope / slope))
print('smallest variance: \t{}'.format(min_distance_perc))
print('Mean squared error:\t{:.2f}'.format(error))
print('Variance score:\t\t{:.2f}'.format(variance_score))
print('closing:  mean, std:\t{}\t{}'.format((closings.values).mean(), (closings.values).std()))
print('closing - line:  mean, std:\t{}\t{}'.format((closings.values-line).mean(), (closings.values-line).std()))
print('Percent values greater than top_channel: {}'.format(above_channel_perc))
print('Percent values below than bottom_channel: {}'.format(below_channel_perc))
print('Channel break: {}'.format(channel_break_top or channel_break_bottom))
print('Slope break: {}'.format(channel_placement))
print('Outer channel break: {}'.format(outer_channel_break))
print('Within Channel Outside % {}'.format(perc_outside_channel > (below_channel_perc + above_channel_perc)))
print('Left of zero to right: {}'.format(hist[0][(hist[1] < 0)[:-1]].sum() / hist[0].sum()))
'''
"""



'''
eurusd
2010 - 2018
m15

    window = 5000:   
        Best Result = tar: 1.5, model: logreg, outcome: 1
    window = 2500:
        Results not as good
    window = 10000
        Good results = tar = 2.5, model: logreg, outcome = 1 at 85% correct
    

    

I mean, that is a good result.
I should just deploy it while exploring more.
Right?.......
And then while those are going.  More feature extractions.


5000
-------------------------------------------------------------------------------
0
-------------------------------------
Logistic Regression Score: 0.5185445825839915
             precision    recall  f1-score   support

          0       0.49      0.55      0.52     18079
          1       0.55      0.49      0.52     20288

avg / total       0.52      0.52      0.52     38367


SVM Score: 0.524513253577293
             precision    recall  f1-score   support

          0       0.50      0.59      0.54     18079
          1       0.56      0.46      0.51     20288

avg / total       0.53      0.52      0.52     38367

NN Score: 0.47566919488101755
             precision    recall  f1-score   support

          0       0.47      0.95      0.63     18079
          1       0.55      0.05      0.09     20288

avg / total       0.51      0.48      0.35     38367


1
-------------------------------------
Logistic Regression Score: 0.4690749863163656
             precision    recall  f1-score   support

          0       0.43      0.42      0.42     17787
          1       0.50      0.52      0.51     20580

avg / total       0.47      0.47      0.47     38367


SVM Score: 0.509422159668465
             precision    recall  f1-score   support

          0       0.47      0.55      0.51     17787
          1       0.55      0.48      0.51     20580

avg / total       0.51      0.51      0.51     38367

NN Score: 0.4688143456616363
             precision    recall  f1-score   support

          0       0.46      0.81      0.59     17787
          1       0.51      0.18      0.26     20580

avg / total       0.49      0.47      0.41     38367


2
-------------------------------------
Logistic Regression Score: 0.5252456794307014
             precision    recall  f1-score   support

          0       0.42      0.44      0.43     15583
          1       0.60      0.58      0.59     22780

avg / total       0.53      0.53      0.53     38363


SVM Score: 0.4798894768396632
             precision    recall  f1-score   support

          0       0.37      0.41      0.39     15583
          1       0.57      0.53      0.55     22780

avg / total       0.49      0.48      0.48     38363

NN Score: 0.5059823267210594
             precision    recall  f1-score   support

          0       0.43      0.68      0.53     15583
          1       0.64      0.39      0.48     22780

avg / total       0.56      0.51      0.50     38363


3
-------------------------------------
Logistic Regression Score: 0.566684182869154
             precision    recall  f1-score   support

          0       0.43      0.52      0.47     13983
          1       0.68      0.59      0.63     24077

avg / total       0.59      0.57      0.57     38060


SVM Score: 0.4931949553336837
             precision    recall  f1-score   support

          0       0.34      0.39      0.36     13983
          1       0.61      0.55      0.58     24077

avg / total       0.51      0.49      0.50     38060

NN Score: 0.599474513925381
             precision    recall  f1-score   support

          0       0.38      0.14      0.21     13983
          1       0.63      0.87      0.73     24077

avg / total       0.54      0.60      0.54     38060


4
-------------------------------------
Logistic Regression Score: 0.6195917041208643
             precision    recall  f1-score   support

          0       0.49      0.61      0.54     13546
          1       0.73      0.62      0.68     23388

avg / total       0.64      0.62      0.63     36934


SVM Score: 0.5012996155304056
             precision    recall  f1-score   support

          0       0.34      0.39      0.36     13546
          1       0.62      0.57      0.59     23388

avg / total       0.52      0.50      0.51     36934

NN Score: 0.5670114257865382
             precision    recall  f1-score   support

          0       0.32      0.15      0.21     13546
          1       0.62      0.81      0.70     23388

avg / total       0.51      0.57      0.52     36934


5
-------------------------------------
Logistic Regression Score: 0.6388019928165913
             precision    recall  f1-score   support

          0       0.53      0.62      0.57     13419
          1       0.73      0.65      0.69     21105

avg / total       0.65      0.64      0.64     34524


SVM Score: 0.5057351407716372
             precision    recall  f1-score   support

          0       0.38      0.42      0.40     13419
          1       0.60      0.56      0.58     21105

avg / total       0.52      0.51      0.51     34524

NN Score: 0.57429614181439
             precision    recall  f1-score   support

          0       0.42      0.25      0.31     13419
          1       0.62      0.78      0.69     21105

avg / total       0.54      0.57      0.54     34524


6
-------------------------------------
Logistic Regression Score: 0.5829419035846725
             precision    recall  f1-score   support

          0       0.48      0.78      0.59     11173
          1       0.76      0.46      0.57     17142

avg / total       0.65      0.58      0.58     28315


SVM Score: 0.5560303725940314
             precision    recall  f1-score   support

          0       0.45      0.51      0.48     11173
          1       0.65      0.58      0.61     17142

avg / total       0.57      0.56      0.56     28315

NN Score: 0.6032491612219671
             precision    recall  f1-score   support

          0       0.49      0.21      0.30     11173
          1       0.63      0.86      0.72     17142

avg / total       0.57      0.60      0.56     28315











10000
-------------------------------------------------------------------------------


0
-------------------------------------
Logistic Regression Score: 0.5133500151236011
             precision    recall  f1-score   support

          0       0.49      0.37      0.42     17441
          1       0.53      0.65      0.58     18926

avg / total       0.51      0.51      0.50     36367


NN Score: 0.5260813374762835
             precision    recall  f1-score   support

          0       0.53      0.10      0.17     17441
          1       0.53      0.92      0.67     18926

avg / total       0.53      0.53      0.43     36367


1
-------------------------------------
Logistic Regression Score: 0.47149888635301235
             precision    recall  f1-score   support

          0       0.39      0.42      0.40     15450
          1       0.54      0.51      0.53     20917

avg / total       0.48      0.47      0.47     36367


NN Score: 0.5822586410757005
             precision    recall  f1-score   support

          0       0.53      0.15      0.24     15450
          1       0.59      0.90      0.71     20917

avg / total       0.56      0.58      0.51     36367


2
-------------------------------------
Logistic Regression Score: 0.527719722802772
             precision    recall  f1-score   support

          0       0.36      0.52      0.42     12232
          1       0.69      0.53      0.60     24132

avg / total       0.58      0.53      0.54     36364


NN Score: 0.5934990650093499
             precision    recall  f1-score   support

          0       0.21      0.08      0.11     12232
          1       0.65      0.86      0.74     24132

avg / total       0.50      0.59      0.53     36364


3
-------------------------------------
Logistic Regression Score: 0.5536716913290002
             precision    recall  f1-score   support

          0       0.32      0.46      0.38     10564
          1       0.73      0.59      0.65     25591

avg / total       0.61      0.55      0.57     36155


NN Score: 0.5781772922140783
             precision    recall  f1-score   support

          0       0.25      0.23      0.24     10564
          1       0.69      0.72      0.71     25591

avg / total       0.56      0.58      0.57     36155


4
-------------------------------------
Logistic Regression Score: 0.5523752733055058
             precision    recall  f1-score   support

          0       0.33      0.48      0.39     10572
          1       0.72      0.59      0.65     24645

avg / total       0.60      0.55      0.57     35217


NN Score: 0.453814918931198
             precision    recall  f1-score   support

          0       0.20      0.27      0.23     10572
          1       0.63      0.53      0.58     24645

avg / total       0.50      0.45      0.47     35217


5
-------------------------------------
Logistic Regression Score: 0.5367695338148392
             precision    recall  f1-score   support

          0       0.40      0.53      0.46     12329
          1       0.66      0.54      0.60     21177

avg / total       0.57      0.54      0.54     33506


NN Score: 0.5103563540858354
             precision    recall  f1-score   support

          0       0.23      0.14      0.17     12329
          1       0.59      0.73      0.65     21177

avg / total       0.46      0.51      0.48     33506


6
-------------------------------------
Logistic Regression Score: 0.5614920071047957
             precision    recall  f1-score   support

          0       0.49      0.71      0.58     11974
          1       0.68      0.45      0.54     16176

avg / total       0.60      0.56      0.56     28150


NN Score: 0.4152753108348135
             precision    recall  f1-score   support

          0       0.35      0.43      0.38     11974
          1       0.49      0.41      0.44     16176

avg / total       0.43      0.42      0.42     28150


7
-------------------------------------
Logistic Regression Score: 0.5274118738404453
             precision    recall  f1-score   support

          0       0.41      0.88      0.56      7307
          1       0.85      0.35      0.49     14253

avg / total       0.70      0.53      0.51     21560


NN Score: 0.3927179962894249
             precision    recall  f1-score   support

          0       0.24      0.37      0.29      7307
          1       0.56      0.40      0.47     14253

avg / total       0.45      0.39      0.41     21560
'''


