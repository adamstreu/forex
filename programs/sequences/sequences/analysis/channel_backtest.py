from datetime import timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy import stats
import scipy
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import os; os.chdir('/sequences')
from libraries.oanda import get_candles


'''

TO DO NEXT:
    
    Does normalizing any data help?
    
    Set up the iteration.
    Go through the candles
    analyze the outcomes
    collect the statistics.



Conditions to meet:
    channel breakout
    in direction opposite of channelk slope
    where % out of channel is less than __
    
Bimodal distribution of transofrmed closing values

Slope Ratio is not any good - needs to get fixed.

Values to track while going through to test laster:
    slope
    percent below, above, summed
    channel break, placement
    regression line stats
    amount above, 
    distance between inner and outer channels
    how well lines match, what sloep they are at - more mismatch might be good (   < .25 ? )
    
    need to do something with the distributions
    channel widths
    
    slope to slope ratio to left / right hist ratio
    
Questions:
    if it hits outer - will it keep going?
    do tails matter?
    
    
good example and loss:

    _from       = '2018-04-08T08:00:00Z'
    _to         = '2018-04-13T08:00:00Z'

    _from       = '2018-03-08T08:00:00Z'
    _to         = '2018-03-13T05:50:00Z' 
    
    _from       = '2017-07-15T08:00:00Z'   # Good example of slope going opposite direction - tight slope to 
    _to         = '2017-07-20T03:00:00Z'



'''


# Instrument
instrument  = 'EUR_USD'
granularity = 'M1'
_from       = '2014-07-17T00:00:00Z'
_to         = '2014-08-19T00:00:00Z'
_end        = pd.to_datetime(_to) + timedelta(hours=180)
_end        = str(_end).replace(' ', 'T') + 'Z'


# Summary filters
perc_outside_channel = .15
channel_break_top = False
channel_break_bottom = False
channel_placement = False
outer_channel_break = False

# Import Candles
candles = get_candles(instrument, granularity, _from, _to)
candles_after = get_candles(instrument, granularity, _from, _end)
#
# Prepare Data Slice for linearg regression
start = 0
stop  = 100000
closings = candles.loc[start: stop, 'midclose']

# Shift Data down to Zero.  Reset Index
closings = closings - closings.values[0]
closings_x = np.arange(closings.shape[0]).reshape(-1, 1)
closings_long = np.arange(candles_after.shape[0]).reshape(-1, 1)
# Calculate Line, slope and summary statistics through linear regression
regr = linear_model.LinearRegression()
regr.fit(closings_x, closings)
line = regr.predict(closings_x)
slope = regr.coef_[0]
error = mean_squared_error(closings, line)
variance_score = r2_score(closings, line)
line_long = regr.predict(closings_long)

# Transform closings horizontal using regression line
trans_closings = closings.values - line
trans_closings_long = candles_after.midclose.values - candles_after.midclose.values[0] - line_long

# Create top and bottom channels that are most perpendicular to each other.
###############################################################################
min_distance = 1000 # set min mean variable for iterations
min_distance_perc = 0
for perc in np.arange(2, 26, 1):  

    # Get indices and solution of largest and smallest % of numbers in closings
    _x_percent_of_closing = perc
    qty = int(_x_percent_of_closing * closings.shape[0] / 100)
    top = np.argpartition(trans_closings, -qty)[-qty:]
    bottom = np.argpartition(trans_closings, qty)[:qty]
    
    # Calculate linear Regression on Top and Bottom Line
    regr = linear_model.LinearRegression()
    regr.fit(top.reshape(-1, 1), trans_closings[top].reshape(-1, 1))
    top_line = regr.predict(np.arange(closings.shape[0]).reshape(-1,1)) 
    regr = linear_model.LinearRegression()
    regr.fit(bottom.reshape(-1, 1), trans_closings[bottom].reshape(-1, 1))
    
    # Calculate top and bottom channel lines by finding min distance between 2
    bottom_line = regr.predict(np.arange(closings.shape[0]).reshape(-1,1))
    mid = int(trans_closings.shape[0] / 2)
    top_lowered = top_line - (top_line[mid] - bottom_line[mid])
    bottom_raised = (top_line[mid] - bottom_line[mid]) + bottom_line

    # differences in lines.  Collect smallest
    if abs(top_line[0] - bottom_raised[0]) < min_distance:
        min_distance = abs(top_line[0] - bottom_raised[0])
        min_distance_perc = perc
###############################################################################

## Get indices and solution of largest and smallest _x_ % of numbers in closings
_x_percent_of_closing = min_distance_perc
qty = int(_x_percent_of_closing * closings.shape[0] / 100)
top = np.argpartition(trans_closings, -qty)[-qty:]
bottom = np.argpartition(trans_closings, qty)[:qty]

# Calculate linear Regression on Top and Bottom Line
regr = linear_model.LinearRegression()
regr.fit(top.reshape(-1, 1), trans_closings[top].reshape(-1, 1))
top_line = regr.predict(np.arange(closings.shape[0]).reshape(-1,1)) 
regr = linear_model.LinearRegression()
regr.fit(bottom.reshape(-1, 1), trans_closings[bottom].reshape(-1, 1))

# Calculate top and bottom channel lines by finding minimal distance between 2
bottom_line = regr.predict(np.arange(closings.shape[0]).reshape(-1,1))
mid = int(trans_closings.shape[0] / 2)
top_lowered = top_line - (top_line[mid] - bottom_line[mid])
bottom_raised = (top_line[mid] - bottom_line[mid]) + bottom_line

# Calculate average top and bottom lines to form channel
channel_top          = (top_line + bottom_raised) / 2
channel_bottom       = (bottom_line + top_lowered) / 2
#channel_top_outer    =  channel_top + (trans_closings[np.argmax(trans_closings)] - 
#                                      channel_top[np.argmax(trans_closings)][0])
#channel_bottom_outer =  channel_bottom + (trans_closings[np.argmin(trans_closings)] - 
#                                       channel_bottom[np.argmin(trans_closings)][0])
outer_top    = (trans_closings[np.argmax(trans_closings)] - 
               channel_top[np.argmax(trans_closings)][0])
outer_bottom = (trans_closings[np.argmin(trans_closings)] - 
               channel_bottom[np.argmin(trans_closings)][0])
outer = (abs(outer_top) + abs(outer_bottom)) / 2
channel_top_outer = channel_top + outer
channel_bottom_outer = channel_bottom - outer

# Sumamry statistics
above_channel_perc = (trans_closings >= channel_top).mean()
below_channel_perc = (trans_closings <= channel_bottom).mean()
pips = int((closings.max() - closings.min()) / .0001)
channel_slope = ((channel_bottom[-1] - channel_top[0])/closings.shape[0])[0]
channel_width = int((channel_top[0][0] - channel_bottom[0][0]) / .0001)
channel_top_width = int(channel_top[0][0] / .0001)
channel_bottom_width = -1 * int(channel_bottom[0][0] / .0001)
channel_outer_width = int((channel_top_outer[0][0] - channel_top[0][0]) / .0001)
hist = np.histogram(closings.values-line, bins=50)

# Channel crosisng opposite direction of slope
if trans_closings[-1] <= channel_bottom[-1]:
    channel_break_bottom = True
if trans_closings[-1] >= channel_top[-1]:
    channel_break_top = True
if slope > 0 and channel_break_bottom:
    channel_placement = True
if slope < 0 and channel_break_top:
    channel_placement = True
if trans_closings[-1] <= channel_bottom_outer[-1]:
    outer_channel_break = True
if trans_closings[-1] >= channel_top_outer[-1]:
    outer_channel_break = True


# Create histogram of transformed closings and find peaks
trans_hist = np.histogram(trans_closings, 50) 
x_dist = ((trans_hist[1][:-1] + trans_hist[1][1:]) / 2)
y_dist = trans_hist[0]
trans_dist_peaks = signal.find_peaks_cwt(y_dist, np.arange(1, 10))
# OR
rv_hist = scipy.stats.rv_histogram(np.histogram(trans_closings, 100)) # provides many evaluations



# Also, k mean was interesting on  trns_closing
clusters = 10
data = np.c_[closings_x.ravel(), trans_closings.ravel()]
k = scipy.cluster.vq.kmeans(data, clusters)


# plot closing values with linear regression line
plt.figure(figsize=(14, 8))
#plt.plot(np.arange(closings.shape[0]), closings.values, color='orange')
plt.plot(np.arange(closings.shape[0]), line, color='blue')
plt.plot(np.arange(closings_long.shape[0]), candles_after.midclose.values - 
         candles_after.midclose.values[0], color='grey')
# plot horizontal transformation with linear regression, top and bottom channels
plt.figure(figsize=(14, 8))
plt.plot(np.arange(closings.shape[0]), trans_closings, color='grey')
plt.plot(np.arange(closings.shape[0]), line-line , color='green')
plt.bar(top, trans_closings[top])
plt.bar(bottom, trans_closings[bottom])
plt.plot(closings_x, top_line)
plt.plot(closings_x, bottom_line)
plt.plot(closings_x, bottom_raised)
plt.plot(closings_x, top_lowered)
plt.plot(closings_x, channel_top, color='blue')
plt.plot(closings_x, channel_bottom, color='blue')
plt.plot(closings_x, channel_top_outer, color='blue')
plt.plot(closings_x, channel_bottom_outer, color='blue')
plt.plot(closings_long[trans_closings.shape[0]:], trans_closings_long[trans_closings.shape[0]:], color='grey')
#plt.plot(closings_long, trans_closings_long, color='lightgreen')
for each in x_dist[trans_dist_peaks]:
    plt.plot(closings_x, np.ones(closings_x.shape[0]) * each, color='black')

# Plot k means
for i in range(clusters):
    plt.plot(k[0][i][0], k[0][i][1], 'o', color='black', markersize=8)
# plot histograms with kernels for both original closing values and transformed
plt.figure(figsize=(14, 2))
sns.distplot(closings.values, color='orange',  kde_kws={"label": "Closing"}, bins=50)
plt.figure(figsize=(14, 2))
sns.distplot(trans_closings, kde_kws={"label": "Transformed"}, bins=50)
plt.figure(figsize = (14, 2))
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


