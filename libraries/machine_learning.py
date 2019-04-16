import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing import Process
from multiprocessing import Manager
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report as cr 





def prediction_probability_slices(y_true, predictions, probabilities, slice_step = .05):
    
    