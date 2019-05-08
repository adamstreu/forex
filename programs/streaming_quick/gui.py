import numpy as np
import pandas as pd
import sys
import os
import pickle
from PyQt5.QtWidgets import QWidget, QLabel, QComboBox, QApplication, QSlider
from PyQt5.QtWidgets import QPushButton, QFrame, QTabWidget, QVBoxLayout, QHBoxLayout
from PyQt5.QtGui import QColor
from PyQt5.QtCore import Qt
os.chdir('/northbend')
from libraries.oanda import create_order

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


plot_params = {
                'interval': 1500,
                'currency_1': 'aud',
                'currency_2': 'cad',
                'pair': 'AUD_CAD',
                'channels': False,
                'supports': False}

# Parameters File
params_file =  '/northbend/tmp/plot_parameters.py'


def save_obj(obj, file):
    with open(file, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(file ):
    with open(file, 'rb') as f:
        return pickle.load(f)


class Example(QWidget):

    
    def __init__(self, pairs_list, currencies):
        super().__init__()
        
        self.instrument = None
        self.direction = None
        self.pairs = pairs_list
        self.currencies = currencies
        self.initUI()
        
        
    def initUI(self):      
        
        # Add up Combo Box        
        self.combo_up = QComboBox(self)
        for currency in currencies:
            self.combo_up.addItem(currency)        
        self.combo_up.activated[str].connect(self.get_pair) 

        # Add Down Combo Box        
        self.combo_down = QComboBox(self)
        for currency in currencies:
            self.combo_down.addItem(currency)   
        self.combo_down.activated[str].connect(self.get_pair)    
            
        # Add Labels for bomboboxes
        self.label_up = QLabel('UP', self)
        self.label_down = QLabel('DOWN', self)
        
        # Add Placement and Close Button
        self.button_go = QPushButton('Go !', self)
        self.button_go.clicked.connect(self.go)
        self.button_stop = QPushButton('Stop !', self)
        self.button_stop.clicked.connect(self.stop)        
        
        # Add An Interval Switch - eventually get value from db
        self.slider_interval = QSlider(Qt.Horizontal, self)
        self.slider_interval.setMinimum(10)
        self.slider_interval.setMaximum(5000)
        self.slider_interval.valueChanged.connect(self.update_parameters)
        
        # Add Slider Label
        label = str(self.slider_interval.value())
        self.slider_interval_label = QLabel(label, self)
        
        # Hide Stop Button
        self.button_stop.hide()
        
        # Arrange Window Items
        self.combo_up.move(50, 50)
        self.combo_down.move(150, 50)
        self.label_up.move(50, 25)
        self.label_down.move(150, 25)
        self.button_go.move(50, 100)
        self.button_stop.move(25, 25)
        self.button_go.resize(200, 100)
        self.button_stop.resize(200, 225)
        self.setGeometry(300, 300, 300, 200)
        self.setWindowTitle('Short Strategy')
        self.show()
        
    
    def update_parameters(self):
        params_file =  '/northbend/tmp/plot_parameters.py'
        params = load_obj(params_file)
        params['interval'] = int(self.slider_interval.value())
        params['instrument'] = self.instrument
        params['direction'] = self.direction
        params['currency_1'] = self.combo_up.currentText().upper()
        print(self.combo_up.currentText().upper())
        params['currency_2'] = self.combo_down.currentText().upper()
        save_obj(params, params_file)      
        print(params)


    def go(self):
        
        # Create Order
        order = create_order(self.instrument, self.direction)
        print(order)
        
        
        # Hide curreny Widget
        self.combo_down.hide()
        self.combo_up.hide()
        self.button_go.hide()
        self.button_stop.show()
        self.label_up.hide()
        self.label_down.hide()    
        
    def stop(self):
        
        # Create Order
        if self.direction == 'long':
            reverse = 'short'
        else:
            reverse = 'long'
        order = create_order(self.instrument, reverse)
        print(order)
        
        # Hide curreny Widget
        self.combo_down.show()
        self.combo_up.show()
        self.button_go.show()
        self.button_stop.hide()
        self.label_up.show()
        self.label_down.show()   
        
    
    def get_pair(self):
        try:
            if self.combo_up.currentText().upper() != \
                    self.combo_down.currentText().upper():
                try_pair = self.combo_up.currentText().upper()
                try_pair += '_'
                try_pair += self.combo_down.currentText().upper() 
                if try_pair in self.pairs:
                    self.instrument = try_pair
                    self.direction  = 'long'
                else:
                    self.instrument =  self.combo_down.currentText().upper()
                    self.instrument += '_'
                    self.instrument += self.combo_up.currentText().upper()
                    self.direction   = 'short'
            else:
                self.instrument = None
                self.direction  = None
        except Exception as e:
            print(e)
        print('asdf')
        self.update_parameters() 
        print('1234')
            
            
  

            
        
        
        
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Example(pairs, currencies)
    sys.exit(app.exec_())       
        
        
