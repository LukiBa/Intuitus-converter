# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 13:45:01 2019

@author: Lukas, Based on: Tutorial: Tutorial : Facial Expression Classification Keras from Kaggle
"""
import numpy as np


def save_data(X_test, y_test, fname=''):
    """
    The function stores loaded data into numpy form for further processing
    """
    np.save( 'X_test' + fname, X_test)
    np.save( 'y_test' + fname, y_test)
    
    