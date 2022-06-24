
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 15:39:06 2022

@author: Tuf
"""
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.activations import linear
from tensorflow.keras.layers import LSTM, Dense,Dropout


class EDA():
    def __init__(self):
        pass
    
    def plot_graph(self,df):
        '''
        

        Parameters
        ----------
        df : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        plt.figure()
        plt.plot(df['cases_new'])
        plt.plot(df['cases_recovered'])
        plt.plot(df['cases_active'])
        plt.legend(['new','recovered','active'])
        plt.show()
    
        
        
class ModelCreation():
    def __init__(self):
        pass
    
    def simple_lstm_layer(self,X_train,num_node=64,drop_rate=0.3,output_node=1):
        model=Sequential()
        model.add(Input(shape=(np.shape(X_train)[1],1))) #input_length, #feature
        model.add(LSTM(num_node,return_sequences=(True)))
        model.add(Dropout(drop_rate))
        model.add(LSTM(num_node))
        model.add(Dropout(drop_rate))
        model.add(Dense(128))
        model.add(Dense(128))
        model.add(Dropout(drop_rate))
        model.add(Dense(output_node,activation='linear'))#Output layer
        model.summary()
        
        return model

class model_evaluation:
        def plot_predicted_graph(self,df_test,predicted,mms):
            plt.figure()
            plt.plot(df_test,'b',label='actual cases')
            plt.plot(predicted,'r',label='predicted cases')
            plt.legend()
            plt.show()


            plt.figure()
            plt.plot(mms.inverse_transform(df_test),'b',label='actual cases')
            plt.plot(mms.inverse_transform(predicted),'r',label='predicted cases')
            plt.legend()
            plt.show()

    