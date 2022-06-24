# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 09:02:54 2022

@author: Tuf
"""

import os
import numpy as np
import pandas as pd
import pickle
import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from tensorflow.keras.utils import plot_model 
from tensorflow.keras.callbacks import TensorBoard

from modules_for_covid_cases import EDA,ModelCreation,model_evaluation
# Objective: to create a deep learning model using LSTM neural 
# network to predict new cases (cases_new) in Malaysia using the past 30 days
# of number of cases
#NaNs in timeseries- df.interpolate()
#%% Statics
CSV_PATH=os.path.join(os.getcwd(),'Dataset','cases_malaysia_train.csv')
MMS_PATH=os.path.join(os.getcwd(),'Model','mms_covid_cases_prediction.pkl')
LOG_PATH=os.path.join(os.getcwd(),'Model','Logs',datetime.datetime.now().\
                      strftime("%Y%m%d-%H%M%S"))
MODEL_SAVE_PATH=os.path.join(os.getcwd(),'Model','model.h5')
#%% EDA

# Step 1: Data Loading
df=pd.read_csv(CSV_PATH)

#%% Step 2: Data Inspection
#1) Statistical summary
df.info()
stat=df.describe().T

#2) Change the datatype from object to numeric
df['cases_new']=pd.to_numeric(df['cases_new'],errors='coerce')
df.info()
df.isna().sum

# NaNs are present in the cases_new column.

#3) Visualize data
eda=EDA()
eda.plot_graph(df)
    
#%% Step 3: Data Cleaning
#1) Interpolate NaN
df['cases_new']=df['cases_new'].interpolate()
df.info()
# have to use interpolation for time-series data 
      
#%% Step 4: Features selection

# In this step, no features are being selected. 
# The only feature that will be test and train is only 'cases_new'

#%% Step 5: Pre-processing
#use MinMaxScalling to extract the data in timeseries

mms=MinMaxScaler()
df=mms.fit_transform(np.expand_dims(df['cases_new'],axis=-1))
with open(MMS_PATH,'wb') as file:
    pickle.dump(mms,file)
    

X_train=[]
y_train=[]
win_size=30

for i in range(win_size,np.shape(df)[0]): #df.shape[0]
    X_train.append(df[i-win_size:i,0])
    y_train.append(df[i,0])
    
X_train=np.array(X_train)
y_train=np.array(y_train)
#%% Model Development

mc=ModelCreation()
model=mc.simple_lstm_layer(X_train,num_node=128)
        
plot_model(model,show_layer_names=(True), show_shapes=(True))

model.compile(optimizer='adam',loss='mse',metrics='mape')

# callbacks
#1) tensorboard callback
TensorBoard_callback = TensorBoard(log_dir=LOG_PATH)

#%% Model Training

X_train=np.expand_dims(X_train,axis=-1)

hist = model.fit(x=X_train,y=y_train,batch_size=100,epochs=150,
                 callbacks=TensorBoard_callback)
#%% Model Plotting

hist.history.keys()

plt.figure()
plt.plot(hist.history['mape'],'r--',label='mape')
plt.show()

plt.figure()
plt.plot(hist.history['loss'],'b--',label='loss')
plt.show()

model.save(MODEL_SAVE_PATH)

#%% Model deployment and analysis

CSV_TEST_PATH=os.path.join(os.getcwd(),'Dataset','cases_malaysia_test.csv')
df_test = pd.read_csv(CSV_TEST_PATH)

df_test['cases_new']=df_test['cases_new'].interpolate()

df_test=mms.transform(np.expand_dims(df_test.iloc[:,1],axis=-1))
con_test=np.concatenate((df,df_test),axis=0)
con_test=con_test[-(win_size+len(df_test)):]

X_test =[]
for i in range(win_size,len(con_test)):
    X_test.append(con_test[i-win_size:i,0])
    
X_test=np.array(X_test)
predicted=model.predict(np.expand_dims(X_test,axis=-1))
#%% To plot the graphs
me=model_evaluation()
me.plot_predicted_graph(df_test,predicted,mms)

#%% To evaluate the model from error (MSE,MAPE)

df_test_inversed=mms.inverse_transform(df_test)
predicted_inversed =mms.inverse_transform(predicted)

print('MAPE : ' + str(mean_absolute_percentage_error(df_test_inversed, 
                                                    predicted_inversed)))
print('mse : ' + str(mean_squared_error(df_test_inversed, predicted_inversed)))

print('MAE : ' + str(mean_absolute_error(df_test_inversed,predicted_inversed)/
                     sum(abs(df_test_inversed))*100))
#%% Discussion
#Since the problem is a regression problem, a linear activation function was
# used at the output layer.
# The mape and mse was used to determine the accuracy and loss of the model
# performance during training.
# The model evaluation showed that the model return 0.14% of mae error which 
# indicated that the model has the accuracy of 86% in predicting the new cases
# by passing in the predictions and the actual price in the dataset. 
# Despite the additions trainings, nodes in dense layer, and adding more layers
# to the model architecture, the model didnt improve much.
# In a conclusion, the model can improve its accuracy by adding the layers such
# simpleRNN layer, increasing number of nodes in the LSTM layer as well as
# increasing the window size would help to improve the model's pwerformance