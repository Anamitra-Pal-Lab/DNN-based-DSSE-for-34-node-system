from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam, SGD
import pandas as pd
import numpy as np
from sklearn import metrics
from matplotlib import pyplot
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from tensorflow.keras import regularizers
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical
from time import time 
from datetime import datetime
import os
#from deepreplay.replay import Replay
#from deepreplay.callbacks import ReplayData
from tensorflow.keras.backend import gradients
import tensorflow as tf
#from keras.callbacks import ModelCheckpoint
import h5py
from tensorflow.keras.layers import LeakyReLU
from sklearn.metrics import r2_score
import random as python_random
from tensorflow.keras import losses
from sklearn.linear_model import LinearRegression
np.random.seed(123)
python_random.seed(123)
tf.random.set_seed(123)

######################### State estimator######################################
target_phase = 1 # phase A:1 , phase B:2 , phase C:3
target_state = 1 # magnitude:1 , angle:2


x_train = pd.read_csv('train_input.csv').to_numpy()
y_train_all = pd.read_csv('train_output.csv').to_numpy()

x_test = pd.read_csv('test_input.csv').to_numpy()
y_test_all = pd.read_csv('test_output.csv').to_numpy()



target_phase_index_A = pd.read_csv('phase_A_output_indexes.csv').to_numpy().flatten()
target_phase_index_B = pd.read_csv('phase_B_output_indexes.csv').to_numpy().flatten()
target_phase_index_C = pd.read_csv('phase_C_output_indexes.csv').to_numpy().flatten()

All_phases_index = []
All_phases_index.append((2*target_phase_index_A-1)-1)
All_phases_index.append((2*target_phase_index_B-1)-1)
All_phases_index.append((2*target_phase_index_C-1)-1)
All_phases_index.append((2*target_phase_index_A)-1)
All_phases_index.append((2*target_phase_index_B)-1)
All_phases_index.append((2*target_phase_index_C)-1)








# if target_phase == 1:
#     if target_state == 1:
#         temp_index = (2*target_phase_index_A-1)-1
#         y_train = y_train_all[:,temp_index[0:]]
#         y_test = y_test_all[:,(2*target_phase_index_A-1)-1]
#     else:
#         y_train = y_train_all[:,(2*target_phase_index_A)-1] 
#         y_test = y_test_all[:,(2*target_phase_index_A)-1]
# elif target_phase == 2:
#     if target_state == 1:
#         y_train = y_train_all[:,(2*target_phase_index_B-1)-1] 
#         y_test = y_test_all[:,(2*target_phase_index_B-1)-1]
#     else:
#         y_train = y_train_all[:,(2*target_phase_index_B)-1] 
#         y_test = y_test_all[:,(2*target_phase_index_B)-1]
# elif target_phase == 3:
#     if target_state == 1:
#         y_train = y_train_all[:,(2*target_phase_index_C-1)-1]  
#         y_test = y_test_all[:,(2*target_phase_index_C-1)-1]
#     else:
#         y_train = y_train_all[:,(2*target_phase_index_C)-1]  
#         y_test = y_test_all[:,(2*target_phase_index_C)-1]
    
 
        


# validation_percentage = 20/100
# x_val = x_train[int(x_train.shape[0]*(1-validation_percentage)):,:]
# y_val = y_train[int(y_train.shape[0]*(1-validation_percentage)):,:] 
# x_train = x_train[0:int(x_train.shape[0]*(1-validation_percentage)),:]
# y_train = y_train[0:int(y_train.shape[0]*(1-validation_percentage)),:]




def training (input_train,output_train,input_val,output_val,input_test,output_test):
    model = Sequential()
     #model.add(Dense(400, input_dim=x_train.shape[1],kernel_regularizer=regularizers.l2(0.01) ,activation='relu')) # Hidden 1
    model.add(Dense(200, activation='relu', input_dim=input_train.shape[1], kernel_initializer='he_normal')) # Hidden 1
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(200, activation='relu', kernel_initializer='he_normal')) # Hidden 2
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(200, activation='relu', kernel_initializer='he_normal')) # Hidden 3
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(200, activation='relu', kernel_initializer='he_normal')) # Hidden 4
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(200, activation='relu', kernel_initializer='he_normal')) # Hidden 5
    model.add(BatchNormalization())
    model.add(Dropout(0.3))


    model.add(Dense(output_train.shape[1], activation='linear',kernel_initializer='he_normal')) # Output
    
    loss_fn = losses.MeanSquaredError()
    Adam(learning_rate=0.09456, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(loss=loss_fn, optimizer='adam', metrics=['MAE'])
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=10, min_lr=0.0001)
    filepath="best_weights.hdf5"
    checkpoint = ModelCheckpoint(filepath,save_weights_only=True, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    model.fit(input_train,output_train,verbose=1,epochs=4000,validation_data = (input_val,output_val),callbacks=[checkpoint,reduce_lr])
    model.load_weights("best_weights.hdf5")
    pred = model.predict(input_test)
    MAPE_each_node = np.sum(abs((output_test-pred)/output_test),axis = 0)/output_test.shape[0]*100
    MAE_each_node = np.sum(abs(output_test-pred),axis = 0)/output_test.shape[0]#*180/np.pi

    
     
    R2_Score_each_node = []
    for i in range(output_test.shape[1]):
        R2_Score_each_node.append(r2_score(output_test[:,i], pred[:,i]))
    return MAPE_each_node, MAE_each_node, R2_Score_each_node

    # if check == 'mag':
    #     temp1 = abs((output_test-pred)/output_test)   
    #     return mag_MAPE_each_node, R2_Score
    # elif check == 'phase':
    #     temp1 = abs(pred - output_test)*180/np.pi
    #     return phase_mae_each_node,R2_Score

# # # Build the neural network
# model = Sequential()
# #  #model.add(Dense(400, input_dim=x_train.shape[1],kernel_regularizer=regularizers.l2(0.01) ,activation='relu')) # Hidden 1
# model.add(Dense(60, activation='relu', input_dim=x_train.shape[1], kernel_initializer='he_normal')) # Hidden 1
# model.add(BatchNormalization())
# model.add(Dropout(0.2))

# model.add(Dense(60, activation='relu', kernel_initializer='he_normal')) # Hidden 2
# model.add(BatchNormalization())
# model.add(Dropout(0.2))

# model.add(Dense(500, activation='relu', kernel_initializer='he_normal')) # Hidden 3
# model.add(BatchNormalization())
# model.add(Dropout(0.5))

# model.add(Dense(500, activation='relu', kernel_initializer='he_normal')) # Hidden 4
# model.add(BatchNormalization())
# model.add(Dropout(0.5))

# model.add(Dense(500, activation='relu', kernel_initializer='he_normal')) # Hidden 5
# model.add(BatchNormalization())
# model.add(Dropout(0.5))


# model.add(Dense(y_train.shape[1], activation='linear',kernel_initializer='he_normal')) # Output



# loss_fn = losses.MeanSquaredError()
# Adam(learning_rate=0.1, beta_1=0.9, beta_2=0.999, amsgrad=False) #learning_rate=0.09456
# model.compile(loss=loss_fn, optimizer='adam', metrics=['MAE'])
# #ES = EarlyStopping (monitor='val_loss',patience=20,verbose=1,restore_best_weights=True)
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=10, min_lr=0.0001)
# filepath="weights.best.hdf5"
# checkpoint = ModelCheckpoint(filepath,save_weights_only=True, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
# #logdir = "logs/scalars/" #+ datetime.now().strftime("%Y%m%d-%H%M%S")

# #logdir = "logs"
# #TB = TensorBoard(log_dir=logdir,write_grads=True,histogram_freq=10)
# #checkpoint = ModelCheckpoint('weights{epoch:08d}.h5', monitor='loss', verbose=1,save_weights_only=True, mode='auto', period=1)
# #history = model.fit(x_train,y_train,verbose=1,epochs=20,validation_split=0.2,callbacks=[ES])
# history = model.fit(x_train,y_train,verbose=1,epochs=4000,validation_data = (x_val,y_val),callbacks=[checkpoint,reduce_lr])
# #history = model.fit(np.reshape(x_train[0:5][:],(5,36)♥),np.reshape(y_train[0:5][:],(5,258)),batch_size=1,verbose=1,epochs=30,validation_split=0.2)

# #_, train_acc = model.evaluate(x_train, y_train, verbose=0)
# #_, test_acc = model.evaluate(x_test, y_test, verbose=0)
# #print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
# model.load_weights("weights.best.hdf5")
# start_SE = time()
# pred = model.predict(x_test)
# end_SE = time()
# elapsed_time = end_SE - start_SE


################################### end function ############################
All_errors= []
for i in range(len(All_phases_index)):
    y_train_temp = y_train_all[:,All_phases_index[i]] # two -1 for mag and one -1 for ang
    y_test_temp = y_test_all[:,All_phases_index[i]]
    validation_percentage = 20/100
    x_val_temp = x_train[int(x_train.shape[0]*(1-validation_percentage)):,:]
    y_val_temp = y_train_temp[int(y_train_all.shape[0]*(1-validation_percentage)):,:] 
    x_train_temp = x_train[0:int(x_train.shape[0]*(1-validation_percentage)),:]
    y_train_temp = y_train_temp[0:int(y_train_all.shape[0]*(1-validation_percentage)),:] 
    All_errors.append(training(x_train_temp,y_train_temp ,x_val_temp,y_val_temp,x_test,y_test_temp))
    print(i)

len(int(32))

error_corresponding = []
r2_corresponding = []
temp1 = []
temp2 = []
for i in range(len(All_errors)):
    temp1 = All_errors[i][0]
    temp2 = All_errors[i][1]
    error_corresponding.append(temp1)
    r2_corresponding.append(temp2)

all_mae_mape =[]
for i in range(len(error_corresponding)):
    all_mae_mape = np.append(all_mae_mape,error_corresponding[i])
average_all_mae_mape = np.mean(all_mae_mape)
len(int(32))

################################### end function #############################



R2_Score = []
for i in range(y_test.shape[1]):
    R2_Score.append(r2_score(y_test[:,i], pred[:,i]))







# R2_Score = []
# MAE_each_node = []
# MAPE_each_node = []
# max_error_value = []
# max_error_index = []
# for i in range(y_test.shape[1]):
#     R2_Score.append(r2_score(y_test[:,i], pred[:,i]))
#     MAE_each_node.append(mean_absolute_error(pred[:,i], y_test[:,i]))
#     MAPE_each_node.append(mean_absolute_percentage_error(pred[:,i], y_test[:,i]))
#     max_error_value.append(max(abs(pred[:,i]-y_test[:,i])))
#     max_error_index.append(np.argmax(abs(pred[:,i]-y_test[:,i])))



# max_error_value_linear = []
reg = LinearRegression().fit(x_train,y_train,)
predict_linear = reg.predict(x_test)
# MAE_each_node_linear = []
R2_score_linear = []
for i in range(y_test.shape[1]):
    R2_score_linear.append(r2_score(y_test[:,i], predict_linear[:,i]))
    # max_error_value_linear.append(max(abs(predict_linear[:,i]-y_test[:,i])))
    # MAE_each_node_linear.append(mean_absolute_error(predict_linear[:,i], y_test[:,i]))
# pyplot.figure()
# pyplot.plot(R2_score_linear, label='R2_linear')
# pyplot.legend( loc='lower right')
############################

# pyplot.figure()
# pyplot.plot(y_test[0:20], label='y_test')
# pyplot.plot(pred[0:20], label='pred') 
# pyplot.plot(Lin_pred[0:20], label='Lin_pred') 
# pyplot.legend( loc='upper left')
    

fig, ax = pyplot.subplots()

# Plot the first curve
ax.plot(R2_score_linear, label='R2 Linear')

# Plot the second curve
ax.plot(R2_Score, label='DNN Linear')


###################################### function #################################

def training (input_train,output_train,input_val,output_val,input_test,output_test,check):
    model = Sequential()
     #model.add(Dense(400, input_dim=x_train.shape[1],kernel_regularizer=regularizers.l2(0.01) ,activation='relu')) # Hidden 1
    model.add(Dense(200, activation='relu', input_dim=input_train.shape[1], kernel_initializer='he_normal')) # Hidden 1
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(200, activation='relu', kernel_initializer='he_normal')) # Hidden 2
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(200, activation='relu', kernel_initializer='he_normal')) # Hidden 3
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(200, activation='relu', kernel_initializer='he_normal')) # Hidden 4
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(200, activation='relu', kernel_initializer='he_normal')) # Hidden 5
    model.add(BatchNormalization())
    model.add(Dropout(0.3))


    model.add(Dense(output_train.shape[1], activation='linear',kernel_initializer='he_normal')) # Output
    
    loss_fn = losses.MeanSquaredError()
    #loss_fn = losses.LogCosh()
    Adam(learning_rate=0.09456, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(loss=loss_fn, optimizer='adam', metrics=['MAE'])
    #ES = EarlyStopping (monitor='val_loss',patience=20,verbose=1,restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=10, min_lr=0.0001)
    filepath="weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath,save_weights_only=True, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    #logdir = "logs/scalars/" #+ datetime.now().strftime("%Y%m%d-%H%M%S")
    
    model.fit(input_train,output_train,verbose=1,epochs=4000,validation_data = (input_val,output_val),callbacks=[checkpoint,reduce_lr])
    model.load_weights("weights.best.hdf5")
    pred = model.predict(input_test)
    mag_MAPE_each_node = np.sum(abs((output_test-pred)/output_test),axis = 0)/output_test.shape[0]*100
    phase_mae_each_node = np.sum(abs(output_test-pred),axis = 0)/output_test.shape[0]*180/np.pi
    mag_mae_each_node = np.sum(abs(output_test-pred),axis = 0)/output_test.shape[0]
    
     
    tolerance_interval_input = []
    R2_Score = []
    for i in range(output_test.shape[1]):
        R2_Score.append(r2_score(output_test[:,i], pred[:,i]))
    if check == 'mag':
        temp1 = abs((output_test-pred)/output_test)
        tolerance_interval_input = np.matrix.flatten(temp1)        
        return mag_MAPE_each_node, R2_Score, tolerance_interval_input, mag_mae_each_node
    elif check == 'phase':
        temp1 = abs(pred - output_test)*180/np.pi
        tolerance_interval_input = np.matrix.flatten(temp1)
        return phase_mae_each_node,R2_Score, tolerance_interval_input

All_phases_index = [targeted_phase_y_index_A,targeted_phase_y_index_B,targeted_phase_y_index_C]
All_errors= []
for i in range(len(All_phases_index)):
    y_train_modified = y_train[:,(2*All_phases_index[i]-1)-1] # two -1 for mag and one -1 for ang
    y_test_modified = y_test [:,(2*All_phases_index[i]-1)-1]
    validation_percentage = 20/100
    x_val_modified = x_train[int(x_train.shape[0]*(1-validation_percentage)):,:]
    y_val_modified = y_train_modified[int(y_train.shape[0]*(1-validation_percentage)):,:] 
    x_train_modified = x_train[0:int(x_train.shape[0]*(1-validation_percentage)),:]
    y_train_modified = y_train_modified[0:int(y_train.shape[0]*(1-validation_percentage)),:] 
    All_errors.append(training(x_train_modified,y_train_modified ,x_val_modified,y_val_modified,x_test,y_test_modified,'mag'))
    print(i)



error_corresponding = []
r2_corresponding = []
temp1 = []
temp2 = []
for i in range(len(All_errors)):
    temp1 = All_errors[i][0]
    temp2 = All_errors[i][1]
    error_corresponding.append(temp1)
    r2_corresponding.append(temp2)

all_mae_mape =[]
for i in range(len(error_corresponding)):
    all_mae_mape = np.append(all_mae_mape,error_corresponding[i])
average_all_mae_mape = np.mean(all_mae_mape)

