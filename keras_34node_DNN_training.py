from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, BatchNormalization, Dropout
from tensorflow.keras.callbacks import  ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam, SGD
import pandas as pd
import numpy as np
from matplotlib import pyplot
import tensorflow as tf
import random as python_random
from tensorflow.keras import losses

np.random.seed(123)
python_random.seed(123)
tf.random.set_seed(123)

######################### data preparation ######################################



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




######################### DNN model creation ######################################


def training (input_train,output_train,input_val,output_val,input_test,output_test):
    model = Sequential()
    number_of_neurons = 200
    dropout_rate = 30/100
    model.add(Dense(number_of_neurons, activation='relu', input_dim=input_train.shape[1], kernel_initializer='he_normal')) # Hidden 1
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    model.add(Dense(number_of_neurons, activation='relu', kernel_initializer='he_normal')) # Hidden 2
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    model.add(Dense(number_of_neurons, activation='relu', kernel_initializer='he_normal')) # Hidden 3
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    model.add(Dense(number_of_neurons, activation='relu', kernel_initializer='he_normal')) # Hidden 4
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    model.add(Dense(number_of_neurons, activation='relu', kernel_initializer='he_normal')) # Hidden 5
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))


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
    MAE_each_node = np.sum(abs(output_test-pred),axis = 0)/output_test.shape[0]*180/np.pi
    return MAPE_each_node, MAE_each_node, output_test, pred


######################### DNN training for each phase and state type ######################################

results = []
for i in range(len(All_phases_index)):
    y_train_temp = y_train_all[:,All_phases_index[i]]
    y_test_temp = y_test_all[:,All_phases_index[i]]
    validation_percentage = 20/100
    x_val_temp = x_train[int(x_train.shape[0]*(1-validation_percentage)):,:]
    y_val_temp = y_train_temp[int(y_train_all.shape[0]*(1-validation_percentage)):,:] 
    x_train_temp = x_train[0:int(x_train.shape[0]*(1-validation_percentage)),:]
    y_train_temp = y_train_temp[0:int(y_train_all.shape[0]*(1-validation_percentage)),:] 
    results.append(training(x_train_temp,y_train_temp ,x_val_temp,y_val_temp,x_test,y_test_temp))  # training function

    
######################### exporting the results ######################################

output_file_names = ["results/phase_A_mag_results.xlsx","results/phase_B_mag_results.xlsx","results/phase_C_mag_results.xlsx",\
                     "results/phase_A_ang_results.xlsx","results/phase_B_ang_results.xlsx","results/phase_C_ang_results.xlsx"]
for i in range(len(results)):
    with pd.ExcelWriter(output_file_names[i], engine='openpyxl') as writer:
        pd.DataFrame(results[i][2]).to_excel(writer, sheet_name='ground_truth', index=False,header=False)
        pd.DataFrame(results[i][3]).to_excel(writer, sheet_name='state_estimates', index=False,header=False)
    ### ploting results 
    pyplot.figure()
    pyplot.plot(results[i][0], label='Mean Absolute Percentage Error')
    pyplot.title(output_file_names[i][8:-13])
    pyplot.ylabel('MAPE [%]')
    pyplot.xlabel('Node number')
    pyplot.figure()
    pyplot.plot(results[i][1], label='Mean Absolute Error')
    pyplot.title(output_file_names[i][8:-13])
    pyplot.ylabel('MAE [%]')
    pyplot.xlabel('Node number')

