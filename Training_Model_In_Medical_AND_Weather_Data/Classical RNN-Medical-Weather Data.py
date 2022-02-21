from datetime import datetime

#Importer des bibio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Dropout, Activation
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow import keras
import tensorflow as tf
import pennylane as qml
from pennylane.operation import Operation, AnyWires


#******** Étape 1: Collecte des données *********************

df_confirmed = pd.read_csv('../Dataset/Tunisie_Data_Avril.csv')
df_confirmedd = pd.read_csv('../Dataset/MultiFinal.csv')
df_confirmeddd = pd.read_csv('../Dataset/MultiFinalHumdite.csv')

df_confirmed.head()
print(df_confirmed)


##******************* structuring times eries data *************
df_confirmed2 = pd.DataFrame(df_confirmed[df_confirmed.columns[4:]].sum(),columns=["confirmed"])
df_confirmed2.index = pd.to_datetime(df_confirmed2.index,format='%m/%d/%y')
df_confirmed2.tail()
df_new1 = df_confirmed2[["confirmed"]]


##******************* structuring times eries data *************
df_confirmed2 = pd.DataFrame(df_confirmedd[df_confirmedd.columns[4:]].sum(),columns=["temperature"])
df_confirmed2.index = pd.to_datetime(df_confirmed2.index,format='%m/%d/%y')
df_confirmed2.tail()
df_new2 = df_confirmed2[["temperature"]]

##******************* structuring times eries data *************
df_confirmed3 = pd.DataFrame(df_confirmeddd[df_confirmedd.columns[4:]].sum(),columns=["humidte"])
df_confirmed3.index = pd.to_datetime(df_confirmed3.index,format='%m/%d/%y')
df_confirmed3.tail()
df_new3 = df_confirmed3[["humidte"]]


df_new = pd.concat([df_new1,df_new2,df_new3], axis=1)
df_new.to_csv(r'C:\Users\MSI\Desktop\ProjetRepair\test.csv',index = False, header=True)

print(df_new)

from matplotlib import pyplot
values = df_new.values
# specify columns to plot
groups = [0, 1,2]
i = 1
# plot each column
pyplot.figure()
for group in groups:
    pyplot.subplot(len(groups), 1, i)
    pyplot.plot(values[:, group])
    pyplot.title(df_new.columns[group], y=0.5, loc='right')
    i += 1
pyplot.show()


plt.subplots(figsize = (25,10))
plt.ylabel('confirmed')
plt.title('Orginal Plot')
plt.plot('confirmed',data=df_new)
#plt.show()

data = df_new.filter(['confirmed'])
print(data.tail())
print("\n")
dataset = data.values
training_data_len = int(np.ceil(len(dataset) * .8))
print("dataset")
print(dataset)
print("training_data_len")
print(training_data_len)


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)
print("scaled_data")
print(scaled_data)


train_x = []
train_y = []
train_data = scaled_data[0:int(training_data_len),:]

for i in range(60,len(train_data)):
    train_x.append(train_data[i-60:i,0])
    train_y.append(train_data[i,0])
    if i<= 61:
        print(train_x)
        print(train_y)

train_x,train_y = np.array(train_x) , np.array(train_y)
train_x = np.reshape(train_x,(train_x.shape[0], train_x.shape[1], 1))

#Model SimpleRNN

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,SimpleRNN,Dropout

model = Sequential()
model.add(SimpleRNN(units = 40 , return_sequences=True , input_shape = (train_x.shape[1],1)))
model.add(SimpleRNN(units = 40 , return_sequences = True))
model.add(Dropout(0.2))
model.add(SimpleRNN(units= 40 , return_sequences = False))
model.add(Dense(25))
model.add(Dense(1))
dot_img_file = '..Dataset/model_RNN_medical_weather.png'
tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)

model.compile(optimizer='adam' , loss='mean_squared_error')
model.fit(train_x,train_y,batch_size=32 , epochs=3)

test_data = scaled_data[training_data_len - 60:,:]
test_x = []
test_y = dataset[training_data_len: , :]
for i in range(60,len(test_data)):
    test_x.append(test_data[i-60:i,0])
test_x = np.array(test_x)
test_x = np.reshape(test_x,(test_x.shape[0] , test_x.shape[1] , 1))
print(test_x.shape)

prediction = model.predict(test_x)
prediction = scaler.inverse_transform(prediction)
print("prediction is")
print((prediction))
rmse =np.sqrt(((prediction - test_y) ** 2).mean())
print("RMSE is " + str(rmse))

pd.options.mode.chained_assignment = None
#Plot/Create the data for the graph
train = data[:training_data_len]
actual = data[training_data_len:]

actual['Predictions'] = prediction

print("pred prede")
print(actual['Predictions'])


#Visualize the data
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('confirmed', fontsize=18)
plt.plot(train['confirmed'])
plt.plot(actual[['confirmed', 'Predictions']])
plt.legend(['Train', 'Actual', 'Predictions'], loc='lower right')
plt.show()