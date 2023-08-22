import pickle
import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D
import tensorflow_estimator
from sklearn.metrics import confusion_matrix
import datetime
from sklearn.model_selection import GridSearchCV,TimeSeriesSplit
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import History

today=datetime.date.today()

TimeFrame='30T'   #'15T','30T','H','D'

data = pd.read_csv('E:/Python/Assignment_3/ETHUSDT_data_new.csv', encoding ='cp1252')
name = data.columns[0].split(';')
split = data['symbol;datetime;open;high;low;close;volume;symbol_id'].str.split(";",expand=True)
split.columns = name


# split.to_csv('E:/Python/Assignment_3/Cleaned.csv')

# split = pd.read_csv('E:/Python/Assignment_3/Cleaned.csv')
part1= split[:14496]
new = split[14497:]
part1['Inserted'] = pd.to_datetime(part1['datetime'])
new['Inserted'] = pd.to_datetime(new['datetime'], format='%d/%m/%y %H:%M')
df = pd.concat([part1,new])

# df['Inserted'] = pd.to_datetime(df['datetime'])
df.index = df.Inserted
df = df.drop(columns=['symbol','symbol_id','datetime','Inserted'])
# Calculate per change
# if Timeframe == 'H':
# df['15min_change'] = df['close'].pct_change(freq='15T')   # 15 minute frequency
# df['30min_change'] =  df['close'].pct_change(freq='30T')   # 30 minute frequency
df = df.astype(float)
df[f'{TimeFrame}_change'] =  df['close'].pct_change(freq=TimeFrame)       # 1 hour frequency


# df['1day_change'] =  df['close'].pct_change(freq='D')      # 1 day frequency
# df['30day_change'] =  df['close'].pct_change(freq='30D')  
#fill na values
# df.to_csv('E:/Python/Assignment_3/Cl_cal.csv')
# df = pd.read_csv('E:/Python/Assignment_3/Cl_cal.csv', parse_dates=['Inserted'])
# df.index = df.Inserted


df['Inserted'] = df.index

newdf_30=df[df['Inserted'].dt.minute%30==0]
newdf_hr=df[df['Inserted'].dt.minute==0]
newdf_day=newdf_hr[newdf_hr['Inserted'].dt.hour==0]

if TimeFrame == '15T':
    df = df
if TimeFrame == '30T':
    df=newdf_30
if TimeFrame == 'H':
    df=newdf_hr
if TimeFrame == 'D':
    df=newdf_day
else:
    df=newdf_day
    sequence_length = 30

if TimeFrame in {'15T'}:   
    df['low_change'] = df[['open', 'low']].pct_change(axis=1)['low']
    df['high_change'] = df[['open', 'high']].pct_change(axis=1)['high']
df.index = df.Inserted
df =df.drop_duplicates(subset=['Inserted'])
# Preprocess data
if TimeFrame in {'15T'}: 
    data=df[['low_change','high_change','volume',f'{TimeFrame}_change']]

if TimeFrame in {'30T','H','D'}: 
    data=df[['open','high','low','volume',f'{TimeFrame}_change']]
else:
    data=df[['open','high','low','volume',f'{TimeFrame}_change']]
data = data.astype(float)
# data.fillna(method="ffill", inplace=True)
data = data.interpolate(method='linear', axis=0).ffill().bfill()
# data = data.resample('15min').fillna("backfill")
# data['15min_change']= data['15min_change'].fillna(pd.rolling_mean(data['15min_change'], 6, min_periods=1))


if TimeFrame in {'15T','30T','H','D'}:    
    X = data.drop([f'{TimeFrame}_change'], axis=1).values
    y = data[f'{TimeFrame}_change'].values
# len(df[df['15min_change'].isnull()])
#manipulated to remove first row
    X=X[1:]
    y = y[1:]
else:
    X = []
    y = []
    for i in range(len(data) - sequence_length):
        X.append(data.drop([f'{TimeFrame}_change'], axis=1).values[i:i+sequence_length])
        y.append(data[f'{TimeFrame}_change'].values[i+sequence_length])
    X = np.array(X)
    y = np.array(y)
# y = np.nan_to_num(y, copy=True, nan=0.0)
# X = np.nan_to_num(X, copy=True, nan=0.0)
# np.mean(X)
# X_mean = np.mean(X, axis=0)
# X_std = np.std(X, axis=0)
# X = (X - X_mean) / X_std
# y_mean = np.mean(y, axis=0)
# y_std = np.std(y, axis=0)
# y = (y - y_mean) / y_std


# Split data into training and testing sets
train_size = int(len(X) * 0.7)
X_train = X[:train_size]
y_train = y[:train_size]
X_test = X[train_size:]
y_test = y[train_size:]


# Reshape data for CNN input
if TimeFrame in {'15T','30T','H','D'}:  
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
else:
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 4))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 4))
# # Define CNN architecture
# model = Sequential()
# model.add(Conv1D(filters=100, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], 1)))
# model.add(Conv1D(filters=100, kernel_size=2, activation='relu'))
# # model.add(Dropout(0.5))
# model.add(MaxPooling1D(pool_size=1))
# model.add(Flatten())
# model.add(Dense(100, activation='relu'))
# model.add(Dense(1))

# # Compile model
# model.compile(loss='mean_squared_error', optimizer='adam')

# Train model
# model.fit(X_train, y_train, epochs=20, batch_size=30, verbose=1)
# model.fit(X_train, y_train, epochs=100, batch_size=170, verbose=1)
sequence_length = X_train.shape[1]
if TimeFrame in {'15T','30T','H','D'}: 
    input_dim = 1
else:
    input_dim = 4

# Define the CNN model
def create_cnn(filters=32, kernel_size=3, pool_size=2, dense_units=64):
    model = Sequential()
    model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', input_shape=(sequence_length, input_dim)))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(Flatten())
    model.add(Dense(dense_units, activation='relu'))
    model.add(Dense(1))  # Output layer
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Create a parameter grid for grid search
param_grid = {
    'filters': [16, 32, 64,100],
    'kernel_size': [2,3, 5],
    'pool_size': [1,2, 4],
    'dense_units': [32, 64,100, 128]
}

# Create the CNN model
model = KerasRegressor(build_fn=create_cnn, verbose=0)

# Perform grid search
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3)
grid_result = grid_search.fit(X_train, y_train)

best_params = grid_result.best_params_
print('Best Params==>', best_params)

final_model = create_cnn(filters=best_params['filters'], kernel_size=best_params['kernel_size'], pool_size=best_params['pool_size'], dense_units=best_params['dense_units'])
history = History()
# final_model.fit(X_train, y_train, epochs=20, batch_size=30, verbose=1, callbacks=[history])

# # Evaluate model
# score = final_model.evaluate(X_test, y_test, verbose=2)
# print('Test Loss:', score)

history = final_model.fit(X_train, y_train, epochs=20, batch_size=30, validation_data=(X_test, y_test), callbacks=[history])

# Use the most recent sequence as input for prediction
# last_sequence = data[f'{TimeFrame}_change'].values[-sequence_length:]
# forecast = final_model.predict(last_sequence.reshape(1, sequence_length, 1))

# # Rescale the forecasted values to the original price range
# forecast = forecast * (data[f'{TimeFrame}_change'].max() - data[f'{TimeFrame}_change'].min()) + data[f'{TimeFrame}_change'].min()


# Make predictions
y_pred = final_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("Mean Absolute Error:", mae)


####################find model output
# filename = f'Eth_per_{TimeFrame}.sav'
# pickle.dump(final_model, open(filename, 'wb'))



plt.plot(y_test, label='Actual')
plt.plot(y_pred, label='Predicted', color='orange')
plt.xlabel('Time Steps')
plt.ylabel('Value')
plt.legend()
# plt.savefig(f'Eth_per_{TimeFrame}.png')
plt.show()



# Fit the model and store the loss history

# history = final_model.fit(X_train, y_train, epochs=20, batch_size=30, validation_data=(X_test, y_test), callbacks=[history])
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
# plt.savefig(f'Eth_per_{TimeFrame}_loss.png')
plt.show()

