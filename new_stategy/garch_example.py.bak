#jupyter notebook --no-browser --port 6061 --ip=192.168.56.101
#sudo pip3 --default-timeout=100 install mpl_finance -i https://pypi.tuna.tsinghua.edu.cn/simple

import datetime as dt
import sys

import numpy as np
import pandas as pd

from arch import arch_model
import arch.data.sp500

data = arch.data.sp500.load()
market = data['Adj Close']
returns = 100 * market.pct_change().dropna()



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt






import chart_studio.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go

import warnings
warnings.filterwarnings("ignore")


from mpl_finance import candlestick_ohlc
import matplotlib.dates as mdates



data_new=data
data_new["Date"] = data_new.index
data_new["Date"] =data_new["Date"].apply(mdates.date2num)
ohlc= data_new[['Date','Open','High','Low','Close']].copy()


f1, ax = plt.subplots(figsize = (16,6))
# plot the candlesticks
candlestick_ohlc(ax, ohlc.values, width=.6, colorup='green', colordown='red')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
# Saving image
plt.show()

df=data
df['log_price'] = np.log(df['Close'])

df['pct_change'] = df['log_price'].diff()

df['stdev'] = df['pct_change'].rolling(window=30, center=False).std()
df['Volatility'] = df['stdev'] * (252**0.5) # Annualize.
plt.figure(figsize=(16,6))
df['Volatility'].plot()
plt.title("Rolling Volatility With 30 Time Periods By Annualized Standard Deviation")
plt.show()


#####################################
df = df.dropna()
vol = df["Volatility"] * 100

from arch import arch_model
am = arch_model(vol, vol='Garch', p=1, o=0, q=1, dist='Normal')
res = am.fit(disp='off')
display(res.summary())

df['forecast_vol'] = 0.1 * np.sqrt(res.params['omega'] + res.params['alpha[1]'] * res.resid**2 + 
                                   res.conditional_volatility**2 * res.params['beta[1]'])

display(df.tail(10)) 
plt.figure(figsize=(16,6))
df["Volatility"].plot()
df["forecast_vol"].plot()
plt.title("Real Rolling Volatility vs Forecast by GARCH(1,1)")
plt.legend()
plt.show()
###########################################################

def rmse_tr(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
skor = rmse_tr(df.loc[df.index[1000:], 'forecast_vol'], df.loc[df.index[1000:], 'Volatility'])
print("Root Mean Squared Error of the model is calculated as ",skor)   



df.shape
training_set = df.iloc[:, 10:11].values
# 100 timestep 
X_train = []
y_train = []
for i in range(1000, df.shape[0]):
    X_train.append(training_set[i-1000:i,0])
    y_train.append(training_set[i,0])
X_train, y_train = np.array(X_train), np.array(y_train)  

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 10, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.1))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 10, return_sequences = True))
regressor.add(Dropout(0.1))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 10, return_sequences = True))
regressor.add(Dropout(0.1))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 10))
regressor.add(Dropout(0.1))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)

regressor.save('my_modelp1.h5')

from keras.models import load_model
regressor = load_model('my_modelp1.h5')

predicted_stock_price = regressor.predict(X_train)

# Visualising the results
plt.figure(figsize=(18,6))
plt.plot(df.iloc[1000:, 11:12].values, color = 'red', label = 'Observed Volatility')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Volatility By LSTM')
plt.title('Real Rolling Volatility vs Forecast of LSTM')
plt.xlabel('Time')
plt.ylabel('Volatility')
plt.legend()
plt.show()

skor2 = rmse_tr(predicted_stock_price, np.array(df.loc[df.index[1000:], 'Volatility']))
print("Root Mean Squared Error of the model is calculated as ",skor2)

###Neural-Garch Model (Combining Garch(1,1) and LSTM)

training_set = df.iloc[:, 10:11].values
# 100 timestep ve 1 çıktı ile data yapısı oluşturalım
X_train = []
y_train = []
for i in range(1000, df.shape[0]):
    X_train.append(training_set[i-1000:i,:])
    y_train.append(training_set[i,0])
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 2))

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 10, return_sequences = True, input_shape = (X_train.shape[1], 2)))
regressor.add(Dropout(0.1))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 10, return_sequences = True))
regressor.add(Dropout(0.1))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 10, return_sequences = True))
regressor.add(Dropout(0.1))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 10))
regressor.add(Dropout(0.1))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)


regressor.save('my_modelp2.h5')

from keras.models import load_model
regressor = load_model('my_modelp2.h5')


predicted_stock_price = regressor.predict(X_train)

# Visualising the results
plt.figure(figsize=(18,6))
plt.plot(df.iloc[1000:, 11:12].values, color = 'red', label = 'Observed Volatility')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Volatility By LSTM-GARCH(1,1)')
plt.title('Real Rolling Volatility vs Forecast of LSTM-GARCH(1,1)')
plt.xlabel('Time')
plt.ylabel('Volatility')
plt.legend()
plt.show()
                             