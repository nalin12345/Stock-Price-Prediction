import quandl, math
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import datetime

style.use('ggplot')

df = quandl.get("EOD/MCD", authtoken="mKhms_hKMzyVLTms56Lt", start_date="2013-09-01", end_date="2017-12-28")
df = df[['Adj_Open', 'Adj_High', 'Adj_Low', 'Adj_Close', 'Adj_Volume']]
df['HL_PCT']=(df['Adj_High']-df['Adj_Low'])/df['Adj_Close']*100.0
df['PCT_change']= (df['Adj_Close']-df['Adj_Open'])/df['Adj_Open']*100.0

df = df[['Adj_Close', 'HL_PCT', 'PCT_change', 'Adj_Volume']]
forecast_col = 'Adj_Close'
df.fillna(value=-99999, inplace=True)
forecast_out = int(math.ceil(0.01*len(df)))
df['label']=df[forecast_col].shift(-forecast_out)

X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace=True)

y = np.array(df['label'])

X_train , X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)

forecast_set = clf.predict(X_lately)
df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day
for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix+= 86400
    df.loc[next_date]= [np.nan for _ in range(len(df.columns)-1)]+[i]

df['Adj_Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
