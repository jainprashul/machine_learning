import pandas as pd
import numpy as np
import quandl , math , datetime
from sklearn import preprocessing, svm, model_selection
from sklearn.linear_model import LinearRegression
from matplotlib import style
import matplotlib.pyplot as plt

style.use('ggplot')

df = quandl.get('WIKI/GOOGL')

#select the important features
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume',]] 

#calc highlow percent and percent change
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low'] ) / df['Adj. Close'] * 100.0 
df['PCT_Change'] = (df['Adj. Close'] - df['Adj. Open'] ) / df['Adj. Open'] * 100.0 

#display imp features
df = df[['Adj. Close', 'HL_PCT', 'PCT_Change', 'Adj. Volume']]

forecast_col = 'Adj. Close'
# if data has nill value replace it by 9999
df.fillna(-99999, inplace=True)

# create predicted value as label
forecast_out = int(math.ceil(0.01*len(df)))
print(forecast_out)

df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace = True)

# defining datasets
X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X) 
x = X[: -forecast_out ]
X_lately = X[-forecast_out:]

df.dropna(inplace=True)
y = np.array(df['label'])
y = np.array(df['label'])


X_train , X_test , y_train , y_test = model_selection.train_test_split(X, y, test_size=0.2)

#classifier
clf = LinearRegression(n_jobs=-1) 
# train
clf.fit(X_train, y_train)
# test
accuracy = clf.score(X_test, y_test)

# for k in ['linear','poly','rbf','sigmoid']:
#     clf = svm.SVR(kernel=k)
#     clf.fit(X_train, y_train)
#     confidence = clf.score(X_test, y_test)
#     print(k,confidence)

forecast_set = clf.predict(X_lately)
print(forecast_set, accuracy, forecast_out)


# ploting the graph
df['Forecast'] = np.nan
# this is for date setting
last_date = df.iloc[-1].name 
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
