import pandas as pd
import numpy as np
import quandl , math
from sklearn import preprocessing, svm, model_selection
from sklearn.linear_model import LinearRegression

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
x = np.array(df.drop(['label'],1))
y = np.array(df['label'])
X = preprocessing.scale(x) 
#x = x[: -forecast_out+ 1 ]
#df.dropna(inplace=True)
y = np.array(df['label'])

X_train , X_test , y_train , y_test = model_selection.train_test_split(X, y, test_size=0.2)

#classifier
clf = LinearRegression(n_jobs=-1) 
# train
clf.fit(X_train, y_train)
# test
accuracy = clf.score(X_test, y_test)

print(accuracy)
