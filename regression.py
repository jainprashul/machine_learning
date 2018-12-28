import pandas as pd
import quandl 
import math

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

forecast_out = int(math.ceil(0.01*len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace = True)

print(df.head())