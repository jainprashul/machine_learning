import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn import preprocessing , model_selection
from sklearn.cluster import KMeans
import pandas as pd


'''
Pclass Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
survival Survival (0 = No; 1 = Yes)
name Name
sex Sex
age Age
sibsp Number of Siblings/Spouses Aboard
parch Number of Parents/Children Aboard
ticket Ticket Number
fare Passenger Fare (British pound)
cabin Cabin
embarked Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
boat Lifeboat
body Body Identification Number
home.dest Home/Destination
'''

df = pd.read_excel('titanic.xls')

df.drop(['body', 'name'], 1 , inplace=True)
df.fillna(0, inplace=True)

#print(df.head())

def handle_non_numeric_data(df):
    columns = df.columns.values

    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]
        
        if df[column].dtype != np.int64 and df[column].dtype != np.float64 :
            col_contents = df[column].values.tolist()
            unique_elements = set(col_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+= 1

            df[column] = list(map(convert_to_int, df[column]))
        #print(text_digit_vals)
    return df

df = handle_non_numeric_data(df)
#print(df.head())

X = np.array(df.drop(['survived'], 1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['survived'])

for i in range(10):
    clf = KMeans(n_clusters=2)
    clf.fit(X)

    correct = 0

    for i in range(len(X)):
        predictme = np.array(X[i].astype(float))
        predictme = predictme.reshape(-1, len(predictme))
        prediction = clf.predict(predictme)
        if prediction[0] == y[i]:
            correct += 1

    print(correct/ len(X))