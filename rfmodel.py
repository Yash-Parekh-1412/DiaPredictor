import pandas as pd
import numpy as np
import plotly.figure_factory as ff
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pickle
import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv('diabetes.csv')


# print(df.head())

#normalising values
df['BloodPressure'] = df['BloodPressure'].replace(0, df['BloodPressure'].mean())
df['SkinThickness'] = df['SkinThickness'].replace(0, df['SkinThickness'].mean())
df['Insulin'] = df['Insulin'].replace(0, df['Insulin'].mean())
df['BMI'] = df['BMI'].replace(0, df['BMI'].mean())
df['Glucose'] = df['Glucose'].replace(0, df['Glucose'].mean())

# df = np.array(df)

# X AND Y DATA
x = df.iloc[:, :8]
y = df.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)


# RF MODEL
rf  = RandomForestClassifier()
rf.fit(x_train, y_train)
# user_result1 = rf.predict(user_data)


# inputt=[float(x) for x in "2 100 60 20 20 20 1 33".split(' ')]
# final=[np.array(inputt)]

# b = rf.predict(final)


# #DT MODEL
# dt = DecisionTreeClassifier()
# dt.fit(x_train, y_train)
# user_result2 = dt.predict(user_data)

#pickel file of the model
pickle.dump(rf, open("rfmodel.pkl", "wb"))
model=pickle.load(open('rfmodel.pkl','rb'))