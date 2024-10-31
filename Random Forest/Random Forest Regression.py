import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
warnings.filterwarnings('ignore')

# Load dataset
dataset = pd.read_csv('/Users/apple/Desktop/ML_Algorithms/REGRESSION_LINEAIRE/kc_house_data.csv')
print(dataset.head(), "\n\n")

#Exploring data
print(dataset.info(),"\n")
print(dataset.isnull().sum(),"\n")
print(dataset.describe(),"\n")

#Data preprocessing
dataset.drop(columns=['id','date'],inplace=True)
print(dataset.columns,"\n\n")

#Feature engineering
features = dataset.iloc[:,1:]
target = dataset.iloc[:,0]
scaler = StandardScaler()
features= scaler.fit_transform(features)
print(features,"\n\n")

#Splitting the dataset
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

#Random Forest Regressor
model = RandomForestRegressor(n_estimators=10,oob_score=True,random_state=42)
model.fit(x_train,y_train)
#Evaluation des models avec OOB
score_OOB = model.oob_score_
print("Score Out Of the Bag:", score_OOB,"\n")
#Prediction 
y_pred = model.predict(x_test)
#Evaluation du modèle
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error (MSE):", mse.round(3),"\n")
print(f"R-squared (R2): {r2:.2f}")

 
