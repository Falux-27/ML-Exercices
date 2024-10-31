import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

#Chargement des données
dataset = pd.read_csv('/Users/apple/Desktop/ML_Algorithms/REGRESSION_LINEAIRE/study_performance.csv')
print(dataset.head(), "\n\n")

#Exploring data
print(dataset.info())

#Handling missing values
val_uniq = dataset.nunique()
print(val_uniq)

#Changing columns names
dataset = dataset.rename(columns={
    "parental_level_of_education":'parents education level',
    "test_preparation_course":'test preparation'})

#Changing some values
dataset['parents education level']=dataset['parents education level'].replace('some high school', 'high school')
dataset['parents education level']=dataset['parents education level'].replace('some college','college')

#Uniq values in each columns
val_uniq = dataset['parents education level'].unique().tolist()
print(val_uniq,"\n")
print(pd.concat([dataset.head(60), dataset.tail(17)]),"\n")

#Calculating mean score
dataset['average_score'] =(dataset['math_score'] + dataset['reading_score'] + dataset['writing_score']) / 3
print(dataset['average_score'],"\n")
#Rounding score
dataset['average_score']= dataset['average_score'].round(1)
print(dataset['average_score'],"\n")

#Encoding 
encoder = LabelEncoder()
t= [x for x in dataset.columns if dataset[x].dtype== 'object']
for col in t:
    dataset[col]= encoder.fit_transform(dataset[col])
print(dataset.head(),"\n")

#Standardisation
scaler = MinMaxScaler(feature_range=(0,5))
columns_to_scale = ['math_score', 'reading_score', 'writing_score', 'average_score']
dataset[columns_to_scale]= scaler.fit_transform(dataset[columns_to_scale])
print(dataset.head(),"\n\n")

#----->Prediction des scores de math

#Splitting data
features =dataset.drop(columns=['math_score'])
target = dataset['math_score']
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

#Model
model =LinearRegression()
#Training model
model.fit(x_train, y_train)

#Equation 
print("L'ordonnée à l'origine:", model.intercept_)
print("Les coefficients du modèle:", model.coef_, "\n")

#Prediction
values_predicted = model.predict(x_test)
print(values_predicted,"\n")


#Evaluation
mse = mean_squared_error(y_test, values_predicted)
r2 = r2_score(y_test, values_predicted)
mae = mean_absolute_error(y_test, values_predicted)

print("Mean Squared Error (MSE):", mse)
print("R-squared (R2):", r2)
print("Mean Absolute Error (MAE):", mae)
print('n')

#--->Prediction des score de Writting

#Splitting data
X_features =dataset.drop(columns=['writing_score'])
Y_target = dataset['writing_score']

x_train, x_test, y_train, y_test = train_test_split(X_features,Y_target, test_size=0.2, random_state=42)

#Model
model_2 = LinearRegression()
#Training model
model_2.fit(X_features, Y_target)

#Equation
print("\nL'ordonnée à l'origine:", model_2.intercept_)
print("Les coefficients du modèle:", model_2.coef_, "\n")

#Prediction
score_predicted = model_2.predict(x_test)

#Evaluation
mse = mean_squared_error(y_test, score_predicted)
r2 = r2_score(y_test, score_predicted)
mae = mean_absolute_error(y_test, score_predicted)

print("Mean Squared Error (MSE):", mse)
print("R-squared (R2):", r2)
print("Mean Absolute Error (MAE):", mae)

#Visualisation 
plt.scatter(y_test, values_predicted)
plt.plot([y_test.min(), y_test.max()],[y_test.min(), y_test.max()], c = 'red')
plt.xlabel("Scores réelles")
plt.ylabel("Scores prédits")
plt.title("Comparaison des scores Maths réels et prédits")
plt.show()
####################################
plt.scatter(y_test, score_predicted)
plt.plot(y_test,score_predicted, c ='red')
plt.xlabel("Scores réelles")
plt.ylabel("Scores prédits")
plt.title("Comparaison des scores de Writing réels et prédits")
plt.show()

