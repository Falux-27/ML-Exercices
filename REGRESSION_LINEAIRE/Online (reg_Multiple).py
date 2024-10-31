import random
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from mpl_toolkits import mplot3d
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


#Load dataset
dataset =pd.read_csv("online.csv")
#Remplacement des valeurs de Area:
dataset['Area'] = dataset['Area'].replace({'Dhaka': 'Ville', 'Ctg': 'banlieu', 'Rangpur': 'Ville'})
dataset.at[49,'Marketing Spend'] =93863.75
dataset.at[19,'Transport'] = 453173.0
dataset.at[48,'Transport'] = 353183.81
print(dataset.head(),"\n\n")

#Visualisation
fig = plt.figure()
axe = fig.add_subplot()
axe.scatter(dataset["Marketing Spend"],dataset["Profit"])
plt.show()


#Encodage des donnees non numériques:
dataset = pd.get_dummies(dataset, columns=['Area'])
print(dataset.head())

#Splitting data :
features = dataset.iloc[:,:-1]
targets = dataset.iloc[:,-1]

#Division des valeurs en test et train:
x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=101)

#Creation de l'equation:
regresseur = LinearRegression()
regresseur.fit(x_train,y_train)

#Faire les predictions sur les donnees de Test:
y_predict = regresseur.predict(x_test)
print("\n\nLes predictions :\n\n",y_predict)

#test sur des valeurs fictives:


        