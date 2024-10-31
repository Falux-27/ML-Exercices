import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

#Chargement des données:

from sklearn.datasets import load_iris  #chargement du Dataframe load_iris
iris = load_iris()
x = iris.data
y = iris.target 
features_names= iris.feature_names
target_names = iris.target_names

print("Les variables indépendantes sont :\n", features_names,"\n")
print("Les variables cibles sont:\n", target_names,"\n")
print("Les valeurs des variables indépendantes:\n",x[0:10],"\n")
print("Les valeurs des variables cibles: \n",y[0:10])


#Fractionnement du jeu de données:

from sklearn.model_selection import train_test_split  #Module de fractionnement du dataset
x_train,  x_test,y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=1)
print(x_train.shape)
print(x_test.shape)

print(y_train.shape)
print(y_test.shape)

"""plt.scatter(x_train,x_train)
plt.title("Données d’entraînement")
plt.xlabel("Features")
plt.ylabel("Targets")
plt.show()
############################
plt.scatter(y_test,y_test)
plt.title("Données de test")
plt.xlabel("Features")
plt.ylabel("Targets")
plt.show()"""

#Entraînement de modèle linéaire:
 
from sklearn.linear_model import LinearRegression 
model = LinearRegression ()  #Création du modèle de régression linéaire
model.fit(x_train, y_train)  #Entraînement du modèle 
 
print("Coefficients du modèle:\n",model.coef_,"\n")
print("Ordonnée à l'origine (intercept):",model.intercept_)

#Sauvegarde du modèle:
import joblib
joblib.dump(model, "regression_lineaire.pkl")