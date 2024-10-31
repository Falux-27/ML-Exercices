import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error ,r2_score

 # Chargement des données
from sklearn.datasets import load_iris
base= load_iris()
print(base.keys())

#Description du dataset iris
print(base.DESCR)

#Transformation en Dataframe
dataset = pd.DataFrame(base.data, columns=base.feature_names)
print(dataset.head())
minval = 5
maxval = 15
#Ajout de colonne
dataset['petal age(mth)'] =np.random.randint(minval, maxval +1 , size=len(dataset))
print(dataset.head())

#Visualisation
fig = plt.figure()
axe = fig.add_subplot(projection = '3d')
axe.scatter(dataset["sepal length (cm)"],dataset["sepal width (cm)"],dataset['petal age(mth)'])
plt.show()

#Visualisation


#Séparer les données en features et target
features = dataset.drop(columns=['petal age(mth)'])
target = dataset['petal age(mth)']

#Division des valeurs
xtrain, xtest, ytrain, ytest = train_test_split(features, target, test_size=0.2, random_state=42)
print(xtrain,"\n\n")

#Creation de l'equation:
regressor = LinearRegression ()
regressor.fit(xtrain,ytrain)

#Evaluation du modele sur les donnees de test
y_predict = regressor.predict(xtest)
R_carrée = r2_score(ytest, y_predict)
print("Le score R-Carrée est:",R_carrée)


