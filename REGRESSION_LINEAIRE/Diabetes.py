import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import seaborn as sns
from mpl_toolkits import mplot3d
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

 # Chargement des données
from sklearn.datasets import load_diabetes
diseases = load_diabetes()
print(diseases.DESCR)


#Transformation en Dataframe
dataset = pd.DataFrame(diseases.data, columns=diseases.feature_names)
print(dataset.head())

#Ajout d'un colonne de progression
val_min = 50.1
val_max = 250.0
progression_values = np.random.uniform(val_min, val_max, size=len(dataset))
# Arrondir les valeurs à deux décimales
dataset["progression"] = np.round(progression_values, 2)
print(dataset.head())

#Visualisation
#sns.pairplot(dataset)
#plt.show()

#Separation des donnees en features et target:
features = dataset.drop(columns=['progression'])
target = dataset['progression']

#Separation des donnees en train et test:
x_train , x_test,y_train, y_test = train_test_split(features,target, test_size= 0.2, random_state=42)
print(x_train,"\n\n") 

#Creation de l'equation:
regressor = LinearRegression()
regressor.fit(x_train,y_train)

#Evaluation du modele sur les donnees de test
y_predict = regressor.predict(x_test)
R_carrée = r2_score(y_test, y_predict)
print("Le score R-Carrée est:",R_carrée)





