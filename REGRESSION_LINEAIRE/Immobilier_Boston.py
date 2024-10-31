import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import seaborn as sns
from sklearn.metrics import r2_score , mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge , Lasso

 # Chargement des données
dataset = pd.read_csv("boston.csv")
print(dataset.head(),"\n\n")
print(dataset.info(),"\n\n")
print(dataset.describe(),"\n\n")

#Separation des donnees
features = dataset.drop(columns=['MEDV'])
target = dataset['MEDV']


#Visualisation
#sns.pairplot(dataset)
#plt.show()

#Separation des donnees en train et test:
xtrain,xtest, ytrain,ytest = train_test_split(features, target, test_size=0.3, random_state=42)
print("les donnes d’entraînement:\n",xtrain,"\n\n")

#Entraînement des donnees:
regressor=LinearRegression()
regressor.fit(xtrain,ytrain) #Creation de l'equation

#Evaluation du modele sur les donnees de test
y_predict = regressor.predict(xtest)
R_carrée = r2_score(ytest, y_predict)
mse = mean_squared_error(ytest, y_predict)
print("Les predictions de notre modele:\n",y_predict)
print("Le score R-Carrée est:",R_carrée,"\n")
print("Le score MSE est:",mse,"\n\n")
 
#la droite d'equation 
print("L’ordonnée est :",regressor.intercept_,"\n")
#Tableau des coefficients
tab_coef = pd.DataFrame(regressor.coef_,features.columns, columns=["Coefficients"])
print("Les coef sont:",tab_coef)

 #Représentation graphique:
plt.scatter(ytest, y_predict)
plt.plot([ytest.min(), ytest.max()],[ytest.min(), ytest.max()], c = "g", linestyle="--")
plt.xlabel("Valeurs réelles")
plt.ylabel("Valeurs prédites")
plt.title("Comparaison des valeurs réelles et prédites")
plt.show()


#Regression Ridge
Ridge_model = Ridge(alpha=1.0)

#Entraînement des donnees
Ridge_model.fit(xtrain, ytrain)
print("l’ordonnée est :", Ridge_model.intercept_)
print("Les coef de regularisation ajustée:",Ridge_model.coef_,"\n\n")

#Prediction avec Ridge sur les donnees de test
z_predict = Ridge_model.predict(xtest)

#Evaluation du modele
R_carrée = r2_score(ytest, z_predict)
mse = mean_squared_error(ytest, y_predict)
print("Le score R-Carrée est:",R_carrée,"\n")
print("Le score MSE est:",mse,"\n\n")

#Représentation graphique:
plt.scatter(ytest, z_predict)
plt.plot([ytest.min(), ytest.max()],[ytest.min(), ytest.max()], c = "g", linestyle="--")
plt.xlabel("Valeurs réelles")
plt.ylabel("Valeurs prédites")
plt.title("Comparaison des valeurs réelles et prédites")
plt.show()
