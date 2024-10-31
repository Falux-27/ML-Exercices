import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

data = pd.read_csv("/Users/apple/Desktop/ML_Algorithms/REGRESSION_LINEAIRE/Salary_Data.csv")
print("Dataframe:\n",data.head())
print(data.info())

#Selection des colonnes:

x = data[['YearsExperience']]
y = data['Salary']

#Graphiques:

plt.scatter(x, y, marker='o', s=25, c='m')

#Separation des donnees :

from sklearn.model_selection import train_test_split
x_train, x_test, y_train , y_test = train_test_split(x, y, test_size=1/3, random_state=0)

#Importation de la librairie LinearRegression:

from sklearn.linear_model import LinearRegression
regresseur = LinearRegression ()  #Instanciation

#Entraînement des donnees: (Produire l'equation linéaire)
regresseur.fit(x_train,y_train)

#Afficher lees coef de l'equation:
print("Les coef du modèle sont:",regresseur.coef_)
print("L’ordonnée a l'origine:", regresseur.intercept_)
 
#Graphique:

plt.plot(x,regresseur.predict(x), label='Régression linéaire')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Régression Linéaire: Expérience vs Salaire')
plt.legend()
plt.show()


####Evaluation du modèle:

#prediction sur la base des test:
y_predict = regresseur.predict(x_test) 

# Création d'un DataFrame pour stocker les prédictions
predictions_df = pd.DataFrame({
    'YearsExperience': x_test.values.flatten(), 
    'Salary': y_test.values, 
    'Predicted_Salary': y_predict
})

print(predictions_df)
#la métrique:
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test,y_predict))
print('MSE:', metrics.mean_squared_error(y_test,y_predict))
print('R-carre:',metrics.r2_score(x_test, y_predict))