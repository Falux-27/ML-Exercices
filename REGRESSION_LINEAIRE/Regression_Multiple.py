import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns 
#Creation du dataframe:
data_houses = pd.DataFrame({
    'Number of Bedrooms': [3, 2, 4, 3, 5, 2, 3, 4, 4, 3],
    'Size': [1500, 1200, 2000, 1800, 2500, 1100, 1700, 1900, 2100, 1600],
    'Age of House': [10, 15, 5, 20, 8, 30, 12, 6, 7, 25],
    'Price': [300000, 250000, 400000, 350000, 500000, 230000, 330000, 420000, 450000, 310000]
})

#Separation des donnees en futures et target
features = data_houses[['Number of Bedrooms','Size','Age of House']]
target = data_houses['Price']

 #Separation des donnees en train et test:
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=101)
print(x_train,"\n\n")
print(y_train,"\n\n")

#entraînement des donnees
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#Creation de l'equation:
regressor.fit(x_train, y_train)

print("L’ordonnée est :",regressor.intercept_,"\n\n")
print("Les coefficients sont : ",regressor.coef_,"\n")

#Evaluation du model sur les donnees de test:
y_predict = regressor.predict(x_test)
print("Les valeurs predites  :\n\n",y_predict,"\n\n")

#Tableau des valeurs predict:
tab_coef = pd.DataFrame(regressor.coef_,features.columns, columns=["Coefficients"])
print(tab_coef)

#Représentation graphique des valeurs:
print('Prix réel:',x_test,"\n\n")
print('Prix prédit:',y_predict)

fig = plt.figure()
axe= fig.add_subplot(111, projection='3d')
axe.scatter(y_test, y_predict)
plt.xlabel('Prix réel')
plt.ylabel('Prix prédit')
plt.title('Comparaison des prix réels vs prédits')

plt.show()
 

 