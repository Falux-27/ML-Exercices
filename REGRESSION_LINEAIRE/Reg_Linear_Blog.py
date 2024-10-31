import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import max_error, mean_absolute_error, mean_squared_error, r2_score

 # Chargement des données
data = pd.DataFrame({
    'superficie': [120, 85, 100, 150, 95, 130, 100, 75, 110, 90],
    'chambres': [3, 2, 2, 4, 2, 3, 2, 1, 3, 2],
    'transport': [1, 0, 1, 1, 0, 1, 0, 1, 1, 0],
    'prix': [300000, 200000, 250000, 400000, 220000, 350000, 240000, 180000, 280000, 210000]
})

# Séparer les données en features et target
X_features = data.drop(columns=["prix"])
Y_target = data['prix']

# Division des données en train et test
test_size =0.2
x_train, x_test, y_train, y_test = train_test_split(X_features, Y_target, test_size=test_size, random_state=42)

# Entraînement du modèle
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#L'equation:
print("Le coef:",regressor.coef_)
print("L'ordonnée :",regressor.intercept_)

# Faire les prédictions sur les données de test
y_predict = regressor.predict(x_test)
print("Les prédictions :\n", y_predict)

# Évaluer le modèle sur des données fictives:
predicts = regressor.predict([[230,3,1]])
print("Les prédictions sur des données fictives :\n", predicts)

# Évaluation du modèle:
print("Erreur absolue moyenne :", mean_absolute_error(y_test, y_predict))
print("Erreur quadratique moyenne :", mean_squared_error(y_test, y_predict))
print("Erreur maximale :", max_error(y_test, y_predict))
print("Coefficient de détermination :", r2_score(y_test, y_predict))

# Visualisation des résultats
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_predict, color = "blue",label = "Valeurs predites")
#ligne de regression
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],'k--', lw=2, label='Ligne y=x')
plt.xlabel('Valeurs réelles')
plt.ylabel('Valeurs prédites')
plt.title('Comparaison des valeurs réelles et prédites')
plt.legend()
plt.show()
