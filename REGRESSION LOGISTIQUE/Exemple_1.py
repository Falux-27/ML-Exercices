import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score ,precision_score , f1_score, recall_score

dataset = pd.read_csv("diabetes2.csv")
print(pd.concat([dataset.head(), dataset.tail()]))

#explore data
print(dataset.info(), "\n")
values_missing = dataset.value_counts(dropna=False)
print("Les valeurs nulles de chaque colonne:\n",dataset.isnull().any(),"\n")

#Visualisation des données
#sns.pairplot( dataset)
#plt.show()


#Separation du dataset en features et target
features = dataset.drop(columns=["Outcome"])
target = dataset["Outcome"]

#Separation des donnees en train et test:  
x_train , x_test, y_train , y_test = train_test_split(features, target, test_size=0.2, random_state=42)
print("Les donnes d’entraînement:\n",x_train,"\n\n",y_train,"\n")

#Entrainement du modele
classifier = LogisticRegression()
classifier.fit(x_train,y_train)

#L'equation de du modele
print("l’ordonnée à l'origine du modele:",classifier.intercept_)
print("Les coefficients du modele:", classifier.coef_)

#Evaluation du modele sur les donnees de test
values_predict = classifier.predict(x_test)
print("Les valeurs predites  :\n",values_predict,"\n\n")

#Comparaison des valeurs reelles aux valeurs predites
tab_comparative = pd.DataFrame({
    "Valeurs reelles":y_test,
    "Valeurs predites":values_predict
})
print("tableau des valeur reelles - predites:\n",tab_comparative.head(20),"\n\n")

#Evaluation de la performance du modele

precise = accuracy_score(y_test, values_predict)
print("La précision globale de notre modèle est d’environ:",precise) 

precision = precision_score(y_test, values_predict)
print("La précision de notre modèle est d’environ:",precision)


recall = recall_score(y_test, values_predict)
print("Le recall de notre modèle est d’environ:",recall)

f1 = f1_score(y_test, values_predict)
print("Le f1 de notre modèle est d’environ:",f1)
