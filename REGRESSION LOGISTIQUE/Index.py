import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score , precision_score, recall_score, f1_score
#Load data
data = pd.read_csv("social_network_Ads.csv")
print(pd.concat([data.head(), data.tail()]))

#explore data
print(data.info(), "\n")
print(data.columns, "\n")
print("Les valeurs nulles de chaque colonne:\n",data.isnull().any(),"\n")


#Visualisation des données
#sns.countplot(x= 'Purchased', data= data)
#plt.show()

#Cleaning data
data.drop('User ID', axis=1 , inplace=True)
print("Nouveau dataset:\n", data,"\n\n")

#Transformer les donnees categorielles
encoder = LabelEncoder()
data['Gender'] = encoder.fit_transform(data['Gender'])
print("DATASET avec les valeurs de la colonne Gender encodées:\n",data,"\n\n")

#Mise en echelle des  caracteritiques
scaler = StandardScaler()
data[['Age', 'EstimatedSalary']] = scaler.fit_transform(data[['Age', 'EstimatedSalary']])
print("DATASET avec les valeurs de la colonne Age et EstimatedSalary mises en echelle:\n",data,"\n\n")

#Separation du dataset en features et target
features = data.drop(columns=["Purchased"])
targets = data["Purchased"]
#Separation des donnees en train et test:
x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)
print("Les donnes d’entraînement:\n",x_train,"\n",y_train,"\n")

#Entrainement du modele
classifier = LogisticRegression()
classifier.fit(x_train, y_train)

#Evaluation du modele sur les donnees de test
values_predicts = classifier.predict(x_test)
print("Les valeurs predites  :\n",values_predicts,"\n\n")

#Comparaison des valeurs reelles aux valeurs predites
comparaison = pd.DataFrame({
    "Valeurs reelles":y_test,
    "Valeurs predites":values_predicts
})
print("tableau des valeur reelles - predites:\n",comparaison.to_string(),"\n\n")

#Les valeurs de l'equation
print("Les coefficients sont : ",classifier.coef_,"\n")
print("Les intercepts sont : ",classifier.intercept_,"\n")

#Evaluation de la performance du modele

precise = accuracy_score(y_test, values_predicts)
print("Notre modèle est correct environ de :",precise) 

precision = precision_score(y_test, values_predicts)
print("La précision de notre modèle est d’environ:",precision)

recall = recall_score(y_test, values_predicts)
print("Le recall de notre modèle est d’environ:",recall)

f1 = f1_score(y_test, values_predicts)
print("Le f1 de notre modèle est d’environ:",f1)
