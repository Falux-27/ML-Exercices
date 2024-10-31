import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score , confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import recall_score, f1_score

#Load data
dataset = pd.read_csv("/Users/apple/Desktop/ML_Algorithms/REGRESSION LOGISTIQUE/weatherAUS.csv")

#About the data

  #transformation de la colonne Date
dataset['Date'] = pd.to_datetime(dataset['Date'])
dataset['year'] = dataset.Date.dt.year
dataset['months']= dataset.Date.dt.month
dataset['day'] = dataset.Date.dt.day
print(pd.concat([dataset.head(), dataset.tail()]),"\n\n")

    #Valeurs unique de chaque colonne
for uniq in dataset.columns:
    print(uniq,' -->',dataset[uniq].nunique(),"\n\n")
        #-->CLEANING DATA
print(dataset.info())

 #Valeurs null des colonnes
k = dataset.isnull().sum()
print("Valeurs null des colonnes:\n",k)

#Collection de tous les colonnes numériques
numerical_cols = [num  for num in dataset.columns if dataset[num].dtype == "float64"]
print("Les colonnes numériques:\n",numerical_cols,"\n")

#Collection de toutes les colonnes catégorielles
cat_columns = [cat for cat in dataset.columns if dataset[cat].dtype == 'object']
print("Les colonnes categorielles:\n",cat_columns,"\n")
 
#Visualisation de la distribution des données
#Comparaison des vitesse de vent en fonction des prediction de pluies
sns.relplot(data=dataset, x='WindGustDir',y='WindGustSpeed',hue='RainTomorrow')
plt.title("Direction du vent - Vitesse des rafales de vent")
plt.show()

#Comparaison des direction du vent entre 9H et 3H
sns.lineplot(data=dataset, x='WindDir9am',y='WindDir3pm',hue='year')
plt.title("Direction du vent entre 9H et 3H du matin")
plt.show()

#Comparaison du nombre de yes-no dans la colonne RainTomorrow
sns.set_style('darkgrid')
sns.countplot(data=dataset,x ='RainTomorrow',hue='RainTomorrow')
plt.title("Nombre de jours avec et sans pluie demain")
plt.show()

#Comparaison du niveau de pluie chaque année
sns.set_style('whitegrid')
sns.barplot(data=dataset,x='year' ,y = 'Rainfall',errorbar=None)
plt.title("Niveau de pluie par année")
plt.xticks(rotation=90)
plt.show()

#Visualisation des pluies dans chaque region 
sns.set_style('ticks')
sns.barplot(data=dataset, x='Location',y='Rainfall',hue='year')
plt.title("Pluie dans chaque région par année")
plt.xticks(rotation=90)
plt.show()

#Remplissages des valeurs manquante

  #Pour les colonnes numériques
for x in numerical_cols:
    dataset[x].fillna(dataset[x].mean(), inplace=True)
    
  #Pour les colonnes catégorielles
for z in cat_columns:
    dataset[z].fillna(dataset[z].mode()[0], inplace=True)
    #Encodage
encoder = LabelEncoder()
for col in cat_columns:
    dataset[col]= encoder.fit_transform(dataset[col])
print("Le dataset après remplissage:\n",dataset.isnull().sum())

#Suppression de la colonne date
dataset.drop(columns=['year','months','day'],axis=1,inplace=True)
dataset.drop(['Date'],axis=1, inplace=True)

#Separation des données en features target
features=dataset.drop(columns=["RainTomorrow"])
target = dataset["RainTomorrow"]

#Mise en échelle des colonnes numériques
scaler = MinMaxScaler()
features[numerical_cols] = scaler.fit_transform(features[numerical_cols])
print("Colonnes features:\n",features.columns)

#Splitting data
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

#Algorithm
model = LogisticRegression(max_iter=10000,solver='liblinear',penalty='l1',verbose=1)

#Entrainement du modele
model.fit(x_train, y_train)

#L'équation du modele
print("l’intercept  du modele:",model.intercept_)
print("Les coefficients du modele:",model.coef_)

#Prediction sur les données de test
values_predict = model.predict(x_test)
print("Les valeurs predites  :\n",values_predict)

#Comparaison des valeurs reelles aux valeurs predites
tab_ = pd.DataFrame({
    'valeurs reelles':y_test,
    'valeurs predites':values_predict
})
print("tableau des valeur reelles - predites:\n",tab_.head(20))

#Evaluation de la performance du modele

    #Accuracy
precision = accuracy_score(y_test, values_predict)
print("Accuracy-score:",precision)

    #Precision
rappell = recall_score(y_test, values_predict)
print("Recall-score:",rappell)

 #F1-score
f1_score = f1_score(y_test, values_predict)
print("F1-score:",f1_score)

#Matrice de confusion
matrix =confusion_matrix(y_test,values_predict)
ax= ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=[0,1])
ax.plot()

plt.show()

