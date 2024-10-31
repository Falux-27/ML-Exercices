import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score , confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import recall_score, f1_score

#Load data
dataset = pd.read_csv("/Users/apple/Desktop/ML_Algorithms/REGRESSION LOGISTIQUE/bank-additional-full.csv", delimiter=";")
print(pd.concat([dataset.head(), dataset.tail()]),"\n\n")
print(dataset.info(),"\n")
print("Les colonnes du datasets :",dataset.columns,"\n\n")

#Cleaning data
dataset.dropna(inplace=True)

#Separation des donnees en features-targets
features = dataset.drop(columns=["y"])
target  = dataset["y"]

#Mappage de la cible
target = target.map({'no':0, 'yes':1})

#Encodage des variables - standardisation
encoder = OneHotEncoder()
columns_to_encode = ["job","marital","education","month","default",
                     "housing","loan","contact","poutcome"]

scaler = StandardScaler()
columns_to_normalized = ["duration","pdays","nr.employed","age"]

min_max_scaler =preprocessing.MinMaxScaler(feature_range=(1,1.5))
columns_min_max = ["euribor3m"]

#Combiner les transformation
transforming = ColumnTransformer(
    transformers=[
        ('encodage',encoder,columns_to_encode),
        ('normalisation',scaler, columns_to_normalized),
        ("mise en échelle",min_max_scaler,columns_min_max)
    ]
)
features_transformed = transforming.fit_transform(features)

#Splitting data
x_train, x_test, y_train, y_test = train_test_split(features_transformed, target, test_size=0.2, random_state=42)

#Selection des features 
pipeliner = Pipeline(steps=[
    ('selection',SelectFromModel(LogisticRegression(max_iter=10000, solver='liblinear',penalty='l1'))), #Sélectionner les variables les plus significatives
    ('model',LogisticRegression(max_iter=10000,solver='liblinear',penalty='l1'))  #Construire le modele avec ces variables
])
#Entraînement des données avec le pipeline
pipeliner.fit(x_train,y_train)

#Prediction sur les donnees de test
values_predicted = pipeliner.predict(x_test)
print("Les predictions :",values_predicted)

#Evaluation du pipeline
perform = pipeliner.score(x_test,y_test)
print(f"La precision du modele sur les donnees de test est d'environ: {perform:.2f}","\n")

#Matrice de confusion
matrix = confusion_matrix(y_test,values_predicted)
#Graphique
fig = ConfusionMatrixDisplay(confusion_matrix=matrix,display_labels=[0,1])
fig.plot()
plt.show()

#Evaluation des performances
    #Accuracy
precision = accuracy_score(y_test, values_predicted)
print("La precision globale de notre modèle est d'environ:",precision)
    #Recall
rappel = recall_score(y_test, values_predicted)
print("Le rappel de notre modèle est d'environ:",rappel)

    #F1-score
f1_score = f1_score(y_test, values_predicted)
print("Le F1-score de notre modèle est d'environ:",f1_score)
 
