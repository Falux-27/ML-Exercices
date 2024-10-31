import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score , confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import recall_score, f1_score

#Load data
dataset = pd.read_csv("/Users/apple/Desktop/ML_Algorithms/REGRESSION LOGISTIQUE/Invistico_Airline.csv")
print(pd.concat([dataset.head(), dataset.tail()]),"\n\n")
#Cleaning data
dataset["Arrival Delay in Minutes"].fillna(value=dataset["Arrival Delay in Minutes"].mean(),inplace=True)
print(dataset.info(),"\n")

#Separation des donnees en features-targets
features =dataset.drop(columns=["satisfaction"])
target = dataset["satisfaction"]
val_isnull = target.isna().sum()
print("Nombre de valeurs nulles:",val_isnull)
uniq_vals = target.nunique()
print("Les valeurs uniques:",uniq_vals)
print("Valeurs NaN dans les targets:\n", target.isna().sum())

#Mappage de la cible
encoder = LabelEncoder()
target = encoder.fit_transform(target)
#target = target.map({'satisfied':1, 'disatisfied':0})

#Encodage des variables - standardisation
encoder = OneHotEncoder()
columns_to_encode = ['Gender','Customer Type','Type of Travel','Class']
standard = StandardScaler()
columns_to_standardized = ["Age","Flight Distance","Seat comfort","Departure/Arrival time convenient",
                           "Food and drink","Gate location","Inflight wifi service","Inflight entertainment",
                           "Online support","Ease of Online booking","On-board service","Leg room service",
                           "Baggage handling","Checkin service","Cleanliness","Online boarding",
                           "Departure Delay in Minutes","Arrival Delay in Minutes"]

#Combiner les transformations
cooking = ColumnTransformer(transformers=[
    ('encoding', encoder,columns_to_encode),
    ('standardisation', standard, columns_to_standardized)
])
features_cooked = cooking.fit_transform(features)

#Splitting data
x_train , x_test, y_train, y_test = train_test_split(features_cooked,target, test_size=0.2, random_state=42)

#Modele
model = LogisticRegression(max_iter=10000,solver='liblinear',penalty='l1')
#Entrainement du pipeline sur le donnees
model.fit(x_train,y_train)

#L'equation
print("l’intercept  du modele:",model.intercept_)
print("Les coefficients du modele:",model.coef_)

# Prediction sur les donnees de test
values_predict =model.predict(x_test)
print("Les valeurs predites  :\n",values_predict)

#Les metriques du modele 

    #Accuracy
precision = accuracy_score(y_test, values_predict)
print("Accuracy-score:",precision)

    #Recall
rappel = recall_score(y_test, values_predict)
print("Recall-score:",rappel)

    #F1-score
f1_score = f1_score(y_test, values_predict)
print("F1-score:",f1_score)

#Optimisation des hyperparameters 
grille_params = {
    'C':[0.001, 0.01, 0.1, 1, 10, 100,1000,10000]
}
# GridSearchCV 
grid_search = GridSearchCV(estimator=model,param_grid=grille_params,cv=10, scoring='accuracy', verbose=1)
grid_search.fit(x_train,y_train)
print("Meilleure performance (accuracy):", grid_search.best_score_)

# Evaluation du modèle final sur l'ensemble de test 
best_model = grid_search.best_estimator_        #Recuperer le meilleur modele
performance = best_model.score(x_test, y_test)
print(f"Performance du modèle sur les donnees de test (accuracy): {performance:.2f}")

#Matrice de confusion
matrix = confusion_matrix(y_test, values_predict)
fig = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=[0,1])
fig.plot()
plt.show()


