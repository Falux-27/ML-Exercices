import numpy as np
import pandas as pd
from sklearn.utils import resample
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

            #From Scratch
#Definition de la classe Bagging
class bagging:
    def __init__(self, base_model,n_estimator):
        self.base_model = base_model
        self.estimator = n_estimator
        self.models = []
        
#Entrainement des models
def train(self,x,y):
    for _ in range(self.estimator):
        x_resampled, y_resampled = resample(x, y, replace=True) #creation de sous ensemble pour chaque modele
        model = self.base_model()   #Instanciation du modele à utiliser
        model.fit(x_resampled, y_resampled)   #Entrainement du modele sur les sous ensembles
        self.models.append(model)  #Ajout des modèles entrainées à la liste

#Prediction avec Bagging
def predict(self, x):
    predictions = [model.predict(x) for model in self.models]
    return np.mean(predictions, axis=0) #Moyenne des predictions des modèles pour obtenir une prédiction finale

#Entrainer et evaluer avec un model precise
base_model = DecisionTreeClassifier
n_estimator = 10
bagging_model = bagging(base_model, n_estimator)

            #With Scikit-Learn
            
#Chargement des données
digits = load_digits()
x, y = digits.data, digits.target

#Diviser les données en ensemble d'entrainement et de test
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2,random_state=42)

#Entrainement
model = DecisionTreeClassifier()
n_estimator = 10
bagging_model = BaggingClassifier(model,n_estimators=n_estimator)
classifier =model.fit(x_train,y_train)
#Prediction
class_predicted = classifier.predict(x_test)
accuracy = accuracy_score(y_test, class_predicted)
print("Accuracy:", accuracy)



 


        