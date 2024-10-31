import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,cross_val_score, StratifiedGroupKFold
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV

# Load dataset
dataset= pd.read_csv("/Users/apple/Desktop/ML_Algorithms/SVM/Cancer_Data.csv")

#Exploring data
print(dataset.shape)
print(dataset.info())
print(dataset.isnull().sum())

#Data preprocessing
dataset = dataset.drop(columns=['Unnamed: 32', 'id'])
print(dataset.columns)
val_uniq = dataset['diagnosis'].value_counts()
print("Les valeurs uniques:",val_uniq)
percent_val_uniq = dataset['diagnosis'].value_counts(normalize=True)
print("Pourcentage de chaque catégorie:",percent_val_uniq.round(2))

#Mapping the target
dataset['diagnosis'] =dataset['diagnosis'].map({'M':0 , 'B':1})
print(pd.concat([dataset.head(), dataset.tail()]))
print(dataset['diagnosis'].nunique())

#Visualisation 
sns.set_style('darkgrid')
sns.countplot(data=dataset,x=dataset['diagnosis'],hue='diagnosis')
plt.title("Répartition des données par diagnostique")
plt.show()

#Splitting in features targets
features = dataset.drop(columns='diagnosis')
target = dataset['diagnosis']

#Scaling
columns_scaling =[]
for col in features:
    if  features[col].dtype == 'float64':
        columns_scaling.append(col)
        
scaler = MinMaxScaler(feature_range=(1,2))
features[columns_scaling] = scaler.fit_transform(features[columns_scaling])

#Splitting data
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


# GridSearchCv
model =SVC()
C = [0.001, 0.01, 0.1, 1,10,100, 1000]
kernel  =['linear', 'poly', 'rbf']
gamma = [0.001,0.01, 0.1, 1, 100, 1000]
param_grid  ={'C':C, 'kernel':kernel, 'gamma':gamma}
grid = GridSearchCV(model, param_grid, cv=10) 
grid.fit(x_train, y_train)

print("Meilleur score : ", grid.best_score_)
print("Meilleurs paramètres : ", grid.best_params_)

# Evaluation du modèle final(RBF)
best_params = grid.best_params_
model = SVC(C=best_params['C'], kernel=best_params['kernel'], gamma=best_params['gamma'],verbose=1)
model.fit(x_train, y_train)

#Prédire les classes pour les données de test
class_predicted = model.predict(x_test)
 
#Matrice de confusion
matrix  = confusion_matrix(y_test, class_predicted)
print("Matrice de confusion:\n", matrix)
rapport = classification_report(y_test, class_predicted)
print("Classification Report:\n", rapport)

#Visualisation de la matrice de confusion
sns.heatmap(data=matrix, annot=True, fmt='g')
plt.title("Matrice de confusion")
plt.xlabel("Prédiction")
plt.ylabel("Valeur réelle")
plt.show()

# Cross-validation

#Les noyaux à comparer
kernel = ['linear', 'poly','rbf']
for x in kernel:
    model = SVC(kernel= x, random_state=42)
    score = cross_val_score(model, features, target, cv=10)
    print(f"Évaluation pour le noyau {x}:")
    print(f"Précision moyenne : {score.mean():.2f}")
    print(f"Écart-type de la précision : {score.std():.2f}","\n")
 