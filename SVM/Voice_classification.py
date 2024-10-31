import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split ,cross_val_score , StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

from sklearn.svm import SVC

# Load dataset
dataset = pd.read_csv("/Users/apple/Desktop/ML_Algorithms/SVM/voice 2.csv")
print(dataset.head(5))

#Exploring data
print(dataset.shape)
print(dataset.info())
print(dataset.isnull().sum())

#Data preprocessing
print(dataset.columns)
dataset =dataset.round(2)
print(dataset)
val_uniq = dataset['label'].unique()
print("Les valeurs uniques:",val_uniq)
percent_val_uniq = dataset['label'].value_counts(normalize=True) 
print("Pourcentage de chaque catégorie:",percent_val_uniq.round(2))
sns.barplot(x = percent_val_uniq.index, y= percent_val_uniq.values, hue=percent_val_uniq.index)
plt.title("Label Distribution")
plt.show()
sns.barplot(data=dataset, x='label', y ='meanfreq' , hue='label')
plt.title("Label vs Meanfreq")
plt.show()

#Feature engineering
    #--->Encoding and scaling
numericals_col = dataset.select_dtypes(include=['int64', 'float64'])
print("Numericals columns:", numericals_col.columns)
features = numericals_col
target = dataset['label']
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
encoder = LabelEncoder()
target = encoder.fit_transform(target)
print(features_scaled)
print(target)

#Splitting the dataset
x_train, x_test, y_train,y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

#Cross validation
classifier = LogisticRegression(max_iter=10000, penalty='l1', solver='liblinear')
fold_numb = 10 
strat_k_folds = StratifiedKFold(n_splits=fold_numb, shuffle=True, random_state=42)
cross_valid = cross_val_score(classifier, features_scaled, target, cv=strat_k_folds)
print(f'Cross-Validation Results (Accuracy): {cross_valid}')
print(f'Cross_validation moyenne: {cross_valid.mean()}')

#Model Logistic Regression
    #Training
classifier.fit(x_train, y_train)
    #Prediction
class_predicted = classifier.predict(x_test)
#Confusion matrix     
matrice = confusion_matrix(y_test, class_predicted)
print("Matrice de confusion:\n", matrice)
#heatmap
sns.heatmap(matrice, annot=True, fmt='d')
plt.title("Matrice de confusion - Logistic Regression")
plt.xlabel("Prédiction")
plt.ylabel("Valeur réelle")
plt.show()
#Metriques du model
metrics = classification_report(y_test, class_predicted)
print("Metriques du modèle de Regression Logistique:\n", metrics)


#Model SVM
    #Cross-validation
noyau = ['linear', 'rbf', 'poly']
for x in noyau:
    model_svm = SVC(kernel=x)
    score = cross_val_score(model_svm, features_scaled, target, cv=strat_k_folds)
    print(f"Évaluation pour le noyau {x}:")
    print(f"Précision moyenne : {score.mean():.2f}")
    print(f"Écart-type de la précision : {score.std():.2f}","\n")

#Training model with best kernel
model = SVC(kernel='rbf', C=10, gamma=0.1, random_state=42)
model.fit(x_train, y_train)
#Prediction
class_predicted = model.predict(x_test)
#Confusion matrix
matrix = confusion_matrix(y_test, class_predicted)
print("Matrice de confusion:\n", matrix)
#Heatmap
sns.heatmap(matrix, annot=True, fmt='d')
plt.title("Matrice de confusion - SVM avec rbf")
plt.xlabel("Prédiction")
plt.ylabel("Valeur réelle")
plt.show()
#Metriques du model
rapport = classification_report(y_test, class_predicted)
print("Metriques SVM avec rbf:\n", rapport)

