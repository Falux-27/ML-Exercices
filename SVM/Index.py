import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report

# Load dataset
col_names = ['variance ','skewness',"curtosis", "entropy", "class"]
dataset = pd.read_csv('/Users/apple/Desktop/ML_Algorithms/SVM/data_banknote_authentication.csv', names= col_names, delimiter=",")
print(pd.concat([dataset.head(), dataset.tail()]))

#Exploring data

#valeur uniques
val_uniq = dataset['class'].unique()
print("Les valeurs uniques:",val_uniq)
#Dimensions
dim = dataset.shape
print("Dimensions:(ligne-colonne)",dim)
#Distribution des classes
class_number = dataset['class'].value_counts()
print("Distribution des classes:",class_number)
#Pourcentage de chaque catégories
class_percentage = dataset['class'].value_counts(normalize=True)
print("Pourcentage de chaque catégorie:",class_percentage,"\n")
#visalisation 
 
sns.set_style('white')
sns.pairplot(dataset, hue='class')
plt.show()

sns.countplot(data=dataset,x = 'class',hue='class')
plt.show()

for col in dataset.columns:
    plt.title(col)
    dataset[col].plot.hist()
    plt.show()
    
#Separation en target-feature
features= dataset.drop('class', axis=1)
target= dataset['class']

#Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=1)

#Création du modèle
model_svm = SVC(kernel='linear')

#Appliquer le modèle
model_svm.fit(X_train, y_train)

#L'équation du modèle
coefficients = model_svm.coef_[0]
intercept = model_svm.intercept_[0]
equation = f"{coefficients[0]} * x1 + {coefficients[1]} * x2 + {intercept} = 0"
print("L'équation du modèle:", equation)

print("LEs coefficients(vecteur normal):", model_svm.coef_)
print("Le bias b:", model_svm.intercept_)
print("Les vecteurs support:",model_svm.support_vectors_)

#Prédire les classes pour les données de test
class_predict = model_svm.predict(X_test)
print("les prediction de classe:", class_predict)

#Evaluation du modèle
cm = confusion_matrix(y_test, class_predict)
print("Matrice de confusion:\n", cm)
rapport = classification_report(y_test, class_predict)
print("Classification Report:\n",rapport)

#Visualisation de la matrice de confiance
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Matrice de confusion")
plt.xlabel("Prédiction")
plt.ylabel("Valeur réelle")
plt.show()
