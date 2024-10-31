import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score

# Création d'un DataFrame avec des données de température, humidité et ensoleillé
data = pd.DataFrame({
    'Température': [30, 20, 25, 28, 22, 24],
    'Humidité': [70, 80, 65, 90, 85, 75],
    'Ensoleillé': ['oui', 'non', 'oui', 'non', 'non','oui']  # 1 = Oui, 0 = Non
})
print(data)

#Conversion des variable categorielle
data= pd.get_dummies(data=data, columns= ['Ensoleillé']).astype(int)
print(data)

# Création d'une matrice de features et d'un vecteur de labels
features= data.iloc[:,:-2]
labels= data.iloc[:,2:4]
print(features.columns)
print(labels)

#Creation du modele
tree = DecisionTreeClassifier()

# Entrainement du modele
tree.fit(features,labels)

# Prediction

# Evaluation du modele
#Visualisation
plot_tree(tree, feature_names=data.columns, class_names=['Oui', 'Non'], filled=True, rounded=True)
plt.show()