import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

#Load data
dataset= pd.read_csv('/Users/apple/Desktop/ML_Algorithms/Decision Tree/drug200.csv')  
print(dataset.head())
print(dataset.info())
#Exploring data
val_null = dataset.isnull().sum()
print(val_null)
val_uniq_target = dataset['Drug'].unique()
print(val_uniq_target)
percentage = dataset['Drug'].value_counts(normalize=True) 
print(f"Pourcentage des classes:",percentage)
sns.barplot(x=percentage.index, y=percentage.values, hue=val_uniq_target)
plt.show()
feature= dataset.iloc[:,:-1]
target= dataset.iloc[:,-1]

#Encoding
encoder = OneHotEncoder(sparse_output=False)
categoricals_col =[col for col in feature.columns if feature[col].dtype == 'object']
column_encoded =encoder.fit_transform(feature[categoricals_col])
dtfrm_encod = pd.DataFrame(column_encoded, columns=encoder.get_feature_names_out(categoricals_col))
feature=feature.drop(columns=categoricals_col, axis=1)
dataset_encoded= pd.concat([feature, dtfrm_encod],axis=1)

encodertwo = LabelEncoder()
target = encodertwo.fit_transform(target) 
print(dataset_encoded,"\n\n")
print(target) 
#Splitting dataset
x_train, x_test, y_train, y_test = train_test_split(dataset_encoded, target, test_size=0.2, random_state=42)

#Models

#Entrainement avec Gini
model = DecisionTreeClassifier(criterion='gini',max_depth=4,random_state=42)
model.fit(x_train, y_train)
#Prediction
y_pred = model.predict(x_test)
#Evaluation
rapport1 = classification_report(y_test, y_pred)
print(rapport1,"\n\n")

#Entrainement avec Entropie
model2 = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=42,ccp_alpha=0.001)
model2.fit(x_train, y_train)
#Prediction
y_pred2 = model2.predict(x_test)
#Score du modèle
score = model2.score(x_test,y_test)
print(f"Score : {score}")
 #Evaluation du modèle
rapport2 = classification_report(y_test, y_pred2)
print(rapport2)
#Matrice de confusion
matrice =confusion_matrix(y_test, y_pred)
print("Matrice de confusion:\n", matrice)
sns.heatmap(matrice, annot=True, fmt='g')
plt.title("Matrice de confusion - Gini")
plt.xlabel("Prédiction")
plt.ylabel("Valeur réelle")
plt.show()

#Visualisation du modèle
plot_tree(model2, feature_names=dataset_encoded.columns,
          filled=True, rounded=True)
plt.show()



