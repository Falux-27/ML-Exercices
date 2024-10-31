import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split,cross_validate,StratifiedKFold,GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

#Load data
dataset= pd.read_csv('/Users/apple/Desktop/ML_Algorithms/Decision Tree/winequality-red.csv')
print(dataset.head())
print(dataset.info())
#Exploring data
val_null = dataset.isnull().sum()
print(val_null)
print(dataset.describe(),"\n\n")
target_uniq = dataset['quality'].unique()
print(f"Unique target values: {target_uniq}")
dataset['quality']=dataset['quality'].map({3:0,4:1,5:2,6:3,7:4,8:5})
dataset['chlorides']=dataset['chlorides'].round(2)
dataset['density']=dataset['density'].round(2)
print(dataset['density'].head())
dataset['quality'].value_counts().plot.bar()
#plt.show()

#Splitting data
feature= dataset.iloc[:,:-1]
#Scaling
scaler =StandardScaler()
feature= scaler.fit_transform(feature)
target= dataset.iloc[:,-1]
x_train, x_test, y_train, y_test = train_test_split(feature, target, test_size=0.2, random_state=42)
#Oversample
smote = SMOTE()
x_train_rs,y_train_rs = smote.fit_resample(x_train,y_train)
#Modele Decision tree
tree= DecisionTreeClassifier()
hyperparams = ({
    'criterion':['gini','entropy'],
    'ccp_alpha': [0.0, 0.015, 0.1, 1.0],
    'splitter':['best','random'],
    'max_depth': [None, 5, 10, 15]
})
grid_search = GridSearchCV(estimator=tree,param_grid=hyperparams,cv=10,)
grid_search.fit(x_train_rs,y_train_rs)
print("Meilleur score : ", grid_search.best_score_)
print("Meilleurs paramètres : ", grid_search.best_params_)
#Evaluation du modèle final
best_params = grid_search.best_params_
model = DecisionTreeClassifier(
    criterion=best_params['criterion'],
    ccp_alpha= best_params['ccp_alpha'],
    splitter=best_params['splitter']
)
model.fit(x_train_rs,y_train_rs)
class_predict = model.predict(x_test)
score = model.score(x_test,y_test)
print("Le score du model:",score,"\n")
metrics = classification_report(y_test, class_predict)
print("Classification report:\n",metrics)
#Matrice de confusion
matrix  =confusion_matrix(y_test, class_predict)
sns.heatmap(matrix, annot=True, fmt='g')
plt.title("Matrice de confusion - Decision Tree")
plt.xlabel("Prédiction")
plt.ylabel("Valeur réelle")
#plt.show()
#Arbre de decision
#plot_tree(model,feature_names=dataset.iloc[:,:-1].columns,filled=True)
#plt.show()

#Modele Logistic Regression
logistic_model =LogisticRegression()
params = ({
    'penalty': ['l1', 'l2', 'elasticnet', 'none'],
    'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
    'class_weight': ['balanced', None]
})
grid=GridSearchCV(estimator=logistic_model,param_grid=params,cv=10,verbose=1)
grid.fit(x_train_rs,y_train_rs)
print('Meilleur score:',grid.best_score_)
print('Meilleurs paramètres',grid.best_params_)
#Evaluation du modèle final
params = grid.best_params_
model=LogisticRegression(
    penalty=params['penalty'],
    C=params['C'],
    class_weight=params['class_weight']
)
model.fit(x_train_rs,y_train_rs)
class_predict = model.predict(x_test) 
rapport = classification_report(y_test, class_predict)
print("Classification report:\n",rapport)