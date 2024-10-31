import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split,StratifiedKFold,GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder,StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.impute import KNNImputer
from sklearn.svm import SVC

#Load data
dataset=pd.read_csv('/Users/apple/Desktop/ML_Algorithms/SVM/alzheimers_disease_data.csv')

#Exploring data
print(dataset.shape)
print(dataset.info())
print(dataset.isnull().sum(),"\n\n")

#Data preprocessing
dataset =dataset.round(2)
print(dataset.head(),"\n\n")
dataset.drop(columns=['PatientID'],axis=1,inplace=True)
dataset.drop(['DoctorInCharge'],axis=1,inplace=True)
print(dataset.columns,"\n")
print(dataset['Diagnosis'].value_counts())
val_uniq=dataset['Diagnosis'].value_counts()

sns.set_style('darkgrid')
sns.barplot(x=val_uniq.index, y=val_uniq.values,hue=val_uniq.index)
plt.title("Diagnosis Distribution")
plt.show()

features =dataset.iloc[:,:-1]
target = dataset.iloc[:,-1]
print(features.columns)

xtrain,xtest,ytrain,ytest =train_test_split(features,target,test_size=0.2,random_state=42)
print(xtrain.shape, ytrain.shape, xtest.shape, ytest.shape,end="\t")
smote =SMOTE()
xtrain_res,ytrain_res=smote.fit_resample(xtrain,ytrain)
print("\n",xtrain_res.shape, ytrain_res.shape,"\n\n")

#Grid_searchCV
model = DecisionTreeClassifier()
params =({
    'criterion':['gini','entropy'],
    'max_depth':[None,3,4,5,6,7],
    'min_samples_split':[2,3],
    'max_leaf_nodes':[4,6,7],
    'ccp_alpha': [0.0, 0.01, 0.1, 1.0]
})
grid_search = GridSearchCV(estimator=model, param_grid=params, cv=10, verbose=1)
grid_search.fit(xtrain_res,ytrain_res)
print(f'Meilleur-score:{grid_search.best_score_:.2f}')
print(f'Meilleurs paramètres:{grid_search.best_params_}')

#Decision Tree
best_param =grid_search.best_params_
Dtree_model = DecisionTreeClassifier(
    criterion=best_param['criterion'],
    max_depth=best_param['max_depth'],
    min_samples_split=best_param['min_samples_split'],
    max_leaf_nodes=best_param['max_leaf_nodes'],
    ccp_alpha=best_param['ccp_alpha'],
    random_state=42
)
Dtree_model.fit(xtrain_res,ytrain_res)
values_predict = Dtree_model.predict(xtest)
rapport = classification_report(ytest,values_predict)
print("\nClassification Report(decision Tree):\n", rapport)
matrix = confusion_matrix(ytest,values_predict)

#Logistic Regression
model_logistic =LogisticRegression(
                    max_iter=10000,
                    penalty='l2',
                     solver='liblinear')
model_logistic.fit(xtrain_res,ytrain_res)
values_predict_logistic = model_logistic.predict(xtest)
rapport_logistic = classification_report(ytest,values_predict_logistic)
print("\nClassification Report(Logistic Regression):\n", rapport_logistic)
matrix_logistic = confusion_matrix(ytest,values_predict_logistic)

#Visualisation
sns.heatmap(matrix,annot=True,fmt='g')
plt.title('Decision Tree Confusion Matrix')
plt.xlabel('Prediction')
plt.ylabel('valeurs réelles')
plot_tree(Dtree_model, feature_names=features.columns,filled=True)
plt.title('Decision Tree')
plt.show()
########
sns.heatmap(matrix_logistic,annot=True,fmt='g')
plt.title('Logistic Regression Confusion Matrix')
plt.xlabel('Prediction')
plt.ylabel('Valeurs réelles')
plt.show()
