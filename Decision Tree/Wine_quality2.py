import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split,cross_validate,KFold,GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

#Load data
dataset= pd.read_csv('/Users/apple/Desktop/ML_Algorithms/Decision Tree/winequality-red.csv')
print(dataset.head())
print(dataset.info())
#Exploring data
print(dataset.isnull().sum())
print(dataset['quality'].value_counts(normalize=True))
#Data preprocessing
mapping = {3: 0, 4: 0, 5: 0, 6: 1, 7: 1, 8: 1}
dataset['quality'] = dataset['quality'].map(mapping)
print(dataset['quality'].value_counts().plot.bar())
#plt.show()
print(dataset['quality'].head())
dataset['chlorides']=dataset['chlorides'].round(2)
dataset['density']=dataset['density'].round(2)

#Splitting data
scaler =StandardScaler()
X = dataset.drop('quality', axis=1)
y = dataset['quality']
X = scaler.fit_transform(X)
X_train,  x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#GridSearch
params =({
    'criterion':['gini','entropy'],
    'max_depth':[4,7,15,None],
    'ccp_alpha': [0.0, 0.015, 0.1, 1.0],
    'splitter':['best','random'],
    'min_samples_leaf':[1,2,3]
})
model = DecisionTreeClassifier()
gridsearch = GridSearchCV(estimator=model, param_grid=params, cv=8, verbose=1)
gridsearch.fit(X_train, y_train)
print("Best Parameters: ", gridsearch.best_params_)
print("Best Score: ", gridsearch.best_score_,"\n\n")
#Evaluation du model final
best_params = gridsearch.best_params_
tree = DecisionTreeClassifier(criterion=best_params['criterion'],
                               max_depth=best_params['max_depth'],
                               ccp_alpha=best_params['ccp_alpha'],
                               splitter=best_params['splitter'],
                               min_samples_leaf=best_params['min_samples_leaf'],
                               random_state=42)
tree.fit(X_train, y_train)
#Prediction
class_pred = tree.predict( x_test)
rapport = classification_report(y_test, class_pred)
print("Classification Report:\n", rapport,"\n")
#Confusion matrix
matrix = confusion_matrix(y_test, class_pred)
print("Confusion Matrix:\n", matrix,"\n\n")
sns.heatmap(matrix, annot=True, fmt='d')
plt.title('Decision Tree Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
plot_tree(tree,feature_names=dataset.iloc[:,:-1].columns,filled=True)
plt.show()

#Modele Logistic Regression
logistic_model =LogisticRegression()
params = ({
    'penalty': ['l1', 'l2', 'elasticnet', 'none'],
    'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
    'class_weight': ['balanced', None]
})
grid=GridSearchCV(estimator=logistic_model,param_grid=params,cv=10,verbose=1)
grid.fit(X_train,y_train)
print('Meilleur score:',grid.best_score_)
print('Meilleurs paramètres',grid.best_params_)
#Evaluation du modèle final
params = grid.best_params_
model=LogisticRegression(
    penalty=params['penalty'],
    C=params['C'],
    class_weight=params['class_weight']
)
model.fit(X_train,y_train)
class_predict = model.predict( x_test) 
rapport = classification_report(y_test, class_predict)
print("Classification report:\n",rapport)
