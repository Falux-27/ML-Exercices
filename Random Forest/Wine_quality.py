import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import  RandomForestClassifier,BaggingClassifier
from sklearn.ensemble import VotingClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import KNNImputer,SimpleImputer
from sklearn.metrics import confusion_matrix
 

#Load data
dataset = pd.read_csv('/Users/apple/Desktop/ML_Algorithms/REGRESSION LOGISTIQUE/winequality-red.csv')
print(dataset.head(),"\n\n")

#Exploring data
print(dataset.info(),'\n')
print(dataset.isnull().sum(),'\n')

#Data preprocessing
print(dataset['quality'].unique())
val_uniq = dataset['quality'].value_counts(normalize=True)
print("Les valeurs uniques:",val_uniq)
sns.barplot(x = val_uniq.index, y= val_uniq.values)
plt.title("Quality Distribution")
plt.show()

bins = [3,5,8]
labels = [0,1]
dataset["quality"]= pd.cut(dataset["quality"],bins=bins, labels=labels)
print(dataset.head(),"\n\n")

print(dataset['quality'].unique())
print(dataset['quality'].value_counts(normalize=True).plot.bar())
plt.title("Quality Distribution")
plt.show()
corr_matrix = dataset.corr()
sns.heatmap(data=corr_matrix, annot=True, cmap='coolwarm', fmt='.2f' )
plt.show()

print(dataset['quality'].unique())
#Imputation of missing values
imputer = SimpleImputer(missing_values=np.nan,strategy='most_frequent')
dataset['quality']= imputer.fit_transform(dataset[['quality']])
#Splitting in features - target
features = dataset.iloc[:,:-1]
target = dataset.iloc[:,-1]
print(target.unique())

#Splitting the dataset
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

#GridSearch
model = RandomForestClassifier()
param_grid= ({
    'bootstrap':[True,False],
    'n_estimators':[25,50,75,100],
    'max_depth':[1,3,5],
    'min_samples_split':[2,3,5],
    'min_samples_leaf':[1,2,3],
    'max_leaf_nodes':[2,3]
})
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1 
)
grid_search.fit(x_train, y_train)
print(f'Meilleur-score: {grid_search.best_score_:.2f}')
print(f'Meilleurs paramètres: {grid_search.best_params_}')

#Model
best_params = grid_search.best_params_
model = RandomForestClassifier(
    bootstrap= best_params['bootstrap'],
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    min_samples_split=best_params['min_samples_split'],
    min_samples_leaf=best_params['min_samples_leaf'],
    max_leaf_nodes=best_params['max_leaf_nodes'],
    oob_score=True,
    n_jobs=-1
)
model.fit(x_train,y_train)

#Prediction
y_pred = model.predict(x_test)

#Evaluation
print(f'Accuracy: {model.score(x_test, y_test):.2f}')
print(f'F1 Score: {f1_score(y_test, y_pred, average="macro"):.2f}')
print(f'RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}')
print(f'R2 Score: {r2_score(y_test, y_pred):.2f}')
print(f'OOB Score: {model.oob_score_:.2f}')


#Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.show()

#Bagging
estimator =grid_search.best_estimator_
_estimator =best_params['n_estimators'] 
bagging_model = BaggingClassifier(estimator=estimator,n_estimators=_estimator, random_state=42)
bagging_model.fit(x_train, y_train)
y_pred_bagging = bagging_model.predict(x_test)

#Evaluation
print(f'Bagging Accuracy: {bagging_model.score(x_test, y_test):.2f}')
print(f'Bagging F1 Score: {f1_score(y_test, y_pred_bagging, average="macro"):.2f}')
print(f'Bagging RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_bagging)):.2f}')
print(f'Bagging R2 Score: {r2_score(y_test, y_pred_bagging):.2f}')

#Confusion Matrix
cm_bagging = confusion_matrix(y_test, y_pred_bagging)
sns.heatmap(cm_bagging, annot=True, fmt='d')
plt.show()

#Voting classifier
model1 = LogisticRegression(max_iter=10000,penalty='l1',solver='liblinear')
param_grid = ({
    'C': [0.1, 1, 10],
    'kernel': ['rbf','poly','linear']  # linear, poly, rbf, sigmoid
})
svm_model = SVC()
grid_search =GridSearchCV(svm_model,param_grid=param_grid,cv=5,n_jobs=-1)
grid_search.fit(x_train, y_train)

model2 = grid_search.best_estimator_
voting_model = VotingClassifier(estimators=[('lr', model1),('svc', model2)], voting='hard')
voting_model.fit(x_train, y_train)
y_pred_voting = voting_model.predict(x_test)

#Evaluation
print(f'Voting Accuracy: {voting_model.score(x_test, y_test):.2f}')
print(f'Voting F1 Score: {f1_score(y_test, y_pred_voting, average="macro"):.2f}')
print(f'Voting RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_voting)):.2f}')
print(f'Voting R2 Score: {r2_score(y_test, y_pred_voting):.2f}')

#Confusion Matrix
cm_voting = confusion_matrix(y_test, y_pred_voting)
sns.heatmap(cm_voting, annot=True, fmt='d')
plt.show()

#Comparing models
models = [('Random Forest', model), ('Bagging', bagging_model), ('Voting', voting_model)]
for name, model in models:
    y_pred = model.predict(x_test)
    print(f'{name} Accuracy: {model.score(x_test, y_test):.2f}')
    print(f'{name} F1 Score: {f1_score(y_test, y_pred, average="macro"):.2f}')
    print(f'{name} RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}')
    print(f'{name} R2 Score: {r2_score(y_test, y_pred):.2f}')
    print("\n")






