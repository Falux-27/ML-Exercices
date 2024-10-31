import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split,cross_val_score , StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, OneHotEncoder,LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer

#-------->Les Pipelines----------->

# Charger les données
iris = load_iris()
dataset = pd.DataFrame(data=iris.data, columns=iris.feature_names)
print(dataset.head())
#Séparation en features-targets
features = iris.data
target = iris.target
print("Les features:",features)
print("Les target:", target)
#Division des données en train-test
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

#Création du pipeline
pipeline = Pipeline(steps= [
    
    ('normalisation', StandardScaler()), 
    ('features_select',SelectFromModel(LogisticRegression(max_iter=10000, solver='liblinear', penalty='l1'))),
    ('model',LogisticRegression(max_iter=10000, solver='liblinear', penalty='l1'))
])
#Entraînement du pipeline sur le donnees
pipeline.fit(x_train, y_train)

#Evaluation sur les donnees de test
y_pred = pipeline.predict(x_test)
print("Les prédictions du modèle:", y_pred)

#Evaluation du pipeline
performance = pipeline.score(x_test, y_test)
print(f"La precision du modele sur les donnees de test est d'environ: {performance:.2f}","\n\n")

#-------->ColumnTransformer----------->

"""Permet d'appliquer différentes transformations à différentes colonnes d'un DataFrame"""
data = pd.DataFrame({
    'age': [25, 32, 47, 51],
    'fare': [72.5, 53.1, 9.5, 8.0],
    'embarked': ['S', 'C', 'Q', 'S'],
    'sex': ['male', 'female', 'female', 'male'],
    'pclass': [1, 2, 3, 1]
})

#Définir les colonnes numériques et catégorielles
col_num = ['age','fare']
categoric_col = ['embarked','sex','pclass']

#Définir les transformations pour chaque colonnes
col_num_transform = StandardScaler()
cat_transform = OneHotEncoder(handle_unknown='ignore')

#Combiner les transformations dans un ColumnTransformer
transform = ColumnTransformer(transformers=[
    ('normalisation',col_num_transform,col_num),
    ('encodage', cat_transform, categoric_col)
])
# Appliquer ColumnTransformer sur les données
data_transformed =transform.fit_transform(data)
print('Le dataset after transformation:\n',data_transformed)

#-------->SimpleImputer----------->

dataframe_ = pd.DataFrame({
    'age': [25, 32, 47, 51,84,10,35,23],
    'fare': [72.5, 53.1, 9.5, 8.0,12.8,3.3,np.nan,np.nan],
    'embarked': [np.nan,np.nan, 'Q', 'S','C','S','C',np.nan],
    'sex': [np.nan,'female',np.nan, 'female','male','male','female',np.nan],
    'pclass': [np.nan, np.nan, 3, 1,1,1,3,3]
})

#Imputation par la moyenne : 'mean'
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
dataframe_['pclass']= imputer.fit_transform(dataframe_[['pclass']]).astype(int)
dataframe_['fare']= imputer.fit_transform(dataframe_[['fare']]).astype(int)
print('Le dataset avec les valeurs manquantes remplies par la moyenne:\n',dataframe_)

#Imputation par la fréquence : 'most_frequent'
imputer_cat= SimpleImputer(missing_values=np.nan, strategy='most_frequent')
dataframe_['embarked'] = imputer_cat.fit_transform(dataframe_[['embarked']]).reshape(-1) #Do the same like ravel
dataframe_['sex'] = imputer_cat.fit_transform(dataframe_[['sex']]).ravel()  #Transforme le tableau 2D en 1D
print('Le dataset avec les valeurs manquantes remplies par la modalité:\n',dataframe_)


#-------->KNNImputer----------->
"""Cette méthode d'imputation est applicable qu'au variable numériques dans le cas échéant nous 
devrons encoder les valeur en format numérique pour pouvoir les imputer"""

dataset_ = pd.DataFrame({
    'age': [25, 32, 47, 51,84,10,35,23],
    'fare': [72.5, 53.1, 9.5, 8.0,12.8,3.3,np.nan,np.nan],
    'embarked': [np.nan,np.nan, 'Q', 'S','C','S','C',np.nan],
    'sex': [np.nan,'female',np.nan, 'female','male','male','female',np.nan],
    'pclass': [np.nan, np.nan, 3, 1,1,1,3,3]
})
imputer_ = KNNImputer(n_neighbors=3)
numerical_cols =[col for  col in dataset_.columns if dataset_[col].dtype in ['float64', 'int64']]
print(numerical_cols)
dataset_[numerical_cols]= imputer_.fit_transform(dataset_[numerical_cols]).astype(int)
print('Le dataset avec les valeurs manquantes remplies par la KNN:\n',dataset_)
 
 
#----->Exercice de prétraitement des données 
dataset = pd.read_csv('/Users/apple/Desktop/ML_Algorithms/Dataset/titanic.csv')
print(dataset.head(),"\n\n")
#Exploring data
print(dataset.isnull().sum(),"\n\n")
print(dataset.info(),"\n\n")
dataset.drop(columns='PassengerId', inplace=True)
#Imputting missing values 
imputerkNN= KNNImputer(n_neighbors=3)
dataset['Age']= imputerkNN.fit_transform(dataset[['Age']])
imputer_s = SimpleImputer(missing_values=np.nan,strategy='most_frequent')
dataset['Cabin']=imputer_s.fit_transform(dataset[['Cabin']]).ravel()
dataset['Embarked']=imputer_s.fit_transform(dataset[['Embarked']]).reshape(-1)
print(dataset.isnull().sum(),"\n\n")
#Splitting into feature - target
features= dataset.iloc[:, 1:]
target= dataset.iloc[:,0]
print(target.value_counts(normalize=True),"\n\n")
print(features.columns)
#Select categorical columns
categoric_columns = [col for col in features.columns if features[col].dtype == 'object']
print("Categoricals columns :",categoric_columns,"\n\n")
#Select numerical columns
numerical_columns = [col for col in features.columns if col not  in categoric_columns]
print("Numerical columns :",numerical_columns,"\n\n")
#Encoding feature's categorical columns
encoder = OneHotEncoder(sparse_output=False)
features_encoded =  encoder.fit_transform(dataset[categoric_columns]).astype(int)
one_hot_dtfrm = pd.DataFrame(features_encoded, columns=encoder.get_feature_names_out(categoric_columns))
features= pd.concat([features, one_hot_dtfrm], axis=1)
#Encoding target
encoder_t = LabelEncoder()
target = encoder_t.fit_transform(target)
 
#Dropping original categorical columns
features = features.drop(categoric_columns, axis=1)
print(features.head(),"\n\n")
#Scaling feature's numerical columns
scaler = StandardScaler()
features[numerical_columns] = scaler.fit_transform(features[numerical_columns]).round(2)
print(features.head(),"\n\n")
#Splitting into training and testing set
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)
#Oversampling
smote =SMOTE()
x_train_res, y_train_res =smote.fit_resample(x_train, y_train)
print(pd.Series(y_train_res).value_counts(),"\n\n")
#Cross validation
model = LogisticRegression(max_iter=10000, penalty='l1',solver='liblinear')
fold_num= 10
cross_val_type = StratifiedKFold(n_splits=fold_num, shuffle=True, random_state=42)
cross_val = cross_val_score(model, features, target, cv=cross_val_type,verbose=1)
print("Score des de chaque plis:",cross_val.round(2),"\n\n")
print("Moyenne des scores :", cross_val.mean().round(2),"\n\n")
#Training the model
    #Logistic Regression model
model.fit(x_train_res, y_train_res)
class_predict = model.predict(x_test)
metrics = classification_report(y_test, class_predict)
print("Classification report:\n", metrics,"\n\n")

#Cross validation for SVM
model_svm= SVC(C=0.1, kernel='rbf')
fold_num= 10
cross_val_type = StratifiedKFold(n_splits=fold_num, shuffle=True, random_state=42)
cross_val = cross_val_score(model_svm, features, target, cv=cross_val_type,verbose=1)
print("Score des de chaque plis:",cross_val.round(2),"\n\n")
print("Moyenne des scores :", cross_val.mean().round(2),"\n\n")
#Training the model
model_svm.fit(x_train_res,y_train_res)
class_predict_svm = model_svm.predict(x_test)
metrics_svm = classification_report(y_test, class_predict_svm)
print("Classification report:\n",metrics_svm,"\n\n")

#Confusion matrix
matrix = confusion_matrix(y_test, class_predict)
sns.heatmap(matrix, annot=True, fmt='d')
plt.title('Confusion Matrix of the Logistic Regression model')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

matrix_svm = confusion_matrix(y_test, class_predict_svm)
sns.heatmap(matrix_svm, annot=True, fmt='d')
plt.title('Confusion Matrix of the Support Vector Machine model')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

 



