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
from imblearn.over_sampling import SMOTE

#Load data
dataset= pd.read_csv('/Users/apple/Desktop/ML_Algorithms/Decision Tree/heart_attack_prediction_dataset.csv')
print(dataset.head())
print(dataset.info())

#Exploring data
val_null = dataset.isnull().sum()
print(val_null)
print(dataset.describe())
 
#Data preprocessing
dataset.drop(columns=['Patient ID'],inplace=True)
dataset.columns = [col.replace(" ", "_") for col in dataset.columns]
print(dataset.columns)
uniq_val = dataset['Heart_Attack_Risk'].unique()
print("Les valeurs uniques:",uniq_val)

percent = dataset["Heart_Attack_Risk"].value_counts(normalize=True).round(2)
print(f'percentage:{percent:}')
sns.barplot(x=percent.index, y=percent.values)
plt.title("Heart_Attack_Risk Distribution")
#plt.show()

rounded_col = [col for col in dataset.columns if dataset[col].dtype == 'float']
for col in rounded_col:
    dataset[col] = dataset[col].round(2)
    
dataset[['Systolic', 'Diastolic']]=dataset["Blood_Pressure"].str.split("/", expand=True)
dataset.drop(columns=['Blood_Pressure'], inplace=True)
 
dataset['Systolic'] = dataset['Systolic'].astype(int)
dataset['Diastolic'] = dataset['Diastolic'].astype(int)
 
feature = dataset.drop(columns=['Heart_Attack_Risk'])
target =dataset['Heart_Attack_Risk']
 
print(dataset.columns,'\n\n')
encoder = OneHotEncoder(sparse_output=False)
label_encod = LabelEncoder()
target = label_encod.fit_transform(target)

cat_col =  ['Sex','Diet', 'Country', 'Continent', 'Hemisphere'] 
print(cat_col,"\n")
column_encoded = encoder.fit_transform(feature[cat_col])
dtfrm_cat= pd.DataFrame(column_encoded,columns=encoder.get_feature_names_out(cat_col))
feature_encoded = pd.concat([feature,dtfrm_cat],axis=1)
feature_encoded.drop(columns=cat_col, axis=1, inplace=True)
print(feature_encoded.columns)

x_train,x_test,y_train,y_test =train_test_split(feature_encoded,target,test_size=0.2,random_state=42)
smote = SMOTE()
x_train_sm,y_train_sm =smote.fit_resample(x_train,y_train)

tree =DecisionTreeClassifier(criterion='entropy',
                             max_depth=None, 
                             ccp_alpha=0.015,random_state=42)
tree.fit(x_train_sm,y_train_sm)
score =tree.score(x_train_sm,y_train_sm)
print(f'Score: {score:.3f}',"\n")
class_pred = tree.predict(x_test)
rapport = classification_report(y_test,class_pred)
print(rapport,"\n")
matrix =confusion_matrix(y_test,class_pred)
print('Matrice de confusion:',matrix)

sns.heatmap(matrix, annot=True, fmt='g')
plt.title('Matrice de confusion - Decision Tree')
plt.xlabel('Prédiction')
plt.ylabel('Valeur réelle')
plt.show()

plot_tree(tree, feature_names=feature_encoded.columns,filled=True)
plt.show()
