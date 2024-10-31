import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split,cross_val_score,cross_validate,StratifiedKFold,GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder,StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.impute import KNNImputer
from sklearn.svm import SVC
#Load data
dataset = pd.read_csv('/Users/apple/Desktop/ML_Algorithms/REGRESSION LOGISTIQUE/Invistico_Airline.csv')

#Exploring data
print(dataset.head())
print(dataset.info(),"\n")

#Data preprocessing
knn_imputer = KNNImputer(n_neighbors=3)
dataset['Arrival Delay in Minutes'] = knn_imputer.fit_transform(dataset[['Arrival Delay in Minutes']])

val_uniq= dataset['satisfaction'].unique()
print("Les valeurs uniques:",val_uniq,"\n\n")

percent_val_uniq = dataset['satisfaction'].value_counts(normalize=True)
print("Pourcentage de chaque catégorie:",percent_val_uniq.round(2))

dataset.columns= [col.replace(' ','_') for col in dataset.columns]
print(dataset.columns,"\n\n")

#Visualisation
sns.countplot(data=dataset,x ='satisfaction',hue='Gender')
plt.title("Satisfaction by Gender")
plt.show()
sns.set_style('darkgrid')
sns.countplot(data=dataset,x='Type_of_Travel',hue='satisfaction')
plt.title("Satisfaction by Type of Travel")
plt.show()
sns.set_style('dark')
sns.countplot(data=dataset, x='Seat_comfort',hue='satisfaction')
plt.title(" Satisfaction by Seat Comfort")
plt.show()
num_uniq = dataset['satisfaction'].value_counts()
print(num_uniq,"\n\n")
plt.pie(num_uniq,explode=[0.1,0.1],shadow=True)
plt.title("Satisfaction distribution")
plt.legend(val_uniq)
plt.show()

numerical_cols= [num for num in dataset.columns if dataset[num].dtype in ['int64','float64']]
feature = dataset.iloc[:,1:]
target = dataset.iloc[:,0]
encoder = OneHotEncoder(sparse_output=False)
categorical_cols =[col for col in feature.columns if feature[col].dtype == 'object']
column_encod= encoder.fit_transform(feature[categorical_cols])
dtfrm_encoded = pd.DataFrame(column_encod, columns=encoder.get_feature_names_out(categorical_cols))
feature=feature.drop(columns=categorical_cols, axis=1,inplace=True)
features =pd.concat([feature, dtfrm_encoded],axis=1)
label_encoder = LabelEncoder()
target = label_encoder.fit_transform(target)

#Cross validation
fold = 10
types = StratifiedKFold(n_splits=fold,shuffle=True,random_state=42)
dtree = DecisionTreeClassifier(criterion='entropy',max_depth=8,ccp_alpha=0.015,random_state=42)
cross_val =cross_val_score(dtree,features,target,cv=types,verbose=1)
print("Score des de chaque plis:",cross_val.round(2),"\n\n")
print("Moyenne des scores :", cross_val.mean().round(2),"\n\n")

x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
#GridSearch
params =({
    'criterion':['gini','entropy'],
    'max_depth':[4,7,15,None],
    'ccp_alpha': [0.0, 0.015, 0.1, 1.0],
    'splitter':['best','random'],
    'min_samples_leaf':[1,2,3]
})
model = DecisionTreeClassifier()
grid = GridSearchCV(estimator=model,param_grid=params,cv=10,verbose=1)
grid.fit(x_train, y_train)
print("Meilleur score des plis:",grid.best_score_,"\n")
print("Meilleurs paramètres:",grid.best_params_,"\n\n")
#Evaluation du modèle final
parameter = grid.best_params_
final_model =DecisionTreeClassifier(criterion=parameter['criterion']
                                     ,max_depth=parameter['max_depth']
                                     ,ccp_alpha=parameter['ccp_alpha']
                                     ,splitter=parameter['splitter']
                                     ,min_samples_leaf=parameter['min_samples_leaf']
                                     ,random_state=42)
final_model.fit(x_train, y_train)
y_pred = final_model.predict(x_test)
rapport =classification_report(y_test,y_pred)
print("Classification report:\n",rapport,"\n\n")
matrix =confusion_matrix(y_test,y_pred)
sns.heatmap(matrix,annot=True,fmt='d')
plt.title('Decision Tree Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
plot_tree(final_model, feature_names=features.columns,filled=True)
plt.show()







