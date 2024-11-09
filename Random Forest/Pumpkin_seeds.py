import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np
from scipy import stats
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report, accuracy_score
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.model_selection import GridSearchCV,cross_val_score
from sklearn.impute import KNNImputer,SimpleImputer
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier,VotingClassifier

#Load dataset
dataset = pd.read_excel("/Users/apple/Desktop/ML_Algorithms/Dataset/Pumpkin_seeds.xlsx")

#exploring the data 

print(dataset.head(),"\n\n")
print(dataset.columns,"\n\n")
print(dataset.info(),"\n\n")
print(dataset.shape,"\n")
print(dataset.describe(),"\n\n")

#Data preprocessing

uniq_val_class = dataset['Class'].unique()
print("Les valeurs uniques de la colonne 'Class':",uniq_val_class,"\n")
mapping = {'Çerçevelik':'classe_1', 'Ürgüp Sivrisi':'classe_2'}
dataset["Class"]=dataset["Class"].map(mapping)
print("Les valeurs des classes après renom:\n",dataset["Class"])
percent_target = dataset["Class"].value_counts(normalize=True)
print("\n\nPourcentage des classes:", percent_target.round(2))
val_null = dataset.isnull().sum()
print("\n\nValeurs manquantes:\n",val_null)

#Visualization
    
    #Class distribution
sns.barplot(x=percent_target.index, y=percent_target.values, hue=uniq_val_class)
plt.title("Classe Distribution")
plt.ylabel('Values')
plt.xlabel('Class')
plt.legend(dataset["Class"].unique())
plt.show()

    #Class and Area distribution
sns.set_style("darkgrid")
sns.barplot(data= dataset , x="Class", y='Area',hue="Class")
plt.title("Class and Area Distribution")
plt.ylabel('Area')
plt.xlabel('Class')
plt.show()

    # Area and Perimeter Distribution
sns.scatterplot(data=dataset, x = dataset['Area'], y= dataset['Perimeter'],hue='Class')
plt.title("Area and Perimeter Distribution")
plt.xlabel('Area')
plt.ylabel('Perimeter')
plt.show()

    #Perimeter distribution
sns.set_style('darkgrid')
sns.boxplot(data=dataset,x=dataset['Perimeter'])
plt.title("Perimeter Distribution")
plt.show()

#Encoding
target = dataset["Class"]
features = dataset.drop("Class", axis=1)
print("Checking feature's column:\n",features.columns,"\n")
encoder = LabelEncoder()
target = encoder.fit_transform(target)
print(target,"\n\n")

#Splitting dataset
x_train , x_test, y_train,y_test = train_test_split(features,target,test_size=0.2,random_state=42)

#Models

     #Random Forest 
rfc_model= RandomForestClassifier()
param_grid = ({
    'max_depth':list(range(1,4)),
    'min_samples_split':[1,2,3],
    'n_estimators':list(range(4,10)),
    'criterion':['gini','entropy'],
    'max_leaf_nodes':list(range(2,6))
})
grid_search = GridSearchCV(rfc_model,param_grid=param_grid,cv=5,n_jobs=-1)
grid_search.fit(x_train,y_train)
accuracy = round(grid_search.best_score_ *100,1)
print("Sample best score:", accuracy,"\n")
print("Best parameters found on development set:\n", grid_search.best_params_,"\n")

#Final model
param_final_model=grid_search.best_params_
final_model = RandomForestClassifier(**param_final_model)
final_model.fit(x_train,y_train)  #Training final model
class_predicted = final_model.predict(x_test)  #Prediction

#Evaluation
rapport = classification_report(y_test,class_predicted)
print("Classification report:\n",rapport)
matrix = confusion_matrix(y_test,class_predicted)   #Confusion matrix
print("Confusion Matrix:\n",matrix,"\n\n")
sns.heatmap(matrix, annot=True, fmt='d')
plt.title('Random Forest Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

    #Voting Classifier

#Checking ouliers
threshold = 3
outlier_index = features[(stats.zscore(features) > threshold).any(axis=1)].index #lignes contenant un outlier dans les colonnes de features
print("Data before removing outlier:", features.shape,"\n")
features = features.drop(outlier_index)
print("Data after removing outlier:", features.shape)
target = np.delete(target, outlier_index)
 #Supprimer les mêmes ligne sur le target

#Scaling
scaler = StandardScaler()
slice_1 = dataset.columns[0:6]  #Selecting columns to standardize
slice_2 = dataset.columns[6:-2]
features_slice_1 = scaler.fit_transform(features[slice_1]) # standardization of values with column's name (slice1-slice2)
features_slice_2 = scaler.fit_transform(features[slice_2])
slice_1_dtfrm = pd.DataFrame(features_slice_1, columns=slice_1)
slice_2_dtfrm = pd.DataFrame(features_slice_2, columns=slice_2)  #Creating dataframe using the columns name
full_features = pd.concat([slice_1_dtfrm,slice_2_dtfrm],axis=1)
print("Features:",full_features,"\n\n")

#Second split
x_train_2, x_test_2, y_train_2, y_test_2 = train_test_split(full_features,target,test_size=0.2,random_state=42)
logistic_model = LogisticRegression(max_iter=10000, penalty='l1',solver='liblinear',verbose=2)
svm_model = SVC()
param_grid_svm =({
    'C':[0.01,0.1,1,10],
    'kernel':['rbf','poly','linear']
})
grid_search_svm = GridSearchCV(svm_model,param_grid=param_grid_svm,cv=5,n_jobs=-1)
grid_search_svm.fit(x_train_2,y_train_2)
final_svm_model = grid_search_svm.best_estimator_
accuracy = round(grid_search_svm.best_score_ *100,1)
print("Sample best score:", accuracy,"\n")

#Final voting classifier model
voting_model = VotingClassifier(estimators=[('log', logistic_model),('svm', final_svm_model)],voting='hard')
voting_model.fit(x_train_2,y_train_2)
class_predicted = voting_model.predict(x_test_2)

#Evaluation 
rapport = classification_report(y_test_2,class_predicted)
print("Classification report:\n",rapport)
matrix = confusion_matrix(y_test_2,class_predicted)

#Visualization
sns.heatmap(matrix, annot=True, fmt='d')
plt.title('Voting Classifier Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

#Comparing the performance of models
models = [('Logistic Regression', logistic_model),('SVM', final_svm_model)]
for name, algorithm in models:
     rapport = classification_report(y_test_2,class_predicted)
     print("Classification report for ",name,":\n",rapport)
     
#Confusion matrix of each models
for name, algorithm in models:
     matrix = confusion_matrix(y_test_2,class_predicted)
     sns.heatmap(matrix, annot=True, fmt='d')
     plt.title('Confusion Matrix for '+name)
     plt.xlabel('Predicted')
     plt.ylabel('Actual')
     plt.show()
     
















