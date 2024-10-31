import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing 
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report

# Load dataset
dataset = pd.read_csv("/Users/apple/Desktop/ML_Algorithms/SVM/Persistent_vs_NonPersistent.csv")

#Exploring data
print(dataset.info())
print(dataset.isnull().sum())

#Data preprocessing
dataset= dataset.drop(columns=['Ptid'])
print(dataset['Persistency_Flag'].unique())
print(dataset['Persistency_Flag'].value_counts().plot.bar(color=['pink', 'magenta']))
plt.show()
print(dataset['Persistency_Flag'].value_counts(normalize=True).round(2))

#Encoding column target
encoder = LabelEncoder()
dataset['Persistency_Flag'] = encoder.fit_transform(dataset['Persistency_Flag'])
print(type(dataset['Persistency_Flag']))

#Rename target column
dataset.rename(columns={'Persistency_Flag':'persistency'}, inplace=True)
#Collecting categoricals columns
categoricals_column = dataset.select_dtypes(include=['object']).columns.tolist()
for i in categoricals_column:
    print(i,":",type(i))

#One-hot encoding
encoder_ = OneHotEncoder(sparse_output=False)
col_encoded = encoder_.fit_transform(dataset[categoricals_column]).astype(int)
one_hote_Encoded = pd.DataFrame(col_encoded, columns=encoder_.get_feature_names_out(categoricals_column))
#Combine encoded and non-encoded dataframe
dataset_encoded = pd.concat([dataset.drop(columns=(categoricals_column),axis=1), one_hote_Encoded], axis=1)
print(dataset_encoded.head())

#Splitting data
x = dataset_encoded.drop(columns=['persistency'])
y = dataset_encoded['persistency']

#Splitting  data into features target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
#Oversampling
smote = SMOTE()
x_train_sm , y_train_sm = smote.fit_resample(x_train, y_train)

#Model 1
classifier = SVC(kernel='linear')
classifier_2 = SVC(kernel='rbf')
#Training model
classifier.fit(x_train_sm, y_train_sm)
classifier_2.fit(x_train_sm, y_train_sm)
#Predicting 
class_predicted = classifier.predict(x_test)
class_predicted_2 = classifier_2.predict(x_test)

#Model 2
classifer_3 = LogisticRegression(penalty='l1', solver='liblinear')
classifer_3.fit(x_train_sm, y_train_sm)
#Predicting
class_predicted_3 = classifer_3.predict(x_test)

#Matrice de confusion
matrix = confusion_matrix(y_test, class_predicted)
matrix_2 = confusion_matrix(y_test, class_predicted_2)
matrix_3 = confusion_matrix(y_test, class_predicted_3)

#Visualisation des matrices de confusion
sns.heatmap(matrix,annot=True, fmt='d')
plt.title("Matrice de confusion (kernel='Linear')")
plt.xlabel("Prédiction")
plt.ylabel("Valeur réelle")
plt.show()

sns.heatmap(matrix_2, annot=True, fmt='g')
plt.title("Matrice de confusion (kernel='rbf')")
plt.xlabel("Prédiction")
plt.ylabel("Valeur réelle")
plt.show()

sns.heatmap(matrix_3, annot=True, fmt='g')
plt.title("Matrice de confusion (Logistic Regression)")
plt.xlabel("Prédiction")
plt.ylabel("Valeur réelle")
plt.show()

#Rapport des metriques
rapport = classification_report(y_test, class_predicted)
rapport_2 = classification_report(y_test, class_predicted_2)
print("Classification Report (kernel='poly'):\n", rapport)
print("Classification Report (kernel='rbf'):\n", rapport_2)