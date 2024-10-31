import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split ,cross_val_score , StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC

#Load dataset
dataset = pd.read_csv("/Users/apple/Desktop/ML_Algorithms/SVM/loan_prediction.csv")

#Exploring data
print(dataset.columns, "\n")
print(dataset.shape)
print(dataset.info())
print(dataset.isnull().sum())

#Data preprocessing

#Fill missing values in numerical columns
for x in dataset.columns:
    if dataset[x].dtype in ['int64', 'float64'] and dataset[x].isnull().sum() > 0:
        dataset[x].fillna(dataset[x].mean(), inplace=True)

#Visualisation
#Figure 1
val_count = dataset['Loan_Status'].value_counts(normalize=True).round(2)
sns.barplot(x=val_count.index, y=val_count.values, hue=val_count.unique())
plt.title("Loan Status Distribution")
plt.show()

#Figure 2
gender_count= dataset['Gender'].value_counts()
sns.barplot(x=gender_count.index, y=gender_count.values, hue=gender_count.unique())
plt.title("Gender Distribution")
plt.show()

#Figure 3
married_count = dataset['Married'].value_counts()
sns.barplot(x =married_count.values, y=married_count.index, hue=married_count.unique())
plt.title("Marital Statut Distribution")
plt.show()

#Figure 4
education_count = dataset['Education'].value_counts()
sns.barplot(x=education_count.index, y=education_count.values, hue=education_count.unique())
plt.title("Education Distribution")
plt.show()

#Figure 5
sns.set_style('darkgrid')
sns.histplot(data=dataset, x='ApplicantIncome')
plt.title("Applicant Income Distribution")
plt.xticks(rotation = 45)
plt.show()

#Figure 6
sns.set_style('whitegrid')
sns.histplot(data=dataset, x='Credit_History',y='Loan_Status', hue='Loan_Status')
plt.title("Credit History Distribution")
plt.show()

#Splitting in features - target
features = dataset.drop(columns=['Loan_Status'])
target = dataset['Loan_Status']

#Encoding and Scaling
#Scaling
numericals_columns = features.select_dtypes(include=['int64', 'float64']).columns
print("Numericals columns:", numericals_columns)
scaler = StandardScaler()
features[numericals_columns] = scaler.fit_transform(features[numericals_columns])

#Encoding
columns_to_encode = features.select_dtypes(include=['object']).columns
encoder = OneHotEncoder(sparse_output=False)
data_encoded = encoder.fit_transform(features[columns_to_encode])
datafrm_encoded = pd.DataFrame(data_encoded, columns=encoder.get_feature_names_out(columns_to_encode))
features = features.drop(columns=columns_to_encode)
features = pd.concat([features, datafrm_encoded], axis=1)

#Handle missing values in numerical columns (if any)
for column in numericals_columns:
    features[column].fillna(value=features[column].mean(), inplace=True)

#Target encoding
encoder_target = LabelEncoder()
target = encoder_target.fit_transform(target)
print(target)

#Splitting the dataset
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

#Oversampling
smote = SMOTE(random_state=1)
x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)

#Logistic Regression model
model = LogisticRegression(penalty='l1', solver='liblinear')
model.fit(x_train_smote, y_train_smote)
#Evaluation of the model on test data
class_predicted = model.predict(x_test)
#Confusion matrix
matrix = confusion_matrix(y_test, class_predicted)
print("Matrice de confusion:\n", matrix)
rapport = classification_report(y_test, class_predicted)
print("Metriques Logistic Regression:\n",rapport)
# Heatmap
sns.heatmap(data=matrix, annot=True, fmt='g')
plt.title("Matrice de confusion - Logistic Regression")
plt.xlabel("Prédiction")
plt.ylabel("Valeur réelle")
plt.show()
 
#SVM models by Cross validation
kernel = ['linear', 'poly', 'rbf']
for noyau in kernel:
    model = SVC(kernel= noyau, random_state=42)
    score = cross_val_score(model, features, target, cv=5)
    print(f"Évaluation pour le noyau {noyau}:")
    print(f"Précision moyenne : {score.mean():.2f}")
    print(f"Écart-type de la précision : {score.std():.2f}","\n")

#SVM model with best parameters
model = SVC(kernel='rbf', C=10, gamma=0.01, random_state=42)
model.fit(x_train_smote, y_train_smote)
#Evaluation of the model on test data
class_predicted = model.predict(x_test)
#Confusion matrix
matrix= confusion_matrix(y_test, class_predicted)
print("Matrice de confusion:\n", matrix)
#Heatmap 
sns.heatmap(matrix, annot=True, fmt='g')
plt.title("Matrice de confusion - SVM avec rbf")
plt.xlabel("Prédiction")
plt.ylabel("Valeur réelle")
plt.show()

#Classification report
rapport = classification_report(y_test, class_predicted)
print("Metriques SVM avec rbf:\n", rapport)

