import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing 
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, classification_report

# Load dataset
dataset =pd.read_csv("/Users/apple/Desktop/ML_Algorithms/SVM/bank-additional-full.csv", delimiter=";")

#Exploring data
print(dataset.shape)
print(dataset.info())
print(dataset['y'].value_counts())
#Data preprocessing
percentage = dataset["y"].value_counts(normalize=True)
print("Pourcentage des classes:", percentage.round(2))
dataset.y.value_counts().plot.bar(color=['red' , 'green'])
plt.show()

print(dataset['job'].unique())
print("Job:\n", dataset['job'].describe(),"\n")

#Collecting categorials and numericals columns
numericals_columns = [col for col in dataset.columns 
                      if dataset[col].dtypes !='object']
print(numericals_columns)

#Splitting in features-target
features = dataset.drop(columns='y')
target = dataset['y']


#Scaling numericals
scaler = MinMaxScaler(feature_range=(1,2))
features[numericals_columns] =scaler.fit_transform(features[numericals_columns].round(2))
print(features)

#Encoding categoricals
for col in dataset.columns:
    if dataset[col].dtypes== 'object' and col != 'y':
        encoder = LabelEncoder()
        features[col] = encoder.fit_transform(features[col])
print(features)

#Mapping the target
print(dataset['y'].nunique())
dataset['y'] = dataset['y'].map({"no":0, "yes":1})

#Splitting the data
x_train , x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42) 
print(x_train.shape, y_train.shape)

#Oversampling
smote  = SMOTE(random_state=1)
x_tran_sm , y_train_sm = smote.fit_resample(x_train, y_train)
 
#Model
classifier = SVC(kernel='rbf')

#Training model
classifier.fit(x_tran_sm, y_train_sm)

#Prediction 
values_predict = classifier.predict(x_test)

#Matrice de confusion
matrix = confusion_matrix(y_test, values_predict)
fig = sns.heatmap(data=matrix, annot=True, fmt='g')
fig.set_title('Matrice de confusion')
plt.show()

#Rapport des metriques 
rapport = classification_report(y_test, values_predict)
print("Classification Report:\n", rapport)



