import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import MinMaxScaler

# Load dataset
dataset = pd.read_csv("/Users/apple/Desktop/ML_Algorithms/REGRESSION LOGISTIQUE/winequality-red.csv")
print(dataset.head(), "\n\n")

#Exploring data
print(dataset.shape)
print(dataset.info())
print(dataset.isnull().sum())

#Data preprocessing
bins = [0,4,6,10]  #(0-4 = 0) (5-7 = 1) (8-10 = 2)
label =[0,1,2]
dataset["quality"]= pd.cut(dataset["quality"],bins= bins, labels=label)
val_uniq = dataset['quality'].nunique()
print("Nombres de valeurs uniques:", val_uniq)
print(dataset)

#visualisation 
sns.set_style('ticks')
sns.countplot(data=dataset , x='quality' , hue='quality')
plt.title("Total par qualité de vin")
plt.show()

#Feature engineering
features = dataset.drop(columns=['quality'])
target = dataset['quality']
scaler = MinMaxScaler(feature_range=(0.5,5))
features[features.columns] = scaler.fit_transform(features[features.columns])
print(features.head(), "\n\n")

#Splitting the dataset
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

#Cross-Validation
classifier = LogisticRegression(max_iter=10000)
stfold = 10
strati_fold = StratifiedGroupKFold(n_splits=stfold, shuffle=True, random_state=42)
cross_val = cross_val_score(classifier, features,target, cv=stfold)
print(f'Cross-Validation Results (Accuracy): {cross_val}')
print(f'Cross_validation moyenne: {cross_val.mean()}')

#Training the model
training_model = classifier.fit(x_train, y_train)

#Prediction
values_predicted = training_model.predict(x_test)
print("\nLes valeurs prédites :\n",values_predicted)

#Evaluation of the model
precision = accuracy_score(y_test, values_predicted)
print("Accuracy Score:", precision)

recall  = recall_score(y_test, values_predicted, average='weighted')
print("Recall Score:", recall)

F1_score = f1_score(y_test, values_predicted, average='weighted')
print("F1 Score:", F1_score,"\n")

#Confusion Matrix
matrix = confusion_matrix(y_test, values_predicted)
print("Confusion Matrix:\n",matrix)
#Heatmap
fig = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=[0,1,2])
fig.plot()
plt.show()


