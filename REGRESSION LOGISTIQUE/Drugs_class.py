import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
 

# Load dataset
dataset = pd.read_csv("/Users/apple/Desktop/ML_Algorithms/REGRESSION LOGISTIQUE/drug200.csv")
print(dataset.head(), "\n\n")
#Exploring data
print(dataset.info())
print(dataset.isnull().sum())
print(dataset.duplicated().sum())
uniq_vals = dataset.nunique()
print(uniq_vals)

#Data preprocessing
#Visualisation
sns.set_style('darkgrid')
sns.countplot(data=dataset,x=dataset['Drug'], hue='Drug')
plt.title("Total par type de drogue")
plt.show()

sns.set_style('whitegrid')
sns.countplot(data=dataset, x=dataset["Sex"],hue='Sex', palette='rocket')
plt.title("Total par sexe")
plt.show()

sns.set_style('darkgrid')
sns.countplot(data=dataset,x= 'BP', hue='BP')
plt.title("Répartition des pression artérielle par type de pression")
plt.show()


#Feature engineering
columns_to_encode = [x for x in dataset.columns if dataset[x].dtypes == 'object']
print("--->",columns_to_encode,"\n")
encoder =LabelEncoder()
for col in columns_to_encode:
    dataset[col]=encoder.fit_transform(dataset[col])
scaler =MinMaxScaler(feature_range=(1,5))
columns_to_scale = ['Age', 'Na_to_K']
dataset[columns_to_scale]= scaler.fit_transform(dataset[columns_to_scale]).round(2)
print(pd.concat([dataset.head(),dataset.tail()]))

#Splitting the dataset
features= dataset.drop(columns=['Drug'])
target = dataset['Drug']

#Cross validation
classifier = LogisticRegression()
fold_numb = 10
k_fold= KFold(n_splits=fold_numb, shuffle=True,random_state=42)
cross_val= cross_val_score(classifier,features,target,cv=fold_numb)
print(f'Cross-Validation score (Accuracy): {cross_val}')
print(f'Cross_validation moyenne: {cross_val.mean()}')

#Splitting the dataset into train and test
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

#Training the model
model = classifier.fit(x_train,y_train)
print("Ordonnée à l'origine:", model.intercept_)
print("Coefficients du modèle:", model.coef_)

#Predicting on test dataset
values_predicted = classifier.predict(x_test)
print("\nLes valeurs prédites :\n",values_predicted,"\n")

#Comparing actual and predicted values
tab = pd.DataFrame({
    'True_values':y_test,
    'Predicted_values':values_predicted
})
print(tab,"\n")

#Evaluation of the model
precision = accuracy_score(y_test, values_predicted)
print("Accuracy Score:", precision)

recall  = recall_score(y_test, values_predicted,average='weighted')
print("Recall Score:", recall)

F1_score = f1_score(y_test, values_predicted,average='weighted')
print("F1 Score:", F1_score,"\n")

#Confusion Matrix
matrix = confusion_matrix(y_test, values_predicted)
print("Confusion Matrix:\n",matrix)
#Heatmap
fig = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=[0,1,2,3,4])
fig.plot()
plt.show()






