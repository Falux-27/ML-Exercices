import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.linear_model import  LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing 
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report

# Chargement des données
dataset =pd.read_csv("/Users/apple/Desktop/ML_Algorithms/SVM/income_evaluation.csv")
print(dataset.head())

# Exploring data
print(dataset.info(),"\n")
print(dataset.describe(),"\n")
print(dataset.shape,"\n")
print(dataset.columns,"\n")
print(dataset[' income'].unique(),"\n")
dataset.drop([' fnlwgt'],axis=1,  inplace=True)
#Remove space in column's names
dataset.columns = dataset.columns.str.replace(' ', '')
print(dataset.columns,"\n")
#checking for null values
print(dataset.isnull().sum())

#Data Preprocessing

#Adding new the categorical column
intervalle = [15, 20, 60, 90]
tranche = ['jeune','adulte', 'vieux']
dataset['tranche_d\'age']= pd.cut(dataset['age'], bins=intervalle, labels=tranche)
dataset['income'] = dataset['income'].str.replace(' ', '')

#Stripping all the spaces of the columns using str.strip()
edu_uniq_vals = dataset['education'].unique()
dataset['education'] = dataset['education'].str.strip()
dataset['workclass']= dataset['workclass'].str.strip()
print(dataset['workclass'].unique(),"\n")

#Data visualization

#pie
f,ax=plt.subplots(1,2,figsize=(5,5))
ax[0] = dataset['income'].value_counts().plot.pie(explode=[0,0],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Income Share')
#Countplot
ax[1] = sns.countplot(x="income", data= dataset, palette="Set1")
ax[1].set_title("Frequency distribution of income variable")
plt.show()

#Income with race
ax = sns.countplot(x="income", hue="race", data=dataset, palette="Set1")
ax.set_title("Frequency distribution of income variable about race")
plt.show()

#Income with Marital status
axe =sns.countplot(data=dataset, x ='income', hue='marital-status', palette='Set1')
axe.set_title("Frequency distribution of income variable about marital status")
plt.show()

#Income with education
axe = sns.countplot(data= dataset, x='income', hue='education', palette='Set1')
axe.set_title("Frequency distribution of income variable about education")
plt.show()

for col in numericals_col: # type: ignore
    sns.histplot(x=dataset[col])
    plt.title("Histogramme de  {}".format(col))
    plt.show()

#Changing income values
low_income = (1000 ,50000)  # Plage pour '<=50K'
high_income=(55000,250000)  # Plage pour '>50K'
def generate_Income(cat):
    if cat =='>50K':
        return np.random.randint(high_income[0], high_income[1])
    elif cat =='<=50K':
        return np.random.randint(low_income[0],low_income[1])
#Remove space in values
dataset['income'] = dataset['income'].map(generate_Income)
print(dataset['income'])
#Findings Categorical Values and Numerical Values
categoricals_col = [cat for cat in dataset.columns if dataset[cat].dtype in ['object', 'category']]
print('There are {} categorical columns \n'.format(len(categoricals_col)))
print('The categorical columns:\n',categoricals_col,"\n")
numericals_col = dataset.select_dtypes(include='int64').columns.tolist()
print('There are {} numerical columns \n'.format(len(numericals_col)),"\n")
print('The numerical columns:\n',numericals_col,"\n\n")
print(dataset.info())
#Label encoding
encoder = LabelEncoder()
for col in dataset.columns:
    if dataset[col].dtype =='object' or 'category':
        dataset[col] = encoder.fit_transform(dataset[col])
print(dataset.head())

#Normalizing the variables
scaler = MinMaxScaler()
numericals_col=[]
for col in dataset.columns:
  if dataset[col].dtype not in categoricals_col:
      if col != 'income' and col != "tranche_d'age":
        numericals_col.append(col)
print(numericals_col)
dataset[numericals_col] = scaler.fit_transform(dataset[numericals_col])
print(dataset.head())

#Features - Target
feature = dataset.drop('income', axis=1)
target = dataset['income']

#Splitting the dataset into train test
x_train, x_test, y_train, y_test = train_test_split(feature, target, test_size=0.2, random_state=42)

#Model
