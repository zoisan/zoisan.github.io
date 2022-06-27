# Titanic - Machine Learning from Disaster

ismailsavruk@gmail.com © 2022

I'll use machine learning to create a model that predicts which passengers survived the Titanic shipwreck.

Dataset from: https://www.kaggle.com/competitions/titanic/data

## 1. Importing  Libraries and Loading Dataset

Let's start by importing necessary libraries first.


```python
# Load libraries
import numpy as np

import pandas as pd
from pandas import read_csv
from pandas.plotting import scatter_matrix

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, RepeatedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn import linear_model
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn import svm 
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


import matplotlib.pyplot as plt

%matplotlib inline
```


```python
import warnings
warnings.filterwarnings('ignore')
```


```python
# Load dataset
titanic = pd.read_csv('Train.csv')
```

Let's take a look at the dataset.


```python
titanic
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>886</th>
      <td>887</td>
      <td>0</td>
      <td>2</td>
      <td>Montvila, Rev. Juozas</td>
      <td>male</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>211536</td>
      <td>13.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>887</th>
      <td>888</td>
      <td>1</td>
      <td>1</td>
      <td>Graham, Miss. Margaret Edith</td>
      <td>female</td>
      <td>19.0</td>
      <td>0</td>
      <td>0</td>
      <td>112053</td>
      <td>30.0000</td>
      <td>B42</td>
      <td>S</td>
    </tr>
    <tr>
      <th>888</th>
      <td>889</td>
      <td>0</td>
      <td>3</td>
      <td>Johnston, Miss. Catherine Helen "Carrie"</td>
      <td>female</td>
      <td>NaN</td>
      <td>1</td>
      <td>2</td>
      <td>W./C. 6607</td>
      <td>23.4500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>889</th>
      <td>890</td>
      <td>1</td>
      <td>1</td>
      <td>Behr, Mr. Karl Howell</td>
      <td>male</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>111369</td>
      <td>30.0000</td>
      <td>C148</td>
      <td>C</td>
    </tr>
    <tr>
      <th>890</th>
      <td>891</td>
      <td>0</td>
      <td>3</td>
      <td>Dooley, Mr. Patrick</td>
      <td>male</td>
      <td>32.0</td>
      <td>0</td>
      <td>0</td>
      <td>370376</td>
      <td>7.7500</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
  </tbody>
</table>
<p>891 rows × 12 columns</p>
</div>



### Content


survival:	Survival	0 = No, 1 = Yes

pclass:	Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd

sex:	Sex	

Age:	Age in years	

sibsp:	# of siblings / spouses aboard the Titanic	

parch:	# of parents / children aboard the Titanic	

ticket:	Ticket number	

fare:	Passenger fare	

cabin:	Cabin number	

embarked:	Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton

### Dropping some columns

When we look at the table, I don't think, the Passanger ID, Name, Ticket no or cabin info will have an effect on survival. So we can drop them.


```python
titanic_new = titanic.drop(columns=['PassengerId','Name', 'Ticket', 'Cabin'])
titanic_new
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>S</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>886</th>
      <td>0</td>
      <td>2</td>
      <td>male</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>13.0000</td>
      <td>S</td>
    </tr>
    <tr>
      <th>887</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>19.0</td>
      <td>0</td>
      <td>0</td>
      <td>30.0000</td>
      <td>S</td>
    </tr>
    <tr>
      <th>888</th>
      <td>0</td>
      <td>3</td>
      <td>female</td>
      <td>NaN</td>
      <td>1</td>
      <td>2</td>
      <td>23.4500</td>
      <td>S</td>
    </tr>
    <tr>
      <th>889</th>
      <td>1</td>
      <td>1</td>
      <td>male</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>30.0000</td>
      <td>C</td>
    </tr>
    <tr>
      <th>890</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>32.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.7500</td>
      <td>Q</td>
    </tr>
  </tbody>
</table>
<p>891 rows × 8 columns</p>
</div>



### Rearranging Columns

Let's rearrange the column values so that Survived would be the last column since it will be our Y values.


```python
titanic_new = titanic_new[['Pclass',  'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked',  'Survived']]
```


```python
titanic_new
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>S</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>C</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>S</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>S</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>S</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>886</th>
      <td>2</td>
      <td>male</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>13.0000</td>
      <td>S</td>
      <td>0</td>
    </tr>
    <tr>
      <th>887</th>
      <td>1</td>
      <td>female</td>
      <td>19.0</td>
      <td>0</td>
      <td>0</td>
      <td>30.0000</td>
      <td>S</td>
      <td>1</td>
    </tr>
    <tr>
      <th>888</th>
      <td>3</td>
      <td>female</td>
      <td>NaN</td>
      <td>1</td>
      <td>2</td>
      <td>23.4500</td>
      <td>S</td>
      <td>0</td>
    </tr>
    <tr>
      <th>889</th>
      <td>1</td>
      <td>male</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>30.0000</td>
      <td>C</td>
      <td>1</td>
    </tr>
    <tr>
      <th>890</th>
      <td>3</td>
      <td>male</td>
      <td>32.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.7500</td>
      <td>Q</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>891 rows × 8 columns</p>
</div>



### Encoding

Let's replace females with "0" and males with "1". I will use map method instead of ordinal encoder or one hot encoder. 


```python
titanic_new['Sex'] = titanic_new['Sex'].map({'female': 0, 'male': 1})
```


```python
titanic_new
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>1</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>S</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>C</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>S</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>S</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>1</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>S</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>886</th>
      <td>2</td>
      <td>1</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>13.0000</td>
      <td>S</td>
      <td>0</td>
    </tr>
    <tr>
      <th>887</th>
      <td>1</td>
      <td>0</td>
      <td>19.0</td>
      <td>0</td>
      <td>0</td>
      <td>30.0000</td>
      <td>S</td>
      <td>1</td>
    </tr>
    <tr>
      <th>888</th>
      <td>3</td>
      <td>0</td>
      <td>NaN</td>
      <td>1</td>
      <td>2</td>
      <td>23.4500</td>
      <td>S</td>
      <td>0</td>
    </tr>
    <tr>
      <th>889</th>
      <td>1</td>
      <td>1</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>30.0000</td>
      <td>C</td>
      <td>1</td>
    </tr>
    <tr>
      <th>890</th>
      <td>3</td>
      <td>1</td>
      <td>32.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.7500</td>
      <td>Q</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>891 rows × 8 columns</p>
</div>



Let's do the similar thing for "Embarked". New values would be C=0, Q=1, S=2


```python
titanic_new['Embarked'] = titanic_new['Embarked'].map({'C':0, 'Q':1, 'S': 2})
```


```python
titanic_new
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>1</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>2.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>2.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>2.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>1</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>2.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>886</th>
      <td>2</td>
      <td>1</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>13.0000</td>
      <td>2.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>887</th>
      <td>1</td>
      <td>0</td>
      <td>19.0</td>
      <td>0</td>
      <td>0</td>
      <td>30.0000</td>
      <td>2.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>888</th>
      <td>3</td>
      <td>0</td>
      <td>NaN</td>
      <td>1</td>
      <td>2</td>
      <td>23.4500</td>
      <td>2.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>889</th>
      <td>1</td>
      <td>1</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>30.0000</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>890</th>
      <td>3</td>
      <td>1</td>
      <td>32.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.7500</td>
      <td>1.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>891 rows × 8 columns</p>
</div>



### Duplicates


```python
#identify duplicate rows
duplicateRows = titanic_new[titanic_new.duplicated()]

#view duplicate rows
duplicateRows
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>47</th>
      <td>3</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>7.7500</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>76</th>
      <td>3</td>
      <td>1</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>7.8958</td>
      <td>2.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>77</th>
      <td>3</td>
      <td>1</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>2.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>87</th>
      <td>3</td>
      <td>1</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>2.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>95</th>
      <td>3</td>
      <td>1</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>2.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>870</th>
      <td>3</td>
      <td>1</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.8958</td>
      <td>2.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>877</th>
      <td>3</td>
      <td>1</td>
      <td>19.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.8958</td>
      <td>2.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>878</th>
      <td>3</td>
      <td>1</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>7.8958</td>
      <td>2.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>884</th>
      <td>3</td>
      <td>1</td>
      <td>25.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.0500</td>
      <td>2.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>886</th>
      <td>2</td>
      <td>1</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>13.0000</td>
      <td>2.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>111 rows × 8 columns</p>
</div>




```python
titanic_new = titanic_new.drop_duplicates()
titanic_new.shape
```




    (780, 8)



We now have 780 rows after removing duplicate values.

## 2. Train and Test Data Split

It is very important to split the test data before data preprocessing steps to avoid data leakage!

We will set aside our test data and only use training data first. Then we will take similar preprocessing steps for test data.


```python
# X values
X = titanic_new.drop(columns='Survived')
X
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>1</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>1</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>885</th>
      <td>3</td>
      <td>0</td>
      <td>39.0</td>
      <td>0</td>
      <td>5</td>
      <td>29.1250</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>887</th>
      <td>1</td>
      <td>0</td>
      <td>19.0</td>
      <td>0</td>
      <td>0</td>
      <td>30.0000</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>888</th>
      <td>3</td>
      <td>0</td>
      <td>NaN</td>
      <td>1</td>
      <td>2</td>
      <td>23.4500</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>889</th>
      <td>1</td>
      <td>1</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>30.0000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>890</th>
      <td>3</td>
      <td>1</td>
      <td>32.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.7500</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>780 rows × 7 columns</p>
</div>




```python
#Target, y values
Y = titanic_new['Survived']
Y
```




    0      0
    1      1
    2      1
    3      1
    4      0
          ..
    885    0
    887    1
    888    0
    889    1
    890    0
    Name: Survived, Length: 780, dtype: int64




```python
#train and test data split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, random_state=42)

(X_train.shape[0], X_test.shape[0])
```




    (624, 156)



So we have 624 data points in the training set and 156 on the test set.


```python
#create a dataframe for training set including target variables.
titanic_train = X_train.copy()

#adding a 'survived column'
titanic_train['Survived'] = Y_train

titanic_train = titanic_train.reset_index(drop=True)

#diamonds_train has both X_train and Y_train values.
titanic_train
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>46.0</td>
      <td>0</td>
      <td>0</td>
      <td>79.2000</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>34.0</td>
      <td>0</td>
      <td>0</td>
      <td>13.0000</td>
      <td>2.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0</td>
      <td>22.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.7500</td>
      <td>2.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>1</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>7.7292</td>
      <td>1.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>0</td>
      <td>21.0</td>
      <td>0</td>
      <td>0</td>
      <td>10.5000</td>
      <td>2.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>619</th>
      <td>2</td>
      <td>1</td>
      <td>21.0</td>
      <td>0</td>
      <td>0</td>
      <td>73.5000</td>
      <td>2.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>620</th>
      <td>3</td>
      <td>1</td>
      <td>22.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>2.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>621</th>
      <td>3</td>
      <td>1</td>
      <td>22.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.8958</td>
      <td>2.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>622</th>
      <td>3</td>
      <td>0</td>
      <td>63.0</td>
      <td>0</td>
      <td>0</td>
      <td>9.5875</td>
      <td>2.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>623</th>
      <td>3</td>
      <td>1</td>
      <td>38.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.8958</td>
      <td>2.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>624 rows × 8 columns</p>
</div>




```python
#create a dataframe for training set including target variables.
titanic_test = X_test.copy()

#adding a 'survived column'
titanic_test['Survived'] = Y_test

titanic_test = titanic_test.reset_index(drop=True)

#diamonds_train has both X_train and Y_train values.
titanic_test
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>1</td>
      <td>24.5</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>2.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>1</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>7.7750</td>
      <td>2.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>7.0500</td>
      <td>2.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>1</td>
      <td>25.0</td>
      <td>1</td>
      <td>0</td>
      <td>26.0000</td>
      <td>2.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1</td>
      <td>64.0</td>
      <td>0</td>
      <td>0</td>
      <td>26.0000</td>
      <td>2.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>151</th>
      <td>3</td>
      <td>1</td>
      <td>32.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>2.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>152</th>
      <td>3</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>7.7875</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>153</th>
      <td>3</td>
      <td>0</td>
      <td>16.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.7333</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>154</th>
      <td>3</td>
      <td>0</td>
      <td>45.0</td>
      <td>0</td>
      <td>1</td>
      <td>14.4542</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>155</th>
      <td>1</td>
      <td>0</td>
      <td>50.0</td>
      <td>0</td>
      <td>0</td>
      <td>28.7125</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>156 rows × 8 columns</p>
</div>



## 3. Handling Missing Values

#### Training Data


```python
# Checking for missing values
titanic_train.isna().sum()
```




    Pclass       0
    Sex          0
    Age         80
    SibSp        0
    Parch        0
    Fare         0
    Embarked     2
    Survived     0
    dtype: int64



We are missing 2 Embarked values and 80 age values. 

Let's start with Embarked column. Since it has categorical values, we can use "Mode" to replace NaN values.


```python
titanic_train.Embarked.mode()
```




    0    2.0
    dtype: float64




```python
titanic_train['Embarked'] = titanic_train['Embarked'].fillna(2)
titanic_train.isna().sum()
```




    Pclass       0
    Sex          0
    Age         80
    SibSp        0
    Parch        0
    Fare         0
    Embarked     0
    Survived     0
    dtype: int64



Great! Let's deal with missing age values.

Since age values are numerical, we can use Mean value to replace all NaN values.


```python
#Since this is an age value, let's round to a nearest whole number.
mean_value=round(titanic_train.Age.mean())
mean_value
```




    30




```python
titanic_train['Age'].fillna(value=mean_value, inplace=True)
titanic_train.isna().sum()
```




    Pclass      0
    Sex         0
    Age         0
    SibSp       0
    Parch       0
    Fare        0
    Embarked    0
    Survived    0
    dtype: int64



Perfect! No missing values on the training set. We need to fill the missing values in the test set separately if any.

Let's not deal with removing outliers since we have a small dataset.

#### Test Data


```python
# Checking for missing values
titanic_test.isna().sum()
```




    Pclass       0
    Sex          0
    Age         24
    SibSp        0
    Parch        0
    Fare         0
    Embarked     0
    Survived     0
    dtype: int64



We are missing 24 age values only. Since age values are numerical, we can use Mean value to replace all NaN values.


```python
#Since this is an age value, let's round to a nearest whole number.
mean_value=round(titanic_test.Age.mean())
mean_value
```




    28




```python
titanic_test['Age'].fillna(value=mean_value, inplace=True)
titanic_test.isna().sum()
```




    Pclass      0
    Sex         0
    Age         0
    SibSp       0
    Parch       0
    Fare        0
    Embarked    0
    Survived    0
    dtype: int64



Perfect! No missing values on the test set.

## 4. EDA and Visualizations

Now we can take a look at a summary of each attribute.

This includes the count, mean, the min and max values as well as some percentiles.


```python
# descriptions
titanic_train.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>624.000000</td>
      <td>624.000000</td>
      <td>624.000000</td>
      <td>624.000000</td>
      <td>624.000000</td>
      <td>624.000000</td>
      <td>624.000000</td>
      <td>624.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2.232372</td>
      <td>0.629808</td>
      <td>30.182837</td>
      <td>0.536859</td>
      <td>0.427885</td>
      <td>35.026122</td>
      <td>1.528846</td>
      <td>0.411859</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.851122</td>
      <td>0.483243</td>
      <td>13.926845</td>
      <td>1.001726</td>
      <td>0.846819</td>
      <td>53.060198</td>
      <td>0.806754</td>
      <td>0.492565</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.420000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>22.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>8.050000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>30.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>16.400000</td>
      <td>2.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>37.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>34.444800</td>
      <td>2.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>80.000000</td>
      <td>8.000000</td>
      <td>6.000000</td>
      <td>512.329200</td>
      <td>2.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



Now we can take a look at the correlation matrix to summarize the correlations between all variables in the dataset. It's also important because it serves as a diagnostic for regression.


```python
#correlation matrix
titanic_train.corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Pclass</th>
      <td>1.000000</td>
      <td>0.084600</td>
      <td>-0.355636</td>
      <td>0.103839</td>
      <td>0.055580</td>
      <td>-0.532171</td>
      <td>0.183078</td>
      <td>-0.316713</td>
    </tr>
    <tr>
      <th>Sex</th>
      <td>0.084600</td>
      <td>1.000000</td>
      <td>0.101837</td>
      <td>-0.096112</td>
      <td>-0.228123</td>
      <td>-0.138926</td>
      <td>0.066549</td>
      <td>-0.511563</td>
    </tr>
    <tr>
      <th>Age</th>
      <td>-0.355636</td>
      <td>0.101837</td>
      <td>1.000000</td>
      <td>-0.281101</td>
      <td>-0.189385</td>
      <td>0.101211</td>
      <td>-0.039014</td>
      <td>-0.060346</td>
    </tr>
    <tr>
      <th>SibSp</th>
      <td>0.103839</td>
      <td>-0.096112</td>
      <td>-0.281101</td>
      <td>1.000000</td>
      <td>0.374014</td>
      <td>0.111576</td>
      <td>0.059261</td>
      <td>-0.045455</td>
    </tr>
    <tr>
      <th>Parch</th>
      <td>0.055580</td>
      <td>-0.228123</td>
      <td>-0.189385</td>
      <td>0.374014</td>
      <td>1.000000</td>
      <td>0.172944</td>
      <td>0.048866</td>
      <td>0.080942</td>
    </tr>
    <tr>
      <th>Fare</th>
      <td>-0.532171</td>
      <td>-0.138926</td>
      <td>0.101211</td>
      <td>0.111576</td>
      <td>0.172944</td>
      <td>1.000000</td>
      <td>-0.255712</td>
      <td>0.226766</td>
    </tr>
    <tr>
      <th>Embarked</th>
      <td>0.183078</td>
      <td>0.066549</td>
      <td>-0.039014</td>
      <td>0.059261</td>
      <td>0.048866</td>
      <td>-0.255712</td>
      <td>1.000000</td>
      <td>-0.157184</td>
    </tr>
    <tr>
      <th>Survived</th>
      <td>-0.316713</td>
      <td>-0.511563</td>
      <td>-0.060346</td>
      <td>-0.045455</td>
      <td>0.080942</td>
      <td>0.226766</td>
      <td>-0.157184</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#seaborn package
import seaborn as sns
plt.rcParams["figure.figsize"]=(12,12)# Custom figure size in inches
sns.heatmap(titanic_train.corr(), annot=True)
plt.title("Correlation Matrix")
```




    Text(0.5, 1.0, 'Correlation Matrix')




    
![png](Titanic_files/Titanic_52_1.png)
    



```python
# box and whisker plots
titanic_train.plot(kind='box', subplots=True, layout=(2,4), sharex=False, sharey=False, figsize=[16,10])
plt.show()
```


    
![png](Titanic_files/Titanic_53_0.png)
    



```python
# histogram
titanic_train.hist(alpha=0.8, rwidth=0.9, figsize=[15,15])
plt.show()
```


    
![png](Titanic_files/Titanic_54_0.png)
    



```python
# scatter plot matrix
scatter_matrix(titanic_train, figsize=[15,15])
plt.show()
```


    
![png](Titanic_files/Titanic_55_0.png)
    



```python
sns.pairplot(titanic_train, hue='Survived')
plt.show()
```


    
![png](Titanic_files/Titanic_56_0.png)
    


### Pie Chart


```python
counts = titanic_train.groupby(['Survived','Sex'])['Survived'].count().values
```


```python
label = ['Female Not-Survived','Male Not-Survived','Female Survived','Male Survived']
```


```python
plt.rcParams['figure.figsize'] = [6,6]  # the whole figure
plt.rcParams["axes.titlesize"] = 20     # for the top title
plt.rcParams["xtick.labelsize"] = 15    # for lables
plt.rcParams["font.size"] = 20          # for percentages
```


```python
explode = [0.05]*len(counts)
colors = ['pink','red','royalblue','green']
plt.title("Survival Rates by Gender")
plt.axis("equal")
plt.pie(counts, labels=label, autopct='%1.0f%%', startangle=-60, explode=explode, colors = colors, shadow=True)
plt.plot();
```


    
![png](Titanic_files/Titanic_61_0.png)
    


### Feature Scaling

We will standardize first and normalize next both training and test data separately.

#### Training Data


```python
#Normalize the training data
normal = MinMaxScaler()
titanic_train_normal = normal.fit_transform(titanic_train)
titanic_train_normal = pd.DataFrame(titanic_train_normal, columns = titanic_train.columns)
titanic_train_normal
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.572757</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.154588</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.5</td>
      <td>1.0</td>
      <td>0.421965</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.025374</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.271174</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.015127</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.371701</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.015086</td>
      <td>0.5</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.5</td>
      <td>0.0</td>
      <td>0.258608</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.020495</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>619</th>
      <td>0.5</td>
      <td>1.0</td>
      <td>0.258608</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.143462</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>620</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.271174</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.015713</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>621</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.271174</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.015412</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>622</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.786378</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.018714</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>623</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.472229</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.015412</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>624 rows × 8 columns</p>
</div>




```python
#Standardize the training data that has been normalized
standard = StandardScaler()
titanic_train_scaled = standard.fit_transform(titanic_train_normal)
titanic_train_scaled = pd.DataFrame(titanic_train_scaled, columns = titanic_train.columns)
titanic_train_scaled
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1.449100</td>
      <td>0.766672</td>
      <td>1.136643</td>
      <td>-0.536364</td>
      <td>-0.50569</td>
      <td>0.833192</td>
      <td>-1.896579</td>
      <td>-0.836823</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.273237</td>
      <td>0.766672</td>
      <td>0.274307</td>
      <td>-0.536364</td>
      <td>-0.50569</td>
      <td>-0.415449</td>
      <td>0.584480</td>
      <td>-0.836823</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.902626</td>
      <td>-1.304338</td>
      <td>-0.588030</td>
      <td>-0.536364</td>
      <td>-0.50569</td>
      <td>-0.514472</td>
      <td>0.584480</td>
      <td>1.194996</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.902626</td>
      <td>0.766672</td>
      <td>-0.013139</td>
      <td>-0.536364</td>
      <td>-0.50569</td>
      <td>-0.514865</td>
      <td>-0.656049</td>
      <td>-0.836823</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.273237</td>
      <td>-1.304338</td>
      <td>-0.659891</td>
      <td>-0.536364</td>
      <td>-0.50569</td>
      <td>-0.462603</td>
      <td>0.584480</td>
      <td>1.194996</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>619</th>
      <td>-0.273237</td>
      <td>0.766672</td>
      <td>-0.659891</td>
      <td>-0.536364</td>
      <td>-0.50569</td>
      <td>0.725680</td>
      <td>0.584480</td>
      <td>-0.836823</td>
    </tr>
    <tr>
      <th>620</th>
      <td>0.902626</td>
      <td>0.766672</td>
      <td>-0.588030</td>
      <td>-0.536364</td>
      <td>-0.50569</td>
      <td>-0.508814</td>
      <td>0.584480</td>
      <td>-0.836823</td>
    </tr>
    <tr>
      <th>621</th>
      <td>0.902626</td>
      <td>0.766672</td>
      <td>-0.588030</td>
      <td>-0.536364</td>
      <td>-0.50569</td>
      <td>-0.511722</td>
      <td>0.584480</td>
      <td>-0.836823</td>
    </tr>
    <tr>
      <th>622</th>
      <td>0.902626</td>
      <td>-1.304338</td>
      <td>2.358287</td>
      <td>-0.536364</td>
      <td>-0.50569</td>
      <td>-0.479814</td>
      <td>0.584480</td>
      <td>1.194996</td>
    </tr>
    <tr>
      <th>623</th>
      <td>0.902626</td>
      <td>0.766672</td>
      <td>0.561752</td>
      <td>-0.536364</td>
      <td>-0.50569</td>
      <td>-0.511722</td>
      <td>0.584480</td>
      <td>-0.836823</td>
    </tr>
  </tbody>
</table>
<p>624 rows × 8 columns</p>
</div>




```python
X_train_scaled = titanic_train_scaled.drop(columns='Survived')
X_train_scaled
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1.449100</td>
      <td>0.766672</td>
      <td>1.136643</td>
      <td>-0.536364</td>
      <td>-0.50569</td>
      <td>0.833192</td>
      <td>-1.896579</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.273237</td>
      <td>0.766672</td>
      <td>0.274307</td>
      <td>-0.536364</td>
      <td>-0.50569</td>
      <td>-0.415449</td>
      <td>0.584480</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.902626</td>
      <td>-1.304338</td>
      <td>-0.588030</td>
      <td>-0.536364</td>
      <td>-0.50569</td>
      <td>-0.514472</td>
      <td>0.584480</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.902626</td>
      <td>0.766672</td>
      <td>-0.013139</td>
      <td>-0.536364</td>
      <td>-0.50569</td>
      <td>-0.514865</td>
      <td>-0.656049</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.273237</td>
      <td>-1.304338</td>
      <td>-0.659891</td>
      <td>-0.536364</td>
      <td>-0.50569</td>
      <td>-0.462603</td>
      <td>0.584480</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>619</th>
      <td>-0.273237</td>
      <td>0.766672</td>
      <td>-0.659891</td>
      <td>-0.536364</td>
      <td>-0.50569</td>
      <td>0.725680</td>
      <td>0.584480</td>
    </tr>
    <tr>
      <th>620</th>
      <td>0.902626</td>
      <td>0.766672</td>
      <td>-0.588030</td>
      <td>-0.536364</td>
      <td>-0.50569</td>
      <td>-0.508814</td>
      <td>0.584480</td>
    </tr>
    <tr>
      <th>621</th>
      <td>0.902626</td>
      <td>0.766672</td>
      <td>-0.588030</td>
      <td>-0.536364</td>
      <td>-0.50569</td>
      <td>-0.511722</td>
      <td>0.584480</td>
    </tr>
    <tr>
      <th>622</th>
      <td>0.902626</td>
      <td>-1.304338</td>
      <td>2.358287</td>
      <td>-0.536364</td>
      <td>-0.50569</td>
      <td>-0.479814</td>
      <td>0.584480</td>
    </tr>
    <tr>
      <th>623</th>
      <td>0.902626</td>
      <td>0.766672</td>
      <td>0.561752</td>
      <td>-0.536364</td>
      <td>-0.50569</td>
      <td>-0.511722</td>
      <td>0.584480</td>
    </tr>
  </tbody>
</table>
<p>624 rows × 7 columns</p>
</div>



We do not need to scale Y values since it takes only 0 and 1.

#### Test Data


```python
#Normalize the training data
normal = MinMaxScaler()
titanic_test_normal = normal.fit_transform(titanic_test)
titanic_test_normal = pd.DataFrame(titanic_test_normal, columns = titanic_test.columns)
titanic_test_normal
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.342200</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.030608</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.392800</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.029563</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.392800</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.026806</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.5</td>
      <td>1.0</td>
      <td>0.349429</td>
      <td>0.25</td>
      <td>0.0</td>
      <td>0.098859</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.913257</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.098859</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>151</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.450629</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.030133</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>152</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.392800</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.029610</td>
      <td>0.5</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>153</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.219315</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.029404</td>
      <td>0.5</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>154</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.638572</td>
      <td>0.00</td>
      <td>0.2</td>
      <td>0.054959</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>155</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.710857</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.109173</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>156 rows × 8 columns</p>
</div>




```python
#Standardize the training data that has been normalized
standard = StandardScaler()
titanic_test_scaled = standard.fit_transform(titanic_test_normal)
titanic_test_scaled = pd.DataFrame(titanic_test_scaled, columns = titanic_test.columns)
titanic_test_scaled
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.807363</td>
      <td>0.801315</td>
      <td>-0.287554</td>
      <td>-0.516922</td>
      <td>-0.470766</td>
      <td>-0.531032</td>
      <td>0.595768</td>
      <td>-0.845154</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.807363</td>
      <td>0.801315</td>
      <td>-0.011783</td>
      <td>-0.516922</td>
      <td>-0.470766</td>
      <td>-0.536650</td>
      <td>0.595768</td>
      <td>-0.845154</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.807363</td>
      <td>0.801315</td>
      <td>-0.011783</td>
      <td>-0.516922</td>
      <td>-0.470766</td>
      <td>-0.551463</td>
      <td>0.595768</td>
      <td>-0.845154</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.348129</td>
      <td>0.801315</td>
      <td>-0.248158</td>
      <td>0.558276</td>
      <td>-0.470766</td>
      <td>-0.164289</td>
      <td>0.595768</td>
      <td>-0.845154</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-1.503621</td>
      <td>0.801315</td>
      <td>2.824710</td>
      <td>-0.516922</td>
      <td>-0.470766</td>
      <td>-0.164289</td>
      <td>0.595768</td>
      <td>-0.845154</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>151</th>
      <td>0.807363</td>
      <td>0.801315</td>
      <td>0.303383</td>
      <td>-0.516922</td>
      <td>-0.470766</td>
      <td>-0.533586</td>
      <td>0.595768</td>
      <td>-0.845154</td>
    </tr>
    <tr>
      <th>152</th>
      <td>0.807363</td>
      <td>-1.247949</td>
      <td>-0.011783</td>
      <td>-0.516922</td>
      <td>-0.470766</td>
      <td>-0.536395</td>
      <td>-0.660175</td>
      <td>1.183216</td>
    </tr>
    <tr>
      <th>153</th>
      <td>0.807363</td>
      <td>-1.247949</td>
      <td>-0.957281</td>
      <td>-0.516922</td>
      <td>-0.470766</td>
      <td>-0.537502</td>
      <td>-0.660175</td>
      <td>1.183216</td>
    </tr>
    <tr>
      <th>154</th>
      <td>0.807363</td>
      <td>-1.247949</td>
      <td>1.327672</td>
      <td>-0.516922</td>
      <td>0.773972</td>
      <td>-0.400185</td>
      <td>-1.916118</td>
      <td>-0.845154</td>
    </tr>
    <tr>
      <th>155</th>
      <td>-1.503621</td>
      <td>-1.247949</td>
      <td>1.721629</td>
      <td>-0.516922</td>
      <td>-0.470766</td>
      <td>-0.108869</td>
      <td>-1.916118</td>
      <td>-0.845154</td>
    </tr>
  </tbody>
</table>
<p>156 rows × 8 columns</p>
</div>




```python
X_test_scaled = titanic_test_scaled.drop(columns='Survived')
X_test_scaled
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.807363</td>
      <td>0.801315</td>
      <td>-0.287554</td>
      <td>-0.516922</td>
      <td>-0.470766</td>
      <td>-0.531032</td>
      <td>0.595768</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.807363</td>
      <td>0.801315</td>
      <td>-0.011783</td>
      <td>-0.516922</td>
      <td>-0.470766</td>
      <td>-0.536650</td>
      <td>0.595768</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.807363</td>
      <td>0.801315</td>
      <td>-0.011783</td>
      <td>-0.516922</td>
      <td>-0.470766</td>
      <td>-0.551463</td>
      <td>0.595768</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.348129</td>
      <td>0.801315</td>
      <td>-0.248158</td>
      <td>0.558276</td>
      <td>-0.470766</td>
      <td>-0.164289</td>
      <td>0.595768</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-1.503621</td>
      <td>0.801315</td>
      <td>2.824710</td>
      <td>-0.516922</td>
      <td>-0.470766</td>
      <td>-0.164289</td>
      <td>0.595768</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>151</th>
      <td>0.807363</td>
      <td>0.801315</td>
      <td>0.303383</td>
      <td>-0.516922</td>
      <td>-0.470766</td>
      <td>-0.533586</td>
      <td>0.595768</td>
    </tr>
    <tr>
      <th>152</th>
      <td>0.807363</td>
      <td>-1.247949</td>
      <td>-0.011783</td>
      <td>-0.516922</td>
      <td>-0.470766</td>
      <td>-0.536395</td>
      <td>-0.660175</td>
    </tr>
    <tr>
      <th>153</th>
      <td>0.807363</td>
      <td>-1.247949</td>
      <td>-0.957281</td>
      <td>-0.516922</td>
      <td>-0.470766</td>
      <td>-0.537502</td>
      <td>-0.660175</td>
    </tr>
    <tr>
      <th>154</th>
      <td>0.807363</td>
      <td>-1.247949</td>
      <td>1.327672</td>
      <td>-0.516922</td>
      <td>0.773972</td>
      <td>-0.400185</td>
      <td>-1.916118</td>
    </tr>
    <tr>
      <th>155</th>
      <td>-1.503621</td>
      <td>-1.247949</td>
      <td>1.721629</td>
      <td>-0.516922</td>
      <td>-0.470766</td>
      <td>-0.108869</td>
      <td>-1.916118</td>
    </tr>
  </tbody>
</table>
<p>156 rows × 7 columns</p>
</div>



## 5. Model Building

We will try various classification models to fit in training data using repeated K-fold cross validation and keep the mean scores. 

First we will try with our training data before scaling it. Then, we will fit the models in our scaled data. Then, we will compare results.

In the final step, we will select the best model and apply it to our unseen test data to measure its performance.



```python
#Models that will be used
model_list = [LogisticRegression, GaussianNB, DecisionTreeClassifier, LinearSVC, KNeighborsClassifier,
              RandomForestClassifier, GradientBoostingClassifier, SGDClassifier, XGBClassifier,
             LGBMClassifier, CatBoostClassifier, LinearDiscriminantAnalysis]
model_names = ['Logistic Regression', 'Gaussian Naive Bayes', 'Decision Tree Classifier', 'Linear SVC', 
               'KNeighbors Classifier', 'Random Forest Classifier', 'Gradient Boosting Classifier', 'SGD Classifier',
              'XGB Classifier', 'LGBM Classifier','Cat Boost Classifier', 'Linear Discriminant Analysis']
```


```python
def train_clf_model(clf_list, clf_names, x_train, y_train):
    training_scores = []
    
    for index, clf in enumerate(clf_list):
        model = clf()
         
        #keep the mean of cross validation scores of the training set
        cv = StratifiedKFold(n_splits=5)
        
        #Avoid printing iteration results of Cat Boost Classifier
        if clf_names[index] == 'Cat Boost Classifier':
            scores = cross_val_score(model, x_train, y_train, scoring='accuracy', cv=cv, fit_params={'verbose': False})
        else:
            scores = cross_val_score(model, x_train, y_train, scoring='accuracy', cv=cv)
        training_scores.append(scores.mean())
    
    #create a dataframe to present results
    data = {'Classifier Names': clf_names,
            'Training Score': training_scores}
    
    #maximum 4 decimal points should be sufficient to compare score values
    pd.options.display.float_format = '{:.4f}'.format
    df = pd.DataFrame(data)
    
    return df.sort_values(by='Training Score', ascending=False).reset_index(drop=True)
```


```python
X_train = titanic_train.drop(columns="Survived")
X_train
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>46.0</td>
      <td>0</td>
      <td>0</td>
      <td>79.2000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>34.0</td>
      <td>0</td>
      <td>0</td>
      <td>13.0000</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0</td>
      <td>22.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.7500</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>1</td>
      <td>30.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.7292</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>0</td>
      <td>21.0</td>
      <td>0</td>
      <td>0</td>
      <td>10.5000</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>619</th>
      <td>2</td>
      <td>1</td>
      <td>21.0</td>
      <td>0</td>
      <td>0</td>
      <td>73.5000</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>620</th>
      <td>3</td>
      <td>1</td>
      <td>22.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>621</th>
      <td>3</td>
      <td>1</td>
      <td>22.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.8958</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>622</th>
      <td>3</td>
      <td>0</td>
      <td>63.0</td>
      <td>0</td>
      <td>0</td>
      <td>9.5875</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>623</th>
      <td>3</td>
      <td>1</td>
      <td>38.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.8958</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
<p>624 rows × 7 columns</p>
</div>




```python
Y_train
```




    789    0
    722    0
    141    1
    388    0
    56     1
          ..
    72     0
    112    0
    287    0
    483    1
    108    0
    Name: Survived, Length: 624, dtype: int64




```python
#Using nonscaled training data.
train_clf_model(model_list, model_names, X_train, Y_train)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Classifier Names</th>
      <th>Training Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Cat Boost Classifier</td>
      <td>0.8157</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Gradient Boosting Classifier</td>
      <td>0.8061</td>
    </tr>
    <tr>
      <th>2</th>
      <td>LGBM Classifier</td>
      <td>0.7981</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Random Forest Classifier</td>
      <td>0.7708</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Logistic Regression</td>
      <td>0.7660</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Linear Discriminant Analysis</td>
      <td>0.7644</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Gaussian Naive Bayes</td>
      <td>0.7644</td>
    </tr>
    <tr>
      <th>7</th>
      <td>XGB Classifier</td>
      <td>0.7596</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Decision Tree Classifier</td>
      <td>0.7195</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Linear SVC</td>
      <td>0.6874</td>
    </tr>
    <tr>
      <th>10</th>
      <td>KNeighbors Classifier</td>
      <td>0.6794</td>
    </tr>
    <tr>
      <th>11</th>
      <td>SGD Classifier</td>
      <td>0.5879</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Using the scaled training data.
train_clf_model(model_list, model_names, X_train_scaled, Y_train)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Classifier Names</th>
      <th>Training Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Cat Boost Classifier</td>
      <td>0.8157</td>
    </tr>
    <tr>
      <th>1</th>
      <td>KNeighbors Classifier</td>
      <td>0.8061</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Gradient Boosting Classifier</td>
      <td>0.8045</td>
    </tr>
    <tr>
      <th>3</th>
      <td>LGBM Classifier</td>
      <td>0.7885</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Random Forest Classifier</td>
      <td>0.7756</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Logistic Regression</td>
      <td>0.7660</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Linear Discriminant Analysis</td>
      <td>0.7644</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Gaussian Naive Bayes</td>
      <td>0.7644</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Linear SVC</td>
      <td>0.7612</td>
    </tr>
    <tr>
      <th>9</th>
      <td>XGB Classifier</td>
      <td>0.7596</td>
    </tr>
    <tr>
      <th>10</th>
      <td>SGD Classifier</td>
      <td>0.7371</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Decision Tree Classifier</td>
      <td>0.7131</td>
    </tr>
  </tbody>
</table>
</div>



So, the best model seems to be the Cat Boost Classifier for both cases. Now, let's use it for model prediction.

### Cat Boost Classifier Predictions


```python
X_test = titanic_test.drop(columns="Survived")
X_test
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>1</td>
      <td>24.5000</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>2.0000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>1</td>
      <td>28.0000</td>
      <td>0</td>
      <td>0</td>
      <td>7.7750</td>
      <td>2.0000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>28.0000</td>
      <td>0</td>
      <td>0</td>
      <td>7.0500</td>
      <td>2.0000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>1</td>
      <td>25.0000</td>
      <td>1</td>
      <td>0</td>
      <td>26.0000</td>
      <td>2.0000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1</td>
      <td>64.0000</td>
      <td>0</td>
      <td>0</td>
      <td>26.0000</td>
      <td>2.0000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>151</th>
      <td>3</td>
      <td>1</td>
      <td>32.0000</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>2.0000</td>
    </tr>
    <tr>
      <th>152</th>
      <td>3</td>
      <td>0</td>
      <td>28.0000</td>
      <td>0</td>
      <td>0</td>
      <td>7.7875</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>153</th>
      <td>3</td>
      <td>0</td>
      <td>16.0000</td>
      <td>0</td>
      <td>0</td>
      <td>7.7333</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>154</th>
      <td>3</td>
      <td>0</td>
      <td>45.0000</td>
      <td>0</td>
      <td>1</td>
      <td>14.4542</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>155</th>
      <td>1</td>
      <td>0</td>
      <td>50.0000</td>
      <td>0</td>
      <td>0</td>
      <td>28.7125</td>
      <td>0.0000</td>
    </tr>
  </tbody>
</table>
<p>156 rows × 7 columns</p>
</div>




```python
#model selection
model = CatBoostClassifier()

#fitting it on the training set
model.fit(X_train, Y_train, verbose=False)

#predict the y values based on X_test
y_pred = model.predict(X_test)
        
#accuracy of the test set
print('The accuracy score is {:.4f}'.format(accuracy_score(Y_test, y_pred)))
```

    The accuracy score is 0.7949
    

#### Confusion Matrix


```python
print("Confusion Matrix:")
cm = confusion_matrix(Y_test, y_pred)
print(cm)
plt.matshow(cm)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
print(classification_report(Y_test, y_pred))
```

    Confusion Matrix:
    [[81 10]
     [22 43]]
    


    
![png](Titanic_files/Titanic_82_1.png)
    


                  precision    recall  f1-score   support
    
               0       0.79      0.89      0.84        91
               1       0.81      0.66      0.73        65
    
        accuracy                           0.79       156
       macro avg       0.80      0.78      0.78       156
    weighted avg       0.80      0.79      0.79       156
    
    

Out of 65 survived, our model predicted 43 of them correctly. Out of 91 not survived, it predicted 81 of them correctly.
