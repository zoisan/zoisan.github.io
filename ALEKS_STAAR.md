# The Effect of ALEKS (Math Software) on Mathematics Achivements on STAAR (Texas State Test)

## What is ALEKS?
ALEKS is an artificially intelligent, research-based, online learning and assessment system that offers course products for Math and being used by over 25 million students. ALEKS helps students master course topics through a continuous cycle of mastery, knowledge retention, and positive feedback. Each student begins a new course with a unique set of knowledge and prerequisite gaps to fill. By determining the student's baseline of knowledge, ALEKS creates an individual and dynamic path to success where students learn and then master topics.

## What is STAAR?
STAAR is the state of Texas' testing program and is based on state curriculum standards in core subjects including reading, writing, mathematics, science, and social studies. STAAR tests are designed to measure what students are learning in each grade and whether or not they are ready for the next grade. The goal is to ensure that all students receive what they need to be academically successful. For STAAR, the labels for the performance categories are

- Masters Grade Level
- Meets Grade Level
- Approaches Grade Level
- Did Not Meet Grade Level (Fails)

Students receive one of the above ratings based on their test performance. Schools and districts are rated based on how students score on the State of Texas Assessments of Academic Readiness (STAAR).

## What is the Data Science Problem?
Our 3rd-9th grade students have blended learning time in which students work on completing their ALEKS topic goals. We believe ALEKS helps students improve their math skills and retain their math knowledge. One metric that measure math progress is STAAR test.

Our problem is to be able to project our students STAAR test performances (pass or fail) based on their ALEKS progress.

Our hypothesis is that students who reach certain amount of mastery on ALEKS will likely to pass the STAAR test (receive Approaches grade level or above rating).

## How was Data Collected?
We gathered data by generating ALEKS reports and combining with STAAR test results. During the data cleaning process, some data has been removed due to not missing fileds and duplicates. We also removed confidential columns like student and teacher names, and ID numbers as well as some irrelevant colums. 

### Loading Libraries


```python
import numpy as np
import pandas as pd
from pandas import read_csv
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV, RepeatedStratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, plot_confusion_matrix, \
roc_auc_score, roc_curve, auc, SCORERS

from sklearn import linear_model
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn import svm 
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA, KernelPCA
from sklearn.utils import resample

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from sklearn.exceptions import ConvergenceWarning
ConvergenceWarning('ignore')


pd.set_option('display.max_columns', None)

%matplotlib inline

random_seed = 42
```


```python
#location of the dataset.
path = "./dataset/ALEKS_2021-22_EOY.xlsx"
```


```python
# Load dataset
df_original = pd.read_excel(path)
```


```python
# Make a copy so that we keep the original just in case.
df = df_original.copy()
```


```python
#Take a look at the data.
df
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
      <th>School Grade</th>
      <th>ALEKS Grade</th>
      <th>Campus</th>
      <th>BOY Mastery</th>
      <th>BOY Mastered Topics</th>
      <th>EOY Mastery</th>
      <th>EOY Mastered Topics</th>
      <th>Total Number of Topics</th>
      <th>Progress</th>
      <th>Learned</th>
      <th>Total Time Spent</th>
      <th>STAAR Performance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7</td>
      <td>7</td>
      <td>Harmony School of Innovation-Waco</td>
      <td>0.125506</td>
      <td>31</td>
      <td>0.125506</td>
      <td>31</td>
      <td>247</td>
      <td>0.010000</td>
      <td>16</td>
      <td>897.600000</td>
      <td>Meets</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9</td>
      <td>9</td>
      <td>Harmony Science Academy-El Paso</td>
      <td>0.073654</td>
      <td>26</td>
      <td>0.073654</td>
      <td>26</td>
      <td>353</td>
      <td>0.010000</td>
      <td>20</td>
      <td>513.183333</td>
      <td>Approaches</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>8</td>
      <td>Harmony School of Innovation-Garland</td>
      <td>0.111570</td>
      <td>27</td>
      <td>0.111570</td>
      <td>27</td>
      <td>242</td>
      <td>0.010000</td>
      <td>35</td>
      <td>1375.300000</td>
      <td>Approaches</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6</td>
      <td>6</td>
      <td>Harmony Science Academy-Garland</td>
      <td>0.366013</td>
      <td>112</td>
      <td>0.366013</td>
      <td>112</td>
      <td>306</td>
      <td>0.010000</td>
      <td>74</td>
      <td>1886.050000</td>
      <td>Approaches</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>5</td>
      <td>Harmony School of Excellence-San Antonio</td>
      <td>0.051587</td>
      <td>13</td>
      <td>0.055556</td>
      <td>14</td>
      <td>252</td>
      <td>0.010000</td>
      <td>13</td>
      <td>842.016667</td>
      <td>Fails</td>
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
      <th>11714</th>
      <td>6</td>
      <td>9</td>
      <td>Harmony School of Discovery-Houston</td>
      <td>0.249292</td>
      <td>88</td>
      <td>2.000000</td>
      <td>353</td>
      <td>353</td>
      <td>1.750708</td>
      <td>324</td>
      <td>5221.383333</td>
      <td>Masters</td>
    </tr>
    <tr>
      <th>11715</th>
      <td>6</td>
      <td>7</td>
      <td>Harmony School of Technology-Houston</td>
      <td>0.234818</td>
      <td>58</td>
      <td>2.000000</td>
      <td>247</td>
      <td>247</td>
      <td>1.765182</td>
      <td>179</td>
      <td>5617.866667</td>
      <td>Masters</td>
    </tr>
    <tr>
      <th>11716</th>
      <td>3</td>
      <td>6</td>
      <td>Harmony Science Academy-Cedar Park</td>
      <td>0.233696</td>
      <td>43</td>
      <td>2.000000</td>
      <td>184</td>
      <td>184</td>
      <td>1.766304</td>
      <td>144</td>
      <td>1174.400000</td>
      <td>Meets</td>
    </tr>
    <tr>
      <th>11717</th>
      <td>4</td>
      <td>5</td>
      <td>Harmony School of Innovation-Carrollton</td>
      <td>0.181818</td>
      <td>30</td>
      <td>1.957576</td>
      <td>158</td>
      <td>165</td>
      <td>1.775758</td>
      <td>133</td>
      <td>1528.883333</td>
      <td>Masters</td>
    </tr>
    <tr>
      <th>11718</th>
      <td>3</td>
      <td>5</td>
      <td>Harmony School of Science-Austin</td>
      <td>0.000000</td>
      <td>0</td>
      <td>1.830303</td>
      <td>137</td>
      <td>165</td>
      <td>1.830303</td>
      <td>143</td>
      <td>770.250000</td>
      <td>Masters</td>
    </tr>
  </tbody>
</table>
<p>11719 rows × 12 columns</p>
</div>



## Explanation of Features
- <b>School Grade:</b> Students current math grade
- <b>ALEKS Grade:</b> Some students can be placed in lower or higher than their school grade in ALEKS depending on their needs.
- <b>Campus:</b> School name
- <b>BOY Mastery:</b> What percentage of total topics a student mastered at the begining of the school year during the initial knowledge check.
- <b>BOY Mastered Topics:</b> How many topics a student mastered at the begining of the school year during the initial knowledge check. 
- <b>EOY Mastery:</b> What percentage of total topics a student mastered by the end of the school year.
- <b>EOY Mastered Topics:</b> How many topics a student mastered by the end of the school year.
- <b>Total Number of Topics:</b> Total number of topics available in student's ALEKS course.
- <b>Progress:</b> The difference between EOY and BOY Mastery.
- <b>Learned:</b> How many topics a student learn during the coursework.
- <b>Attempted:</b> How many topics a student attempted to learn during the coursework.
- <b>Total Time Spent:</b> How much time in minutes a student spent on ALEKS during the entire coursework.
- <b>STAAR Performance:</b> Performance category of a student based on STAAR test result.


```python
#Data Types
df.dtypes
```




    School Grade                int64
    ALEKS Grade                 int64
    Campus                     object
    BOY Mastery               float64
    BOY Mastered Topics         int64
    EOY Mastery               float64
    EOY Mastered Topics         int64
    Total Number of Topics      int64
    Progress                  float64
    Learned                     int64
    Total Time Spent          float64
    STAAR Performance          object
    dtype: object




```python
#We need to encode target values. We are only interested in pass/fail status.
df["Target"] = df["STAAR Performance"].map({"Fails":0, "Approaches":1, "Meets":1, "Masters":1})
df
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
      <th>School Grade</th>
      <th>ALEKS Grade</th>
      <th>Campus</th>
      <th>BOY Mastery</th>
      <th>BOY Mastered Topics</th>
      <th>EOY Mastery</th>
      <th>EOY Mastered Topics</th>
      <th>Total Number of Topics</th>
      <th>Progress</th>
      <th>Learned</th>
      <th>Total Time Spent</th>
      <th>STAAR Performance</th>
      <th>Target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7</td>
      <td>7</td>
      <td>Harmony School of Innovation-Waco</td>
      <td>0.125506</td>
      <td>31</td>
      <td>0.125506</td>
      <td>31</td>
      <td>247</td>
      <td>0.010000</td>
      <td>16</td>
      <td>897.600000</td>
      <td>Meets</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9</td>
      <td>9</td>
      <td>Harmony Science Academy-El Paso</td>
      <td>0.073654</td>
      <td>26</td>
      <td>0.073654</td>
      <td>26</td>
      <td>353</td>
      <td>0.010000</td>
      <td>20</td>
      <td>513.183333</td>
      <td>Approaches</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>8</td>
      <td>Harmony School of Innovation-Garland</td>
      <td>0.111570</td>
      <td>27</td>
      <td>0.111570</td>
      <td>27</td>
      <td>242</td>
      <td>0.010000</td>
      <td>35</td>
      <td>1375.300000</td>
      <td>Approaches</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6</td>
      <td>6</td>
      <td>Harmony Science Academy-Garland</td>
      <td>0.366013</td>
      <td>112</td>
      <td>0.366013</td>
      <td>112</td>
      <td>306</td>
      <td>0.010000</td>
      <td>74</td>
      <td>1886.050000</td>
      <td>Approaches</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>5</td>
      <td>Harmony School of Excellence-San Antonio</td>
      <td>0.051587</td>
      <td>13</td>
      <td>0.055556</td>
      <td>14</td>
      <td>252</td>
      <td>0.010000</td>
      <td>13</td>
      <td>842.016667</td>
      <td>Fails</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>11714</th>
      <td>6</td>
      <td>9</td>
      <td>Harmony School of Discovery-Houston</td>
      <td>0.249292</td>
      <td>88</td>
      <td>2.000000</td>
      <td>353</td>
      <td>353</td>
      <td>1.750708</td>
      <td>324</td>
      <td>5221.383333</td>
      <td>Masters</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11715</th>
      <td>6</td>
      <td>7</td>
      <td>Harmony School of Technology-Houston</td>
      <td>0.234818</td>
      <td>58</td>
      <td>2.000000</td>
      <td>247</td>
      <td>247</td>
      <td>1.765182</td>
      <td>179</td>
      <td>5617.866667</td>
      <td>Masters</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11716</th>
      <td>3</td>
      <td>6</td>
      <td>Harmony Science Academy-Cedar Park</td>
      <td>0.233696</td>
      <td>43</td>
      <td>2.000000</td>
      <td>184</td>
      <td>184</td>
      <td>1.766304</td>
      <td>144</td>
      <td>1174.400000</td>
      <td>Meets</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11717</th>
      <td>4</td>
      <td>5</td>
      <td>Harmony School of Innovation-Carrollton</td>
      <td>0.181818</td>
      <td>30</td>
      <td>1.957576</td>
      <td>158</td>
      <td>165</td>
      <td>1.775758</td>
      <td>133</td>
      <td>1528.883333</td>
      <td>Masters</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11718</th>
      <td>3</td>
      <td>5</td>
      <td>Harmony School of Science-Austin</td>
      <td>0.000000</td>
      <td>0</td>
      <td>1.830303</td>
      <td>137</td>
      <td>165</td>
      <td>1.830303</td>
      <td>143</td>
      <td>770.250000</td>
      <td>Masters</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>11719 rows × 13 columns</p>
</div>



## Do we have a balanced or imbalanced dataset?


```python
print('The percentage of "0" cases is {:.1f}%'.format((df.Target.value_counts()[0]/df.shape[0])*100))
```

    The percentage of "0" cases is 25.8%
    

Only 26% of the data belongs to class 0 and 74% belongs to class 1. So, we have an imbalanced dataset.

## Train and Test Data Split

It is very important to split the test data before data preprocessing steps to avoid data leakage!

We will set aside our test data and only use training data first. Then we will take similar preprocessing steps for test data.


```python
#X values
X = df.drop(columns=["Campus","STAAR Performance","Target"])
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
      <th>School Grade</th>
      <th>ALEKS Grade</th>
      <th>BOY Mastery</th>
      <th>BOY Mastered Topics</th>
      <th>EOY Mastery</th>
      <th>EOY Mastered Topics</th>
      <th>Total Number of Topics</th>
      <th>Progress</th>
      <th>Learned</th>
      <th>Total Time Spent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7</td>
      <td>7</td>
      <td>0.125506</td>
      <td>31</td>
      <td>0.125506</td>
      <td>31</td>
      <td>247</td>
      <td>0.010000</td>
      <td>16</td>
      <td>897.600000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9</td>
      <td>9</td>
      <td>0.073654</td>
      <td>26</td>
      <td>0.073654</td>
      <td>26</td>
      <td>353</td>
      <td>0.010000</td>
      <td>20</td>
      <td>513.183333</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>8</td>
      <td>0.111570</td>
      <td>27</td>
      <td>0.111570</td>
      <td>27</td>
      <td>242</td>
      <td>0.010000</td>
      <td>35</td>
      <td>1375.300000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6</td>
      <td>6</td>
      <td>0.366013</td>
      <td>112</td>
      <td>0.366013</td>
      <td>112</td>
      <td>306</td>
      <td>0.010000</td>
      <td>74</td>
      <td>1886.050000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>5</td>
      <td>0.051587</td>
      <td>13</td>
      <td>0.055556</td>
      <td>14</td>
      <td>252</td>
      <td>0.010000</td>
      <td>13</td>
      <td>842.016667</td>
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
    </tr>
    <tr>
      <th>11714</th>
      <td>6</td>
      <td>9</td>
      <td>0.249292</td>
      <td>88</td>
      <td>2.000000</td>
      <td>353</td>
      <td>353</td>
      <td>1.750708</td>
      <td>324</td>
      <td>5221.383333</td>
    </tr>
    <tr>
      <th>11715</th>
      <td>6</td>
      <td>7</td>
      <td>0.234818</td>
      <td>58</td>
      <td>2.000000</td>
      <td>247</td>
      <td>247</td>
      <td>1.765182</td>
      <td>179</td>
      <td>5617.866667</td>
    </tr>
    <tr>
      <th>11716</th>
      <td>3</td>
      <td>6</td>
      <td>0.233696</td>
      <td>43</td>
      <td>2.000000</td>
      <td>184</td>
      <td>184</td>
      <td>1.766304</td>
      <td>144</td>
      <td>1174.400000</td>
    </tr>
    <tr>
      <th>11717</th>
      <td>4</td>
      <td>5</td>
      <td>0.181818</td>
      <td>30</td>
      <td>1.957576</td>
      <td>158</td>
      <td>165</td>
      <td>1.775758</td>
      <td>133</td>
      <td>1528.883333</td>
    </tr>
    <tr>
      <th>11718</th>
      <td>3</td>
      <td>5</td>
      <td>0.000000</td>
      <td>0</td>
      <td>1.830303</td>
      <td>137</td>
      <td>165</td>
      <td>1.830303</td>
      <td>143</td>
      <td>770.250000</td>
    </tr>
  </tbody>
</table>
<p>11719 rows × 10 columns</p>
</div>




```python
#y values
y = df["Target"]
y
```




    0        1
    1        1
    2        1
    3        1
    4        0
            ..
    11714    1
    11715    1
    11716    1
    11717    1
    11718    1
    Name: Target, Length: 11719, dtype: int64



Let's see if we have a balanced dataset.


```python
#Train and test data split
X_train, X_test, Y_train, Y_test = train_test_split(X, y, train_size=0.8, random_state=random_seed)
```


```python
#Let's take a look at the shape
(X_train.shape, X_test.shape)
```




    ((9375, 10), (2344, 10))




```python
#create a dataframe of trainging data including the target values for EDA and visualization
df_train = X_train.copy()
df_train["Target"] = Y_train
```


```python
#reseting index values
df_train.reset_index(drop=True, inplace=True)
df_train
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
      <th>School Grade</th>
      <th>ALEKS Grade</th>
      <th>BOY Mastery</th>
      <th>BOY Mastered Topics</th>
      <th>EOY Mastery</th>
      <th>EOY Mastered Topics</th>
      <th>Total Number of Topics</th>
      <th>Progress</th>
      <th>Learned</th>
      <th>Total Time Spent</th>
      <th>Target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9</td>
      <td>9</td>
      <td>0.099150</td>
      <td>35</td>
      <td>0.750708</td>
      <td>265</td>
      <td>353</td>
      <td>0.651558</td>
      <td>218</td>
      <td>2750.700000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>5</td>
      <td>0.095238</td>
      <td>24</td>
      <td>0.253968</td>
      <td>64</td>
      <td>252</td>
      <td>0.158730</td>
      <td>56</td>
      <td>1676.616667</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>3</td>
      <td>0.445652</td>
      <td>41</td>
      <td>0.967391</td>
      <td>89</td>
      <td>92</td>
      <td>0.521739</td>
      <td>59</td>
      <td>840.216667</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>4</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.178571</td>
      <td>45</td>
      <td>252</td>
      <td>0.178571</td>
      <td>70</td>
      <td>1569.933333</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>5</td>
      <td>0.253968</td>
      <td>64</td>
      <td>0.611111</td>
      <td>154</td>
      <td>252</td>
      <td>0.357143</td>
      <td>110</td>
      <td>1579.483333</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>9370</th>
      <td>3</td>
      <td>5</td>
      <td>0.609195</td>
      <td>159</td>
      <td>1.896552</td>
      <td>234</td>
      <td>261</td>
      <td>1.287356</td>
      <td>85</td>
      <td>1575.483333</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9371</th>
      <td>6</td>
      <td>6</td>
      <td>0.009804</td>
      <td>3</td>
      <td>0.297386</td>
      <td>91</td>
      <td>306</td>
      <td>0.287582</td>
      <td>125</td>
      <td>2030.900000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9372</th>
      <td>3</td>
      <td>3</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.301136</td>
      <td>53</td>
      <td>176</td>
      <td>0.301136</td>
      <td>62</td>
      <td>1072.866667</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9373</th>
      <td>6</td>
      <td>6</td>
      <td>0.035948</td>
      <td>11</td>
      <td>0.114379</td>
      <td>35</td>
      <td>306</td>
      <td>0.078431</td>
      <td>24</td>
      <td>777.800000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9374</th>
      <td>7</td>
      <td>7</td>
      <td>0.538874</td>
      <td>201</td>
      <td>0.970509</td>
      <td>362</td>
      <td>373</td>
      <td>0.431635</td>
      <td>230</td>
      <td>9005.350000</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>9375 rows × 11 columns</p>
</div>




```python
#checking missing values
df_train.isna().sum()
```




    School Grade              0
    ALEKS Grade               0
    BOY Mastery               0
    BOY Mastered Topics       0
    EOY Mastery               0
    EOY Mastered Topics       0
    Total Number of Topics    0
    Progress                  0
    Learned                   0
    Total Time Spent          0
    Target                    0
    dtype: int64




```python
#create a dataframe of testing data including the target values for EDA and visualization
df_test = X_test.copy()
df_test["Target"] = Y_test
df_test
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
      <th>School Grade</th>
      <th>ALEKS Grade</th>
      <th>BOY Mastery</th>
      <th>BOY Mastered Topics</th>
      <th>EOY Mastery</th>
      <th>EOY Mastered Topics</th>
      <th>Total Number of Topics</th>
      <th>Progress</th>
      <th>Learned</th>
      <th>Total Time Spent</th>
      <th>Target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>101</th>
      <td>6</td>
      <td>6</td>
      <td>0.022876</td>
      <td>7</td>
      <td>0.049020</td>
      <td>15</td>
      <td>306</td>
      <td>0.026144</td>
      <td>12</td>
      <td>339.966667</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5271</th>
      <td>6</td>
      <td>6</td>
      <td>0.042484</td>
      <td>13</td>
      <td>0.336601</td>
      <td>103</td>
      <td>306</td>
      <td>0.294118</td>
      <td>92</td>
      <td>1760.516667</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5146</th>
      <td>4</td>
      <td>4</td>
      <td>0.190202</td>
      <td>66</td>
      <td>0.475504</td>
      <td>165</td>
      <td>347</td>
      <td>0.285303</td>
      <td>112</td>
      <td>1748.416667</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9219</th>
      <td>3</td>
      <td>3</td>
      <td>0.358696</td>
      <td>33</td>
      <td>0.989130</td>
      <td>91</td>
      <td>92</td>
      <td>0.630435</td>
      <td>72</td>
      <td>1069.366667</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1845</th>
      <td>7</td>
      <td>7</td>
      <td>0.072874</td>
      <td>18</td>
      <td>0.198381</td>
      <td>49</td>
      <td>247</td>
      <td>0.125506</td>
      <td>37</td>
      <td>489.800000</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3116</th>
      <td>9</td>
      <td>9</td>
      <td>0.110701</td>
      <td>60</td>
      <td>0.293358</td>
      <td>159</td>
      <td>542</td>
      <td>0.182657</td>
      <td>89</td>
      <td>4016.683333</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2199</th>
      <td>7</td>
      <td>7</td>
      <td>0.085020</td>
      <td>21</td>
      <td>0.226721</td>
      <td>56</td>
      <td>247</td>
      <td>0.141700</td>
      <td>72</td>
      <td>1070.950000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2503</th>
      <td>4</td>
      <td>4</td>
      <td>0.422680</td>
      <td>82</td>
      <td>0.577320</td>
      <td>112</td>
      <td>194</td>
      <td>0.154639</td>
      <td>29</td>
      <td>1268.850000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4194</th>
      <td>6</td>
      <td>6</td>
      <td>0.477124</td>
      <td>146</td>
      <td>0.712418</td>
      <td>218</td>
      <td>306</td>
      <td>0.235294</td>
      <td>71</td>
      <td>857.133333</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10183</th>
      <td>9</td>
      <td>9</td>
      <td>0.039660</td>
      <td>14</td>
      <td>0.889518</td>
      <td>314</td>
      <td>353</td>
      <td>0.849858</td>
      <td>345</td>
      <td>7239.133333</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>2344 rows × 11 columns</p>
</div>



## EDA and Visualization


```python
df_train.describe()
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
      <th>School Grade</th>
      <th>ALEKS Grade</th>
      <th>BOY Mastery</th>
      <th>BOY Mastered Topics</th>
      <th>EOY Mastery</th>
      <th>EOY Mastered Topics</th>
      <th>Total Number of Topics</th>
      <th>Progress</th>
      <th>Learned</th>
      <th>Total Time Spent</th>
      <th>Target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>9375.000000</td>
      <td>9375.000000</td>
      <td>9375.000000</td>
      <td>9375.000000</td>
      <td>9375.000000</td>
      <td>9375.000000</td>
      <td>9375.000000</td>
      <td>9375.000000</td>
      <td>9375.000000</td>
      <td>9375.000000</td>
      <td>9375.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>6.131093</td>
      <td>6.276053</td>
      <td>0.182286</td>
      <td>57.014187</td>
      <td>0.620316</td>
      <td>145.900373</td>
      <td>296.035200</td>
      <td>0.438037</td>
      <td>104.352427</td>
      <td>1871.136590</td>
      <td>0.740800</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.991230</td>
      <td>1.999989</td>
      <td>0.170929</td>
      <td>66.894774</td>
      <td>0.455455</td>
      <td>97.389451</td>
      <td>111.528648</td>
      <td>0.356982</td>
      <td>81.565921</td>
      <td>1352.090297</td>
      <td>0.438219</td>
    </tr>
    <tr>
      <th>min</th>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.005666</td>
      <td>2.000000</td>
      <td>46.000000</td>
      <td>0.005666</td>
      <td>1.000000</td>
      <td>300.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>4.000000</td>
      <td>5.000000</td>
      <td>0.049020</td>
      <td>13.000000</td>
      <td>0.289256</td>
      <td>71.000000</td>
      <td>242.000000</td>
      <td>0.175637</td>
      <td>43.000000</td>
      <td>866.091667</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>6.000000</td>
      <td>6.000000</td>
      <td>0.127451</td>
      <td>34.000000</td>
      <td>0.515581</td>
      <td>125.000000</td>
      <td>254.000000</td>
      <td>0.330065</td>
      <td>82.000000</td>
      <td>1565.683333</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>8.000000</td>
      <td>8.000000</td>
      <td>0.269841</td>
      <td>74.000000</td>
      <td>0.794595</td>
      <td>203.000000</td>
      <td>353.000000</td>
      <td>0.578696</td>
      <td>145.000000</td>
      <td>2510.691667</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>9.000000</td>
      <td>11.000000</td>
      <td>0.979381</td>
      <td>543.000000</td>
      <td>2.000000</td>
      <td>909.000000</td>
      <td>920.000000</td>
      <td>1.766304</td>
      <td>587.000000</td>
      <td>13411.483333</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#correlation matrix
df_train.corr()
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
      <th>School Grade</th>
      <th>ALEKS Grade</th>
      <th>BOY Mastery</th>
      <th>BOY Mastered Topics</th>
      <th>EOY Mastery</th>
      <th>EOY Mastered Topics</th>
      <th>Total Number of Topics</th>
      <th>Progress</th>
      <th>Learned</th>
      <th>Total Time Spent</th>
      <th>Target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>School Grade</th>
      <td>1.000000</td>
      <td>0.973330</td>
      <td>-0.208268</td>
      <td>0.041657</td>
      <td>-0.217535</td>
      <td>0.153210</td>
      <td>0.574057</td>
      <td>-0.177796</td>
      <td>0.147183</td>
      <td>0.214281</td>
      <td>0.186505</td>
    </tr>
    <tr>
      <th>ALEKS Grade</th>
      <td>0.973330</td>
      <td>1.000000</td>
      <td>-0.108516</td>
      <td>0.149140</td>
      <td>-0.048929</td>
      <td>0.216933</td>
      <td>0.629998</td>
      <td>-0.010444</td>
      <td>0.128143</td>
      <td>0.167515</td>
      <td>0.234163</td>
    </tr>
    <tr>
      <th>BOY Mastery</th>
      <td>-0.208268</td>
      <td>-0.108516</td>
      <td>1.000000</td>
      <td>0.850134</td>
      <td>0.701424</td>
      <td>0.570563</td>
      <td>0.160060</td>
      <td>0.416090</td>
      <td>0.011513</td>
      <td>-0.115815</td>
      <td>0.286687</td>
    </tr>
    <tr>
      <th>BOY Mastered Topics</th>
      <td>0.041657</td>
      <td>0.149140</td>
      <td>0.850134</td>
      <td>1.000000</td>
      <td>0.640949</td>
      <td>0.748768</td>
      <td>0.561648</td>
      <td>0.410693</td>
      <td>0.113443</td>
      <td>-0.041645</td>
      <td>0.272336</td>
    </tr>
    <tr>
      <th>EOY Mastery</th>
      <td>-0.217535</td>
      <td>-0.048929</td>
      <td>0.701424</td>
      <td>0.640949</td>
      <td>1.000000</td>
      <td>0.644223</td>
      <td>0.177805</td>
      <td>0.939971</td>
      <td>0.262756</td>
      <td>0.037418</td>
      <td>0.354977</td>
    </tr>
    <tr>
      <th>EOY Mastered Topics</th>
      <td>0.153210</td>
      <td>0.216933</td>
      <td>0.570563</td>
      <td>0.748768</td>
      <td>0.644223</td>
      <td>1.000000</td>
      <td>0.598129</td>
      <td>0.548713</td>
      <td>0.659344</td>
      <td>0.381417</td>
      <td>0.368535</td>
    </tr>
    <tr>
      <th>Total Number of Topics</th>
      <td>0.574057</td>
      <td>0.629998</td>
      <td>0.160060</td>
      <td>0.561648</td>
      <td>0.177805</td>
      <td>0.598129</td>
      <td>1.000000</td>
      <td>0.150217</td>
      <td>0.282669</td>
      <td>0.122344</td>
      <td>0.221571</td>
    </tr>
    <tr>
      <th>Progress</th>
      <td>-0.177796</td>
      <td>-0.010444</td>
      <td>0.416090</td>
      <td>0.410693</td>
      <td>0.939971</td>
      <td>0.548713</td>
      <td>0.150217</td>
      <td>1.000000</td>
      <td>0.329706</td>
      <td>0.103181</td>
      <td>0.315626</td>
    </tr>
    <tr>
      <th>Learned</th>
      <td>0.147183</td>
      <td>0.128143</td>
      <td>0.011513</td>
      <td>0.113443</td>
      <td>0.262756</td>
      <td>0.659344</td>
      <td>0.282669</td>
      <td>0.329706</td>
      <td>1.000000</td>
      <td>0.670765</td>
      <td>0.210404</td>
    </tr>
    <tr>
      <th>Total Time Spent</th>
      <td>0.214281</td>
      <td>0.167515</td>
      <td>-0.115815</td>
      <td>-0.041645</td>
      <td>0.037418</td>
      <td>0.381417</td>
      <td>0.122344</td>
      <td>0.103181</td>
      <td>0.670765</td>
      <td>1.000000</td>
      <td>0.057438</td>
    </tr>
    <tr>
      <th>Target</th>
      <td>0.186505</td>
      <td>0.234163</td>
      <td>0.286687</td>
      <td>0.272336</td>
      <td>0.354977</td>
      <td>0.368535</td>
      <td>0.221571</td>
      <td>0.315626</td>
      <td>0.210404</td>
      <td>0.057438</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#let's round the numbers up to 2 decimal places
fig = px.imshow(df_train.corr().round(2), text_auto=True)
fig.update_layout(
    autosize=False,
    width=800,
    height=800)
fig.show()
{% include fig1.html %}
```


<div>                            <div id="6ec33d60-9927-4c54-bc3d-e93b5ae0bf96" class="plotly-graph-div" style="height:800px; width:800px;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("6ec33d60-9927-4c54-bc3d-e93b5ae0bf96")) {                    Plotly.newPlot(                        "6ec33d60-9927-4c54-bc3d-e93b5ae0bf96",                        [{"coloraxis":"coloraxis","name":"0","texttemplate":"%{z}","x":["School Grade","ALEKS Grade","BOY Mastery","BOY Mastered Topics","EOY Mastery","EOY Mastered Topics","Total Number of Topics","Progress","Learned","Total Time Spent","Target"],"y":["School Grade","ALEKS Grade","BOY Mastery","BOY Mastered Topics","EOY Mastery","EOY Mastered Topics","Total Number of Topics","Progress","Learned","Total Time Spent","Target"],"z":[[1.0,0.97,-0.21,0.04,-0.22,0.15,0.57,-0.18,0.15,0.21,0.19],[0.97,1.0,-0.11,0.15,-0.05,0.22,0.63,-0.01,0.13,0.17,0.23],[-0.21,-0.11,1.0,0.85,0.7,0.57,0.16,0.42,0.01,-0.12,0.29],[0.04,0.15,0.85,1.0,0.64,0.75,0.56,0.41,0.11,-0.04,0.27],[-0.22,-0.05,0.7,0.64,1.0,0.64,0.18,0.94,0.26,0.04,0.35],[0.15,0.22,0.57,0.75,0.64,1.0,0.6,0.55,0.66,0.38,0.37],[0.57,0.63,0.16,0.56,0.18,0.6,1.0,0.15,0.28,0.12,0.22],[-0.18,-0.01,0.42,0.41,0.94,0.55,0.15,1.0,0.33,0.1,0.32],[0.15,0.13,0.01,0.11,0.26,0.66,0.28,0.33,1.0,0.67,0.21],[0.21,0.17,-0.12,-0.04,0.04,0.38,0.12,0.1,0.67,1.0,0.06],[0.19,0.23,0.29,0.27,0.35,0.37,0.22,0.32,0.21,0.06,1.0]],"type":"heatmap","xaxis":"x","yaxis":"y","hovertemplate":"x: %{x}<br>y: %{y}<br>color: %{z}<extra></extra>"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"xaxis":{"anchor":"y","domain":[0.0,1.0],"scaleanchor":"y","constrain":"domain"},"yaxis":{"anchor":"x","domain":[0.0,1.0],"autorange":"reversed","constrain":"domain"},"coloraxis":{"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"margin":{"t":60},"autosize":false,"width":800,"height":800},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('6ec33d60-9927-4c54-bc3d-e93b5ae0bf96');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>


It seems that there is some type of correlation between Target values and EOY Mastered Topics, EOY Mastery, and Progress.

### ALEKS Performace by STAAR Ratings


```python
#Let's take a look at the avarege ALEKS performances by STAAR ratings.
df_new = df.groupby("STAAR Performance", as_index=False).mean()
df_new.sort_values(by=["Progress"], inplace=True)
df_new.reset_index(drop=True, inplace=True)
df_new
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
      <th>STAAR Performance</th>
      <th>School Grade</th>
      <th>ALEKS Grade</th>
      <th>BOY Mastery</th>
      <th>BOY Mastered Topics</th>
      <th>EOY Mastery</th>
      <th>EOY Mastered Topics</th>
      <th>Total Number of Topics</th>
      <th>Progress</th>
      <th>Learned</th>
      <th>Total Time Spent</th>
      <th>Target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Fails</td>
      <td>5.499175</td>
      <td>5.481677</td>
      <td>0.098385</td>
      <td>25.760977</td>
      <td>0.345371</td>
      <td>84.550017</td>
      <td>254.091779</td>
      <td>0.246993</td>
      <td>75.027402</td>
      <td>1726.626896</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Approaches</td>
      <td>6.087448</td>
      <td>6.083284</td>
      <td>0.144783</td>
      <td>40.482748</td>
      <td>0.463920</td>
      <td>126.024093</td>
      <td>281.182629</td>
      <td>0.319150</td>
      <td>103.671921</td>
      <td>1984.748939</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Meets</td>
      <td>6.233803</td>
      <td>6.414085</td>
      <td>0.219267</td>
      <td>68.414085</td>
      <td>0.738133</td>
      <td>169.247887</td>
      <td>304.512676</td>
      <td>0.518870</td>
      <td>117.207243</td>
      <td>1943.140966</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Masters</td>
      <td>6.729511</td>
      <td>7.185719</td>
      <td>0.283454</td>
      <td>99.848048</td>
      <td>0.991198</td>
      <td>215.672177</td>
      <td>350.616954</td>
      <td>0.707753</td>
      <td>126.559620</td>
      <td>1828.738393</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



## Number of Topics Completed by STAAR Performance


```python
fig1 = px.pie(df_new, values='EOY Mastered Topics', names='STAAR Performance', 
              color='STAAR Performance', title='Number of Topics Completed by Performance')

fig1.update_traces(textposition='inside', textinfo='percent+label')
fig1.show()
```


<div>                            <div id="022fc524-214d-43f4-b78f-cc7c5e52186f" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("022fc524-214d-43f4-b78f-cc7c5e52186f")) {                    Plotly.newPlot(                        "022fc524-214d-43f4-b78f-cc7c5e52186f",                        [{"customdata":[["Fails"],["Approaches"],["Meets"],["Masters"]],"domain":{"x":[0.0,1.0],"y":[0.0,1.0]},"hovertemplate":"STAAR Performance=%{customdata[0]}<br>EOY Mastered Topics=%{value}<extra></extra>","labels":["Fails","Approaches","Meets","Masters"],"legendgroup":"","marker":{"colors":["#636efa","#EF553B","#00cc96","#ab63fa"]},"name":"","showlegend":true,"values":[84.55001650709805,126.02409280190363,169.24788732394367,215.67217727752373],"type":"pie","textinfo":"percent+label","textposition":"inside"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"legend":{"tracegroupgap":0},"title":{"text":"Number of Topics Completed by Performance"}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('022fc524-214d-43f4-b78f-cc7c5e52186f');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>


As you see, students who "Fail" the STAAR test are accountable for only 14% of topics mastered by all students. On the other hand, students who "Master" the STAAR are responsible for the biggest chunk: 36%  of all topics mastered.

We can also observe strong relationship between number of topics mastered and STAAR Performances.

## ALEKS Progress by STAAR Performance


```python
fig2 = px.pie(df_new, values='Progress', names='STAAR Performance', 
              color='STAAR Performance', title='ALEKS Progress by STAAR Performance')

fig2.update_traces(textposition='inside', textinfo='percent+label')
fig2.show()
```


<div>                            <div id="d8f41004-fac5-4a6f-b50c-4061c25fc323" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("d8f41004-fac5-4a6f-b50c-4061c25fc323")) {                    Plotly.newPlot(                        "d8f41004-fac5-4a6f-b50c-4061c25fc323",                        [{"customdata":[["Fails"],["Approaches"],["Meets"],["Masters"]],"domain":{"x":[0.0,1.0],"y":[0.0,1.0]},"hovertemplate":"STAAR Performance=%{customdata[0]}<br>Progress=%{value}<extra></extra>","labels":["Fails","Approaches","Meets","Masters"],"legendgroup":"","marker":{"colors":["#636efa","#EF553B","#00cc96","#ab63fa"]},"name":"","showlegend":true,"values":[0.2469925574992667,0.31915002454868857,0.5188702549667097,0.7077525989477303],"type":"pie","textinfo":"percent+label","textposition":"inside"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"legend":{"tracegroupgap":0},"title":{"text":"ALEKS Progress by STAAR Performance"}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('d8f41004-fac5-4a6f-b50c-4061c25fc323');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>


This is really alligned with the above chart. Students who show less progress on ALEKS perform lower on STAAR test.

## Total Time Spent by STAAR Performance


```python
fig3 = px.pie(df_new, values='Total Time Spent', names='STAAR Performance', 
              color='STAAR Performance', title='Total Time Spent by Performance')

fig3.update_traces(textposition='inside', textinfo='percent+label')
fig3.show()
```


<div>                            <div id="3d31f75d-9df9-44f0-8515-1b2ee998c40a" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("3d31f75d-9df9-44f0-8515-1b2ee998c40a")) {                    Plotly.newPlot(                        "3d31f75d-9df9-44f0-8515-1b2ee998c40a",                        [{"customdata":[["Fails"],["Approaches"],["Meets"],["Masters"]],"domain":{"x":[0.0,1.0],"y":[0.0,1.0]},"hovertemplate":"STAAR Performance=%{customdata[0]}<br>Total Time Spent=%{value}<extra></extra>","labels":["Fails","Approaches","Meets","Masters"],"legendgroup":"","marker":{"colors":["#636efa","#EF553B","#00cc96","#ab63fa"]},"name":"","showlegend":true,"values":[1726.6268955650933,1984.7489391235376,1943.1409657947686,1828.7383925430884],"type":"pie","textinfo":"percent+label","textposition":"inside"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"legend":{"tracegroupgap":0},"title":{"text":"Total Time Spent by Performance"}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('3d31f75d-9df9-44f0-8515-1b2ee998c40a');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>


It seems students spend similar amount of times on ALEKS, but they show different progress. That means, it takes different amount of time for students to master a topic on ALEKS. So, let's take a look at the learning rate of each group. 

## Learning Rate


```python
#Let's calculate and add a learning rate column. (Number of topic a student learns per hour)
#Since time values are based on minutes, I will divide the total time by 60 (or multiply the numerator by 60 instead).
df_new["Learning Rate"] = df_new["Learned"]*60/(df_new["Total Time Spent"])

#Let's drop some columns.
df_new = df_new.drop(columns=["School Grade","ALEKS Grade","Target"])
df_new
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
      <th>STAAR Performance</th>
      <th>BOY Mastery</th>
      <th>BOY Mastered Topics</th>
      <th>EOY Mastery</th>
      <th>EOY Mastered Topics</th>
      <th>Total Number of Topics</th>
      <th>Progress</th>
      <th>Learned</th>
      <th>Total Time Spent</th>
      <th>Learning Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Fails</td>
      <td>0.098385</td>
      <td>25.760977</td>
      <td>0.345371</td>
      <td>84.550017</td>
      <td>254.091779</td>
      <td>0.246993</td>
      <td>75.027402</td>
      <td>1726.626896</td>
      <td>2.607190</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Approaches</td>
      <td>0.144783</td>
      <td>40.482748</td>
      <td>0.463920</td>
      <td>126.024093</td>
      <td>281.182629</td>
      <td>0.319150</td>
      <td>103.671921</td>
      <td>1984.748939</td>
      <td>3.134056</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Meets</td>
      <td>0.219267</td>
      <td>68.414085</td>
      <td>0.738133</td>
      <td>169.247887</td>
      <td>304.512676</td>
      <td>0.518870</td>
      <td>117.207243</td>
      <td>1943.140966</td>
      <td>3.619107</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Masters</td>
      <td>0.283454</td>
      <td>99.848048</td>
      <td>0.991198</td>
      <td>215.672177</td>
      <td>350.616954</td>
      <td>0.707753</td>
      <td>126.559620</td>
      <td>1828.738393</td>
      <td>4.152358</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig4 = px.pie(df_new, values='Learning Rate', names='STAAR Performance', 
              color='STAAR Performance', title='Learning Rate by Performance')

fig4.update_traces(textposition='inside', textinfo='percent+label')
fig4.show()
```


<div>                            <div id="f1fdc77d-365f-4010-b587-f34a51fe0da8" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("f1fdc77d-365f-4010-b587-f34a51fe0da8")) {                    Plotly.newPlot(                        "f1fdc77d-365f-4010-b587-f34a51fe0da8",                        [{"customdata":[["Fails"],["Approaches"],["Meets"],["Masters"]],"domain":{"x":[0.0,1.0],"y":[0.0,1.0]},"hovertemplate":"STAAR Performance=%{customdata[0]}<br>Learning Rate=%{value}<extra></extra>","labels":["Fails","Approaches","Meets","Masters"],"legendgroup":"","marker":{"colors":["#636efa","#EF553B","#00cc96","#ab63fa"]},"name":"","showlegend":true,"values":[2.607189844273038,3.134056487399168,3.619106761392127,4.152358389881948],"type":"pie","textinfo":"percent+label","textposition":"inside"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"legend":{"tracegroupgap":0},"title":{"text":"Learning Rate by Performance"}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('f1fdc77d-365f-4010-b587-f34a51fe0da8');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>


Again, there is a strong correlation between the STAAR Performance of students and ALEKS topics learning rate, rather than time spent on ALEKS.

## Addding New Feature


```python
# Let's add this new feature to our training and testing data.
df_train["Learning Rate"] = df_train["Learned"]*60/(df_train["Total Time Spent"])
df_train
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
      <th>School Grade</th>
      <th>ALEKS Grade</th>
      <th>BOY Mastery</th>
      <th>BOY Mastered Topics</th>
      <th>EOY Mastery</th>
      <th>EOY Mastered Topics</th>
      <th>Total Number of Topics</th>
      <th>Progress</th>
      <th>Learned</th>
      <th>Total Time Spent</th>
      <th>Target</th>
      <th>Learning Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9</td>
      <td>9</td>
      <td>0.099150</td>
      <td>35</td>
      <td>0.750708</td>
      <td>265</td>
      <td>353</td>
      <td>0.651558</td>
      <td>218</td>
      <td>2750.700000</td>
      <td>1</td>
      <td>4.755153</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>5</td>
      <td>0.095238</td>
      <td>24</td>
      <td>0.253968</td>
      <td>64</td>
      <td>252</td>
      <td>0.158730</td>
      <td>56</td>
      <td>1676.616667</td>
      <td>0</td>
      <td>2.004036</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>3</td>
      <td>0.445652</td>
      <td>41</td>
      <td>0.967391</td>
      <td>89</td>
      <td>92</td>
      <td>0.521739</td>
      <td>59</td>
      <td>840.216667</td>
      <td>1</td>
      <td>4.213199</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>4</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.178571</td>
      <td>45</td>
      <td>252</td>
      <td>0.178571</td>
      <td>70</td>
      <td>1569.933333</td>
      <td>0</td>
      <td>2.675273</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>5</td>
      <td>0.253968</td>
      <td>64</td>
      <td>0.611111</td>
      <td>154</td>
      <td>252</td>
      <td>0.357143</td>
      <td>110</td>
      <td>1579.483333</td>
      <td>1</td>
      <td>4.178582</td>
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
      <th>9370</th>
      <td>3</td>
      <td>5</td>
      <td>0.609195</td>
      <td>159</td>
      <td>1.896552</td>
      <td>234</td>
      <td>261</td>
      <td>1.287356</td>
      <td>85</td>
      <td>1575.483333</td>
      <td>1</td>
      <td>3.237102</td>
    </tr>
    <tr>
      <th>9371</th>
      <td>6</td>
      <td>6</td>
      <td>0.009804</td>
      <td>3</td>
      <td>0.297386</td>
      <td>91</td>
      <td>306</td>
      <td>0.287582</td>
      <td>125</td>
      <td>2030.900000</td>
      <td>0</td>
      <td>3.692944</td>
    </tr>
    <tr>
      <th>9372</th>
      <td>3</td>
      <td>3</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.301136</td>
      <td>53</td>
      <td>176</td>
      <td>0.301136</td>
      <td>62</td>
      <td>1072.866667</td>
      <td>0</td>
      <td>3.467346</td>
    </tr>
    <tr>
      <th>9373</th>
      <td>6</td>
      <td>6</td>
      <td>0.035948</td>
      <td>11</td>
      <td>0.114379</td>
      <td>35</td>
      <td>306</td>
      <td>0.078431</td>
      <td>24</td>
      <td>777.800000</td>
      <td>0</td>
      <td>1.851376</td>
    </tr>
    <tr>
      <th>9374</th>
      <td>7</td>
      <td>7</td>
      <td>0.538874</td>
      <td>201</td>
      <td>0.970509</td>
      <td>362</td>
      <td>373</td>
      <td>0.431635</td>
      <td>230</td>
      <td>9005.350000</td>
      <td>1</td>
      <td>1.532422</td>
    </tr>
  </tbody>
</table>
<p>9375 rows × 12 columns</p>
</div>




```python
df_test["Learning Rate"] = df_test["Learned"]*60/(df_test["Total Time Spent"])
df_test
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
      <th>School Grade</th>
      <th>ALEKS Grade</th>
      <th>BOY Mastery</th>
      <th>BOY Mastered Topics</th>
      <th>EOY Mastery</th>
      <th>EOY Mastered Topics</th>
      <th>Total Number of Topics</th>
      <th>Progress</th>
      <th>Learned</th>
      <th>Total Time Spent</th>
      <th>Target</th>
      <th>Learning Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>101</th>
      <td>6</td>
      <td>6</td>
      <td>0.022876</td>
      <td>7</td>
      <td>0.049020</td>
      <td>15</td>
      <td>306</td>
      <td>0.026144</td>
      <td>12</td>
      <td>339.966667</td>
      <td>0</td>
      <td>2.117855</td>
    </tr>
    <tr>
      <th>5271</th>
      <td>6</td>
      <td>6</td>
      <td>0.042484</td>
      <td>13</td>
      <td>0.336601</td>
      <td>103</td>
      <td>306</td>
      <td>0.294118</td>
      <td>92</td>
      <td>1760.516667</td>
      <td>1</td>
      <td>3.135443</td>
    </tr>
    <tr>
      <th>5146</th>
      <td>4</td>
      <td>4</td>
      <td>0.190202</td>
      <td>66</td>
      <td>0.475504</td>
      <td>165</td>
      <td>347</td>
      <td>0.285303</td>
      <td>112</td>
      <td>1748.416667</td>
      <td>1</td>
      <td>3.843477</td>
    </tr>
    <tr>
      <th>9219</th>
      <td>3</td>
      <td>3</td>
      <td>0.358696</td>
      <td>33</td>
      <td>0.989130</td>
      <td>91</td>
      <td>92</td>
      <td>0.630435</td>
      <td>72</td>
      <td>1069.366667</td>
      <td>1</td>
      <td>4.039774</td>
    </tr>
    <tr>
      <th>1845</th>
      <td>7</td>
      <td>7</td>
      <td>0.072874</td>
      <td>18</td>
      <td>0.198381</td>
      <td>49</td>
      <td>247</td>
      <td>0.125506</td>
      <td>37</td>
      <td>489.800000</td>
      <td>1</td>
      <td>4.532462</td>
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
      <th>3116</th>
      <td>9</td>
      <td>9</td>
      <td>0.110701</td>
      <td>60</td>
      <td>0.293358</td>
      <td>159</td>
      <td>542</td>
      <td>0.182657</td>
      <td>89</td>
      <td>4016.683333</td>
      <td>0</td>
      <td>1.329455</td>
    </tr>
    <tr>
      <th>2199</th>
      <td>7</td>
      <td>7</td>
      <td>0.085020</td>
      <td>21</td>
      <td>0.226721</td>
      <td>56</td>
      <td>247</td>
      <td>0.141700</td>
      <td>72</td>
      <td>1070.950000</td>
      <td>0</td>
      <td>4.033802</td>
    </tr>
    <tr>
      <th>2503</th>
      <td>4</td>
      <td>4</td>
      <td>0.422680</td>
      <td>82</td>
      <td>0.577320</td>
      <td>112</td>
      <td>194</td>
      <td>0.154639</td>
      <td>29</td>
      <td>1268.850000</td>
      <td>0</td>
      <td>1.371320</td>
    </tr>
    <tr>
      <th>4194</th>
      <td>6</td>
      <td>6</td>
      <td>0.477124</td>
      <td>146</td>
      <td>0.712418</td>
      <td>218</td>
      <td>306</td>
      <td>0.235294</td>
      <td>71</td>
      <td>857.133333</td>
      <td>1</td>
      <td>4.970055</td>
    </tr>
    <tr>
      <th>10183</th>
      <td>9</td>
      <td>9</td>
      <td>0.039660</td>
      <td>14</td>
      <td>0.889518</td>
      <td>314</td>
      <td>353</td>
      <td>0.849858</td>
      <td>345</td>
      <td>7239.133333</td>
      <td>1</td>
      <td>2.859458</td>
    </tr>
  </tbody>
</table>
<p>2344 rows × 12 columns</p>
</div>



## Scaling the Data

We need to normalize the data since the features have different scales.


```python
#Let's normalize the training data first.
normal = MinMaxScaler()
df_train_normal = normal.fit_transform(df_train)
df_train_normal = pd.DataFrame(df_train_normal, columns=df_train.columns)
df_train_normal
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
      <th>School Grade</th>
      <th>ALEKS Grade</th>
      <th>BOY Mastery</th>
      <th>BOY Mastered Topics</th>
      <th>EOY Mastery</th>
      <th>EOY Mastered Topics</th>
      <th>Total Number of Topics</th>
      <th>Progress</th>
      <th>Learned</th>
      <th>Total Time Spent</th>
      <th>Target</th>
      <th>Learning Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.000000</td>
      <td>0.750</td>
      <td>0.101238</td>
      <td>0.064457</td>
      <td>0.373580</td>
      <td>0.289967</td>
      <td>0.351259</td>
      <td>0.366851</td>
      <td>0.370307</td>
      <td>0.186912</td>
      <td>1.0</td>
      <td>0.045615</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.333333</td>
      <td>0.250</td>
      <td>0.097243</td>
      <td>0.044199</td>
      <td>0.124504</td>
      <td>0.068357</td>
      <td>0.235698</td>
      <td>0.086937</td>
      <td>0.093857</td>
      <td>0.104993</td>
      <td>0.0</td>
      <td>0.019092</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.455034</td>
      <td>0.075506</td>
      <td>0.482229</td>
      <td>0.095921</td>
      <td>0.052632</td>
      <td>0.293117</td>
      <td>0.098976</td>
      <td>0.041202</td>
      <td>1.0</td>
      <td>0.040390</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.166667</td>
      <td>0.125</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.086698</td>
      <td>0.047409</td>
      <td>0.235698</td>
      <td>0.098206</td>
      <td>0.117747</td>
      <td>0.096857</td>
      <td>0.0</td>
      <td>0.025563</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.333333</td>
      <td>0.250</td>
      <td>0.259315</td>
      <td>0.117864</td>
      <td>0.303583</td>
      <td>0.167585</td>
      <td>0.235698</td>
      <td>0.199630</td>
      <td>0.186007</td>
      <td>0.097585</td>
      <td>1.0</td>
      <td>0.040057</td>
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
      <th>9370</th>
      <td>0.000000</td>
      <td>0.250</td>
      <td>0.622021</td>
      <td>0.292818</td>
      <td>0.948129</td>
      <td>0.255788</td>
      <td>0.245995</td>
      <td>0.727969</td>
      <td>0.143345</td>
      <td>0.097280</td>
      <td>1.0</td>
      <td>0.030980</td>
    </tr>
    <tr>
      <th>9371</th>
      <td>0.500000</td>
      <td>0.375</td>
      <td>0.010010</td>
      <td>0.005525</td>
      <td>0.146274</td>
      <td>0.098126</td>
      <td>0.297483</td>
      <td>0.160121</td>
      <td>0.211604</td>
      <td>0.132014</td>
      <td>0.0</td>
      <td>0.035375</td>
    </tr>
    <tr>
      <th>9372</th>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.148155</td>
      <td>0.056229</td>
      <td>0.148741</td>
      <td>0.167820</td>
      <td>0.104096</td>
      <td>0.058946</td>
      <td>0.0</td>
      <td>0.033200</td>
    </tr>
    <tr>
      <th>9373</th>
      <td>0.500000</td>
      <td>0.375</td>
      <td>0.036705</td>
      <td>0.020258</td>
      <td>0.054511</td>
      <td>0.036384</td>
      <td>0.297483</td>
      <td>0.041329</td>
      <td>0.039249</td>
      <td>0.036441</td>
      <td>0.0</td>
      <td>0.017620</td>
    </tr>
    <tr>
      <th>9374</th>
      <td>0.666667</td>
      <td>0.500</td>
      <td>0.550219</td>
      <td>0.370166</td>
      <td>0.483792</td>
      <td>0.396913</td>
      <td>0.374142</td>
      <td>0.241940</td>
      <td>0.390785</td>
      <td>0.663949</td>
      <td>1.0</td>
      <td>0.014545</td>
    </tr>
  </tbody>
</table>
<p>9375 rows × 12 columns</p>
</div>




```python
#Let's normalize the testing data next.
normal = MinMaxScaler()
df_test_normal = normal.fit_transform(df_test)
df_test_normal = pd.DataFrame(df_test_normal, columns=df_test.columns)
df_test_normal
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
      <th>School Grade</th>
      <th>ALEKS Grade</th>
      <th>BOY Mastery</th>
      <th>BOY Mastered Topics</th>
      <th>EOY Mastery</th>
      <th>EOY Mastered Topics</th>
      <th>Total Number of Topics</th>
      <th>Progress</th>
      <th>Learned</th>
      <th>Total Time Spent</th>
      <th>Target</th>
      <th>Learning Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.500000</td>
      <td>0.375</td>
      <td>0.025064</td>
      <td>0.011309</td>
      <td>0.017552</td>
      <td>0.011198</td>
      <td>0.274232</td>
      <td>0.011336</td>
      <td>0.020677</td>
      <td>0.003096</td>
      <td>0.0</td>
      <td>0.035993</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.500000</td>
      <td>0.375</td>
      <td>0.046547</td>
      <td>0.021002</td>
      <td>0.162368</td>
      <td>0.109742</td>
      <td>0.274232</td>
      <td>0.158184</td>
      <td>0.171053</td>
      <td>0.112640</td>
      <td>1.0</td>
      <td>0.054509</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.166667</td>
      <td>0.125</td>
      <td>0.208395</td>
      <td>0.106624</td>
      <td>0.232315</td>
      <td>0.179171</td>
      <td>0.322695</td>
      <td>0.153353</td>
      <td>0.208647</td>
      <td>0.111707</td>
      <td>1.0</td>
      <td>0.067393</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.393006</td>
      <td>0.053312</td>
      <td>0.490960</td>
      <td>0.096305</td>
      <td>0.021277</td>
      <td>0.342482</td>
      <td>0.133459</td>
      <td>0.059343</td>
      <td>1.0</td>
      <td>0.070965</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.666667</td>
      <td>0.500</td>
      <td>0.079845</td>
      <td>0.029079</td>
      <td>0.092765</td>
      <td>0.049272</td>
      <td>0.204492</td>
      <td>0.065786</td>
      <td>0.067669</td>
      <td>0.014650</td>
      <td>1.0</td>
      <td>0.079930</td>
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
      <th>2339</th>
      <td>1.000000</td>
      <td>0.750</td>
      <td>0.121290</td>
      <td>0.096931</td>
      <td>0.140593</td>
      <td>0.172452</td>
      <td>0.553191</td>
      <td>0.097104</td>
      <td>0.165414</td>
      <td>0.286622</td>
      <td>0.0</td>
      <td>0.021646</td>
    </tr>
    <tr>
      <th>2340</th>
      <td>0.666667</td>
      <td>0.500</td>
      <td>0.093153</td>
      <td>0.033926</td>
      <td>0.107036</td>
      <td>0.057111</td>
      <td>0.204492</td>
      <td>0.074660</td>
      <td>0.133459</td>
      <td>0.059465</td>
      <td>0.0</td>
      <td>0.070856</td>
    </tr>
    <tr>
      <th>2341</th>
      <td>0.166667</td>
      <td>0.125</td>
      <td>0.463111</td>
      <td>0.132472</td>
      <td>0.283586</td>
      <td>0.119821</td>
      <td>0.141844</td>
      <td>0.081751</td>
      <td>0.052632</td>
      <td>0.074726</td>
      <td>0.0</td>
      <td>0.022408</td>
    </tr>
    <tr>
      <th>2342</th>
      <td>0.500000</td>
      <td>0.375</td>
      <td>0.522762</td>
      <td>0.235864</td>
      <td>0.351617</td>
      <td>0.238522</td>
      <td>0.274232</td>
      <td>0.125949</td>
      <td>0.131579</td>
      <td>0.042977</td>
      <td>1.0</td>
      <td>0.087893</td>
    </tr>
    <tr>
      <th>2343</th>
      <td>1.000000</td>
      <td>0.750</td>
      <td>0.043454</td>
      <td>0.022617</td>
      <td>0.440799</td>
      <td>0.346025</td>
      <td>0.329787</td>
      <td>0.462725</td>
      <td>0.646617</td>
      <td>0.535118</td>
      <td>1.0</td>
      <td>0.049487</td>
    </tr>
  </tbody>
</table>
<p>2344 rows × 12 columns</p>
</div>



## Modeling the Data


```python
#Models that will be used
model_list = [LogisticRegression, GaussianNB, DecisionTreeClassifier, LinearSVC, KNeighborsClassifier,
              RandomForestClassifier, GradientBoostingClassifier, SGDClassifier, XGBClassifier,
             LGBMClassifier, LinearDiscriminantAnalysis, AdaBoostClassifier]

#Name of the models
model_names = ['Logistic Regression', 'Gaussian Naive Bayes', 'Decision Tree Classifier', 'Linear SVC', 
               'KNeighbors Classifier', 'Random Forest Classifier', 'Gradient Boosting Classifier', 'SGD Classifier',
              'XGB Classifier', 'LGBM Classifier','Linear Discriminant Analysis','AdaBoost Classifier']
```

Let's create a function that takes the training set as input and gives the accuracy of each model as output.


```python
def train_clf_model(clf_list, clf_names, x_train, y_train):
    """
    This function returns the accuracy scores of the training data for each models used, in descending order.

    Parameters
    ----------
    clf_list : list
        List of the models to be used.
    clf_names : list
        List of the model names.
    x_train : dataframe values
        Feature values of the training set.
    y_train : list
        Target values of the training set.
        
    Returns
    -------
    Pandas DataFrame
        Accuracy scores of the training data for each model.
    """    
    
    training_scores = []
    
    for index, clf in enumerate(clf_list):
        model = clf()
         
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=random_seed) #StratifiedKFold(n_splits=4, shuffle=True)
        
        #keep the mean of cross validation scores of the training set
        scores = cross_val_score(model, x_train, y_train, scoring='accuracy', cv=cv)
        training_scores.append(scores.mean())
    
    #create a dataframe to present results
    data = {'Classifier Names': clf_names,
            'Accuracy Score': training_scores}
    
    #maximum 4 decimal points should be sufficient to compare score values
    pd.options.display.float_format = '{:.4f}'.format
    df = pd.DataFrame(data)
    
    return df.sort_values(by='Accuracy Score', ascending=False).reset_index(drop=True)
```


```python
X_train = df_train_normal.drop(columns="Target")
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
      <th>School Grade</th>
      <th>ALEKS Grade</th>
      <th>BOY Mastery</th>
      <th>BOY Mastered Topics</th>
      <th>EOY Mastery</th>
      <th>EOY Mastered Topics</th>
      <th>Total Number of Topics</th>
      <th>Progress</th>
      <th>Learned</th>
      <th>Total Time Spent</th>
      <th>Learning Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0000</td>
      <td>0.7500</td>
      <td>0.1012</td>
      <td>0.0645</td>
      <td>0.3736</td>
      <td>0.2900</td>
      <td>0.3513</td>
      <td>0.3669</td>
      <td>0.3703</td>
      <td>0.1869</td>
      <td>0.0456</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.3333</td>
      <td>0.2500</td>
      <td>0.0972</td>
      <td>0.0442</td>
      <td>0.1245</td>
      <td>0.0684</td>
      <td>0.2357</td>
      <td>0.0869</td>
      <td>0.0939</td>
      <td>0.1050</td>
      <td>0.0191</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.4550</td>
      <td>0.0755</td>
      <td>0.4822</td>
      <td>0.0959</td>
      <td>0.0526</td>
      <td>0.2931</td>
      <td>0.0990</td>
      <td>0.0412</td>
      <td>0.0404</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.1667</td>
      <td>0.1250</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0867</td>
      <td>0.0474</td>
      <td>0.2357</td>
      <td>0.0982</td>
      <td>0.1177</td>
      <td>0.0969</td>
      <td>0.0256</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.3333</td>
      <td>0.2500</td>
      <td>0.2593</td>
      <td>0.1179</td>
      <td>0.3036</td>
      <td>0.1676</td>
      <td>0.2357</td>
      <td>0.1996</td>
      <td>0.1860</td>
      <td>0.0976</td>
      <td>0.0401</td>
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
    </tr>
    <tr>
      <th>9370</th>
      <td>0.0000</td>
      <td>0.2500</td>
      <td>0.6220</td>
      <td>0.2928</td>
      <td>0.9481</td>
      <td>0.2558</td>
      <td>0.2460</td>
      <td>0.7280</td>
      <td>0.1433</td>
      <td>0.0973</td>
      <td>0.0310</td>
    </tr>
    <tr>
      <th>9371</th>
      <td>0.5000</td>
      <td>0.3750</td>
      <td>0.0100</td>
      <td>0.0055</td>
      <td>0.1463</td>
      <td>0.0981</td>
      <td>0.2975</td>
      <td>0.1601</td>
      <td>0.2116</td>
      <td>0.1320</td>
      <td>0.0354</td>
    </tr>
    <tr>
      <th>9372</th>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.1482</td>
      <td>0.0562</td>
      <td>0.1487</td>
      <td>0.1678</td>
      <td>0.1041</td>
      <td>0.0589</td>
      <td>0.0332</td>
    </tr>
    <tr>
      <th>9373</th>
      <td>0.5000</td>
      <td>0.3750</td>
      <td>0.0367</td>
      <td>0.0203</td>
      <td>0.0545</td>
      <td>0.0364</td>
      <td>0.2975</td>
      <td>0.0413</td>
      <td>0.0392</td>
      <td>0.0364</td>
      <td>0.0176</td>
    </tr>
    <tr>
      <th>9374</th>
      <td>0.6667</td>
      <td>0.5000</td>
      <td>0.5502</td>
      <td>0.3702</td>
      <td>0.4838</td>
      <td>0.3969</td>
      <td>0.3741</td>
      <td>0.2419</td>
      <td>0.3908</td>
      <td>0.6639</td>
      <td>0.0145</td>
    </tr>
  </tbody>
</table>
<p>9375 rows × 11 columns</p>
</div>




```python
Y_train = df_train_normal["Target"]
Y_train
```




    0       1.0
    1       0.0
    2       1.0
    3       0.0
    4       1.0
           ... 
    9370    1.0
    9371    0.0
    9372    0.0
    9373    0.0
    9374    1.0
    Name: Target, Length: 9375, dtype: float64




```python
X_test = df_test_normal.drop(columns="Target")
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
      <th>School Grade</th>
      <th>ALEKS Grade</th>
      <th>BOY Mastery</th>
      <th>BOY Mastered Topics</th>
      <th>EOY Mastery</th>
      <th>EOY Mastered Topics</th>
      <th>Total Number of Topics</th>
      <th>Progress</th>
      <th>Learned</th>
      <th>Total Time Spent</th>
      <th>Learning Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.5000</td>
      <td>0.3750</td>
      <td>0.0251</td>
      <td>0.0113</td>
      <td>0.0176</td>
      <td>0.0112</td>
      <td>0.2742</td>
      <td>0.0113</td>
      <td>0.0207</td>
      <td>0.0031</td>
      <td>0.0360</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.5000</td>
      <td>0.3750</td>
      <td>0.0465</td>
      <td>0.0210</td>
      <td>0.1624</td>
      <td>0.1097</td>
      <td>0.2742</td>
      <td>0.1582</td>
      <td>0.1711</td>
      <td>0.1126</td>
      <td>0.0545</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.1667</td>
      <td>0.1250</td>
      <td>0.2084</td>
      <td>0.1066</td>
      <td>0.2323</td>
      <td>0.1792</td>
      <td>0.3227</td>
      <td>0.1534</td>
      <td>0.2086</td>
      <td>0.1117</td>
      <td>0.0674</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.3930</td>
      <td>0.0533</td>
      <td>0.4910</td>
      <td>0.0963</td>
      <td>0.0213</td>
      <td>0.3425</td>
      <td>0.1335</td>
      <td>0.0593</td>
      <td>0.0710</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.6667</td>
      <td>0.5000</td>
      <td>0.0798</td>
      <td>0.0291</td>
      <td>0.0928</td>
      <td>0.0493</td>
      <td>0.2045</td>
      <td>0.0658</td>
      <td>0.0677</td>
      <td>0.0147</td>
      <td>0.0799</td>
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
    </tr>
    <tr>
      <th>2339</th>
      <td>1.0000</td>
      <td>0.7500</td>
      <td>0.1213</td>
      <td>0.0969</td>
      <td>0.1406</td>
      <td>0.1725</td>
      <td>0.5532</td>
      <td>0.0971</td>
      <td>0.1654</td>
      <td>0.2866</td>
      <td>0.0216</td>
    </tr>
    <tr>
      <th>2340</th>
      <td>0.6667</td>
      <td>0.5000</td>
      <td>0.0932</td>
      <td>0.0339</td>
      <td>0.1070</td>
      <td>0.0571</td>
      <td>0.2045</td>
      <td>0.0747</td>
      <td>0.1335</td>
      <td>0.0595</td>
      <td>0.0709</td>
    </tr>
    <tr>
      <th>2341</th>
      <td>0.1667</td>
      <td>0.1250</td>
      <td>0.4631</td>
      <td>0.1325</td>
      <td>0.2836</td>
      <td>0.1198</td>
      <td>0.1418</td>
      <td>0.0818</td>
      <td>0.0526</td>
      <td>0.0747</td>
      <td>0.0224</td>
    </tr>
    <tr>
      <th>2342</th>
      <td>0.5000</td>
      <td>0.3750</td>
      <td>0.5228</td>
      <td>0.2359</td>
      <td>0.3516</td>
      <td>0.2385</td>
      <td>0.2742</td>
      <td>0.1259</td>
      <td>0.1316</td>
      <td>0.0430</td>
      <td>0.0879</td>
    </tr>
    <tr>
      <th>2343</th>
      <td>1.0000</td>
      <td>0.7500</td>
      <td>0.0435</td>
      <td>0.0226</td>
      <td>0.4408</td>
      <td>0.3460</td>
      <td>0.3298</td>
      <td>0.4627</td>
      <td>0.6466</td>
      <td>0.5351</td>
      <td>0.0495</td>
    </tr>
  </tbody>
</table>
<p>2344 rows × 11 columns</p>
</div>




```python
Y_test = df_test_normal["Target"]
Y_test
```




    0      0.0000
    1      1.0000
    2      1.0000
    3      1.0000
    4      1.0000
            ...  
    2339   0.0000
    2340   0.0000
    2341   0.0000
    2342   1.0000
    2343   1.0000
    Name: Target, Length: 2344, dtype: float64




```python
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
      <th>Accuracy Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Gradient Boosting Classifier</td>
      <td>0.8331</td>
    </tr>
    <tr>
      <th>1</th>
      <td>LGBM Classifier</td>
      <td>0.8331</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AdaBoost Classifier</td>
      <td>0.8271</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Random Forest Classifier</td>
      <td>0.8271</td>
    </tr>
    <tr>
      <th>4</th>
      <td>KNeighbors Classifier</td>
      <td>0.8234</td>
    </tr>
    <tr>
      <th>5</th>
      <td>XGB Classifier</td>
      <td>0.8227</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Linear SVC</td>
      <td>0.8020</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Logistic Regression</td>
      <td>0.7987</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Linear Discriminant Analysis</td>
      <td>0.7963</td>
    </tr>
    <tr>
      <th>9</th>
      <td>SGD Classifier</td>
      <td>0.7936</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Decision Tree Classifier</td>
      <td>0.7707</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Gaussian Naive Bayes</td>
      <td>0.6535</td>
    </tr>
  </tbody>
</table>
</div>



Gradient Boosting Classifier has the highest accuracy.


### Hyperparameter Tuning with Grid Search


```python
params = {
#     'n_estimators':range(100,1000,100),
#     'subsample':[0.6,0.7,0.75,0.8,0.85,0.9, 1.0],
#     'max_features':range(1,10),
#     'min_samples_split':range(1,10),
#     'min_samples_leaf':range(1,10),
    'max_depth':range(1,10),
}
gsearch = GridSearchCV(estimator = GradientBoostingClassifier(), 
param_grid = params, scoring='accuracy',n_jobs=2, cv=5)
gsearch.fit(X_train, Y_train)
gsearch.best_params_, gsearch.best_score_
```




    ({'max_depth': 5}, 0.8344533333333333)



So, max_depth=5 will give us better score than the default value of 3.

We have tried to optimize other parameters but they did not improve the model accuracy.


```python
#model selection
model_1 = GradientBoostingClassifier()
model = GradientBoostingClassifier(max_depth=5)

#fitting it on the training set
model_1.fit(X_train, Y_train)
model.fit(X_train, Y_train)

#predict the y values based on X_test with and without the hyperparameter tuning:
y_pred_1 = model_1.predict(X_test)
y_pred = model.predict(X_test)    

#accuracy of the test set
print('The accuracy score without hyperparameter optimization is {:.2f}%'.format(accuracy_score(Y_test, y_pred_1)*100))
print('The accuracy score with the hyperparameter optimization is {:.2f}%'.format(accuracy_score(Y_test, y_pred)*100))
```

    The accuracy score without hyperparameter optimization is 82.85%
    The accuracy score with the hyperparameter optimization is 83.23%
    

The accuracy went up by 0.4% with the hyperparameter optimization.

### Confusion Matrix

Let's take a look at the Confusion Matrix.


```python
print("Confusion Matrix:")
cm = confusion_matrix(Y_test, y_pred)
print(cm)

plot_confusion_matrix(model, X_test, Y_test)  
#ConfusionMatrixDisplay.from_predictions(Y_test, y_pred)
plt.title('Confusion matrix')
#plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.rcParams["figure.figsize"]=(6,6)
plt.grid(False)
plt.show()
print(classification_report(Y_test, y_pred))
```

    Confusion Matrix:
    [[ 380  219]
     [ 174 1571]]
    


    
![png](output_62_1.png)
    


                  precision    recall  f1-score   support
    
             0.0       0.69      0.63      0.66       599
             1.0       0.88      0.90      0.89      1745
    
        accuracy                           0.83      2344
       macro avg       0.78      0.77      0.77      2344
    weighted avg       0.83      0.83      0.83      2344
    
    

Our model seems to predict one class better than other due to imbalanced target values.
Let's take a look at the feature importances:


```python
model.feature_importances_
```




    array([0.03255987, 0.19570894, 0.04636211, 0.02812671, 0.18113919,
           0.24337241, 0.07580116, 0.03990442, 0.03353915, 0.08398665,
           0.03949939])




```python
#Let's look at feature importances
features = X_test.columns
sorted_idx = model.feature_importances_.argsort()
plt.barh(features[sorted_idx], model.feature_importances_[sorted_idx])
plt.xlabel("Feature Importance")
```




    Text(0.5, 0, 'Feature Importance')




    
![png](output_65_1.png)
    


### Findings

So, the number of topics mastered by students at the end of the year is the most important feature to predict students' STAAR pass/fail determination. We predict that the more topics student mastered, the more likely they pass the STAAR. 

Second most important factor is the ALEKS grade. That means students who work on upper grade level math subjects perform better on the STAAR than other students.

Third important feature is End of Year Mastery Percentage. Students with higher mastery percentage tend to do better on STAAR.

### Different Tresholds

We would like to predict students who will fail STAAR (0 cases) as much as possible because our goal is to identify potential students who will likely to fail STAAR test based on their ALEKS performance. That way, we can intervene earlier.

That requires higher Recall scores rather than Precision since we would like to catch as much relevant instances as possible. 

We can ply with the treshold value to increase recall of 0 cases.


```python
threshold = 0.62
y_pred = (model.predict_proba(X_test)[:, 1] > threshold).astype('float')
print(confusion_matrix(Y_test, y_pred))
print(classification_report(Y_test, y_pred))
```

    [[ 451  148]
     [ 262 1483]]
                  precision    recall  f1-score   support
    
             0.0       0.63      0.75      0.69       599
             1.0       0.91      0.85      0.88      1745
    
        accuracy                           0.83      2344
       macro avg       0.77      0.80      0.78      2344
    weighted avg       0.84      0.83      0.83      2344
    
    

As you see, without changing the accuracy score (still 83%), we were able to increase recall of 0 cases by changing the treshold from default 0.5 to 0.62. Now, our model cathces 451 of 599 true positive cases as opposed to 380 over 599.

## Dimensionality Reduction with PCA

Let's try Principal Component Analysis to see of it helps model perform better.


```python
# Generate all the principal components
pca = PCA()
X_train_pc = pca.fit_transform(X_train)
# X_test_pc = pca.transform(X_test)
np.set_printoptions(suppress=True)
print(pca.explained_variance_ratio_)
```

    [0.49273665 0.32040925 0.08232128 0.06034154 0.02515731 0.01123495
     0.00341417 0.00211674 0.0013453  0.0009228  0.00000002]
    

Let's visualize the PCA with 2 components.


```python
def plot_2d_space(X, y, label='Classes'):   
    colors = ['red', 'green']
    markers = ['o', 's']
    for l, c, m in zip(np.unique(y), colors, markers):
        plt.scatter(
            X[y==l, 0],
            X[y==l, 1],
            c=c, label=l, marker=m,
            edgecolor='none', alpha=0.5
        )
    plt.title(label)
    plt.legend(loc='upper right')
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.show()
```


```python
plot_2d_space(X_train_pc, Y_train, '2 PCA components')
```


    
![png](output_74_0.png)
    


### Choosing the number of components


```python
plt.plot(range(1,12), np.cumsum(pca.explained_variance_ratio_), lw=3)
plt.plot([1, 11], [0.95, 0.95], color="red", lw=2, linestyle="--")
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
```




    Text(0, 0.5, 'cumulative explained variance')




    
![png](output_76_1.png)
    


As you see, 4 components contain approximately 95% of the variance.


```python
pca = PCA(n_components=4)

X_train_pc = pca.fit_transform(X_train)

model.fit(X_train_pc, Y_train)

X_test_pc = pca.transform(X_test)

y_pred_pc = model.predict(X_test_pc) 

#accuracy of the test set
print('The accuracy score with PCA is {:.2f}%'.format(accuracy_score(Y_test, y_pred_pc)*100))
```

    The accuracy score with PCA is 82.30%
    

### PCA with Kernel

Let's try PCA with RBF Kernel.


```python
kpca = KernelPCA(n_components=4, kernel ='rbf')
X_train_kpca = kpca.fit_transform(X_train)

model.fit(X_train_kpca, Y_train)

X_test_kpca = kpca.transform(X_test)

#predict the y values based on X_test with and without the hyperparameter tuning:
y_pred_kpca = model.predict(X_test_kpca) 

#accuracy of the test set
print('The accuracy score with PCA is {:.2f}%'.format(accuracy_score(Y_test, y_pred_kpca)*100))
```

    The accuracy score with PCA is 82.51%
    

The best score with PCA is 82.51% which is close to 83.23%.

## Addressing Imbalanced Classes


```python
# Display class counts
df_train_normal.Target.value_counts()
```




    1.0000    6945
    0.0000    2430
    Name: Target, dtype: int64



As you see, we have a datasets with a disproportionate ratio of observations in each class. Therefore accuracy score can be a misleading metric to evaluate model performance. Let's try to balance the dataset with 2 different methods.

### 1. Let's Down-sample Majority Class


```python
# Separate majority and minority classes
df_minority = df_train_normal[df_train_normal.Target==0]
df_majority = df_train_normal[df_train_normal.Target==1]
 
# Downsample majority class
df_majority_downsampled = resample(df_majority, 
                                 replace=False,    # sample without replacement
                                 n_samples=2430,     # to match minority class
                                 random_state=random_seed) # reproducible results
 
# Combine minority class with downsampled majority class
df_downsampled = pd.concat([df_majority_downsampled, df_minority])
 
# Display new class counts
df_downsampled.Target.value_counts()
```




    1.0000    2430
    0.0000    2430
    Name: Target, dtype: int64




```python
# Separate input features (X) and target variable (y)
y_d = df_downsampled.Target
X_d = df_downsampled.drop('Target', axis=1)

train_clf_model(model_list, model_names, X_d, y_d)
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
      <th>Accuracy Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Gradient Boosting Classifier</td>
      <td>0.7962</td>
    </tr>
    <tr>
      <th>1</th>
      <td>LGBM Classifier</td>
      <td>0.7905</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AdaBoost Classifier</td>
      <td>0.7886</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Random Forest Classifier</td>
      <td>0.7876</td>
    </tr>
    <tr>
      <th>4</th>
      <td>XGB Classifier</td>
      <td>0.7799</td>
    </tr>
    <tr>
      <th>5</th>
      <td>KNeighbors Classifier</td>
      <td>0.7778</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Linear SVC</td>
      <td>0.7720</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Logistic Regression</td>
      <td>0.7700</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Linear Discriminant Analysis</td>
      <td>0.7638</td>
    </tr>
    <tr>
      <th>9</th>
      <td>SGD Classifier</td>
      <td>0.7578</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Decision Tree Classifier</td>
      <td>0.7283</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Gaussian Naive Bayes</td>
      <td>0.7106</td>
    </tr>
  </tbody>
</table>
</div>



It seems that the accuracy decreased! Let's predict the testing results.


```python
#model selection
model_d = GradientBoostingClassifier()

#fitting it on the training set
model_d.fit(X_d, y_d)

#predict the y values based on X_test
y_pred_d = model_d.predict(X_test)
        
#accuracy of the test set
print('The accuracy score is {:.2f}%'.format(accuracy_score(Y_test, y_pred_d)*100))
```

    The accuracy score is 78.16%
    

The accuracy scores actually went down! So downsampling majority class did not work!

### 2. Let's Up-sample Minority Class


```python
df_minority = df_train_normal[df_train_normal.Target==0]
df_majority = df_train_normal[df_train_normal.Target==1]
 
# Upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=6945,    # to match majority class
                                 random_state=random_seed) # reproducible results
 
# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])
 
# Display new class counts
df_upsampled.Target.value_counts()
```




    1.0000    6945
    0.0000    6945
    Name: Target, dtype: int64



Now, we have the same number of observations for in each class.


```python
# Separate input features (X) and target variable (y)
y_u = df_upsampled.Target
X_u = df_upsampled.drop('Target', axis=1)

train_clf_model(model_list, model_names, X_u, y_u)
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
      <th>Accuracy Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Random Forest Classifier</td>
      <td>0.9154</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Decision Tree Classifier</td>
      <td>0.8970</td>
    </tr>
    <tr>
      <th>2</th>
      <td>XGB Classifier</td>
      <td>0.8945</td>
    </tr>
    <tr>
      <th>3</th>
      <td>LGBM Classifier</td>
      <td>0.8651</td>
    </tr>
    <tr>
      <th>4</th>
      <td>KNeighbors Classifier</td>
      <td>0.8339</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Gradient Boosting Classifier</td>
      <td>0.8166</td>
    </tr>
    <tr>
      <th>6</th>
      <td>AdaBoost Classifier</td>
      <td>0.8014</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Linear SVC</td>
      <td>0.7739</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Logistic Regression</td>
      <td>0.7719</td>
    </tr>
    <tr>
      <th>9</th>
      <td>SGD Classifier</td>
      <td>0.7655</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Linear Discriminant Analysis</td>
      <td>0.7641</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Gaussian Naive Bayes</td>
      <td>0.7087</td>
    </tr>
  </tbody>
</table>
</div>



The Random Forest Classifier did a better job on the training set after upsampling the minority class!

Let's see the predictions.


```python
#model selection
model_u = RandomForestClassifier()

#fitting it on the training set
model_u.fit(X_u, y_u)

#predict the y values based on X_test with and without the hyperparameter tuning:
y_pred_u = model_u.predict(X_test)    

#accuracy of the test set
print('The accuracy score is {:.2f}%'.format(accuracy_score(Y_test, y_pred_u)*100))
```

    The accuracy score is 83.11%
    

The accuracy has not improved. 


```python
#Random Forest Confusion Matrix
print("Confusion Matrix for Random Forest Classifier:")
cm = confusion_matrix(Y_test, y_pred_u)
print(cm)
print(classification_report(Y_test, y_pred_u))
```

    Confusion Matrix for Random Forest Classifier:
    [[ 377  222]
     [ 174 1571]]
                  precision    recall  f1-score   support
    
             0.0       0.68      0.63      0.66       599
             1.0       0.88      0.90      0.89      1745
    
        accuracy                           0.83      2344
       macro avg       0.78      0.76      0.77      2344
    weighted avg       0.83      0.83      0.83      2344
    
    

### Area Under the ROC Curve

Since we have an imbalanced dataset, we can use other performance metrics for evaluating the model.

For a general-purpose metric for classification, Area Under ROC Curve (AUROC) is generally recommended.

AUROC represents the likelihood of the model distinguishing observations from two classes.


```python
#model selection
model = GradientBoostingClassifier(max_depth=5)

#fitting it on the training set
model.fit(X_train, Y_train)

#predict the y values based on X_test with and without the hyperparameter tuning:
y_pred = model.predict(X_test)    

#accuracy of the test set
print('The accuracy score with the hyperparameter optimization is {:.2f}%'.format(accuracy_score(Y_test, y_pred)*100))
```

    The accuracy score with the hyperparameter optimization is 83.28%
    


```python
predicted = model.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(Y_test, predicted[:,1])
```


```python
plt.figure()
lw = 3
plt.plot(
    fpr,
    tpr,
    color="darkorange",
    lw=lw,
    label="ROC curve (area = %0.2f)" % roc_auc_score(Y_test, predicted[:,1]),
)
plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.xlim([0.0, 1.02])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic")
plt.legend(loc="lower right")
plt.show()
```


    
![png](output_102_0.png)
    


So, our score now is 88% based on AUROC!

### Precision-Recall curve


Another alternative metric is Precision Recall Curve.


```python
precision, recall, thresholds = precision_recall_curve(Y_test, predicted[:,1])
```


```python
plt.figure()
lw = 2
plt.plot(
    recall,
    precision,
    color="green",
    lw=lw,
    label="AUC: %0.2f" % auc(recall, precision),
)
plt.plot([0, 1], [0.5, 0.5], color="navy", lw=lw, linestyle="--")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend(loc="lower left")
plt.show()
```


    
![png](output_107_0.png)
    


The the area under the Precision-Recall curve is 0.96, so our score is 96%.


```python

```
