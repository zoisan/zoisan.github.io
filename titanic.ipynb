{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "2dfdf718",
   "metadata": {},
   "source": [
    "# Kaggle: Titanic - Machine Learning from Disaster"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0a3fc509",
   "metadata": {},
   "source": [
    "ismailsavruk@gmail.com © 2022\n",
    "\n",
    "I'll use machine learning to create a model that predicts which passengers survived the Titanic shipwreck.\n",
    "\n",
    "Dataset from: https://www.kaggle.com/competitions/titanic/data"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b8e19778",
   "metadata": {},
   "source": [
    "## A. Getting Started\n",
    "\n",
    "### Loading  Libraries"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b0d75a5c",
   "metadata": {},
   "source": [
    "Let's start by importing necessary libraries first."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "136bb4dd",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load libraries\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "from pandas import read_csv\n",
    "from pandas.plotting import scatter_matrix\n",
    "\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.model_selection import cross_val_score\n",
    "from sklearn.model_selection import StratifiedKFold\n",
    "from sklearn.metrics import classification_report\n",
    "from sklearn.metrics import confusion_matrix\n",
    "from sklearn.metrics import accuracy_score\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "from sklearn.discriminant_analysis import LinearDiscriminantAnalysis\n",
    "from sklearn.naive_bayes import GaussianNB\n",
    "from sklearn.svm import SVC\n",
    "from sklearn import linear_model\n",
    "from sklearn import svm \n",
    "\n",
    "from sklearn.svm import LinearSVC\n",
    "from sklearn.pipeline import make_pipeline\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9abf62c1",
   "metadata": {},
   "source": [
    "### Loading  Dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "c82fb432",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load dataset\n",
    "titanic = pd.read_csv('Train.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "76e414b3",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Embarked</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Braund, Mr. Owen Harris</td>\n",
       "      <td>male</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>A/5 21171</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>\n",
       "      <td>female</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>PC 17599</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>C85</td>\n",
       "      <td>C</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>Heikkinen, Miss. Laina</td>\n",
       "      <td>female</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>STON/O2. 3101282</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>\n",
       "      <td>female</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>113803</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>C123</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Allen, Mr. William Henry</td>\n",
       "      <td>male</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>373450</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>886</th>\n",
       "      <td>887</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>Montvila, Rev. Juozas</td>\n",
       "      <td>male</td>\n",
       "      <td>27.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>211536</td>\n",
       "      <td>13.0000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>887</th>\n",
       "      <td>888</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Graham, Miss. Margaret Edith</td>\n",
       "      <td>female</td>\n",
       "      <td>19.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>112053</td>\n",
       "      <td>30.0000</td>\n",
       "      <td>B42</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>888</th>\n",
       "      <td>889</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Johnston, Miss. Catherine Helen \"Carrie\"</td>\n",
       "      <td>female</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>W./C. 6607</td>\n",
       "      <td>23.4500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>889</th>\n",
       "      <td>890</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Behr, Mr. Karl Howell</td>\n",
       "      <td>male</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>111369</td>\n",
       "      <td>30.0000</td>\n",
       "      <td>C148</td>\n",
       "      <td>C</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>890</th>\n",
       "      <td>891</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Dooley, Mr. Patrick</td>\n",
       "      <td>male</td>\n",
       "      <td>32.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>370376</td>\n",
       "      <td>7.7500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Q</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>891 rows × 12 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     PassengerId  Survived  Pclass  \\\n",
       "0              1         0       3   \n",
       "1              2         1       1   \n",
       "2              3         1       3   \n",
       "3              4         1       1   \n",
       "4              5         0       3   \n",
       "..           ...       ...     ...   \n",
       "886          887         0       2   \n",
       "887          888         1       1   \n",
       "888          889         0       3   \n",
       "889          890         1       1   \n",
       "890          891         0       3   \n",
       "\n",
       "                                                  Name     Sex   Age  SibSp  \\\n",
       "0                              Braund, Mr. Owen Harris    male  22.0      1   \n",
       "1    Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   \n",
       "2                               Heikkinen, Miss. Laina  female  26.0      0   \n",
       "3         Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   \n",
       "4                             Allen, Mr. William Henry    male  35.0      0   \n",
       "..                                                 ...     ...   ...    ...   \n",
       "886                              Montvila, Rev. Juozas    male  27.0      0   \n",
       "887                       Graham, Miss. Margaret Edith  female  19.0      0   \n",
       "888           Johnston, Miss. Catherine Helen \"Carrie\"  female   NaN      1   \n",
       "889                              Behr, Mr. Karl Howell    male  26.0      0   \n",
       "890                                Dooley, Mr. Patrick    male  32.0      0   \n",
       "\n",
       "     Parch            Ticket     Fare Cabin Embarked  \n",
       "0        0         A/5 21171   7.2500   NaN        S  \n",
       "1        0          PC 17599  71.2833   C85        C  \n",
       "2        0  STON/O2. 3101282   7.9250   NaN        S  \n",
       "3        0            113803  53.1000  C123        S  \n",
       "4        0            373450   8.0500   NaN        S  \n",
       "..     ...               ...      ...   ...      ...  \n",
       "886      0            211536  13.0000   NaN        S  \n",
       "887      0            112053  30.0000   B42        S  \n",
       "888      2        W./C. 6607  23.4500   NaN        S  \n",
       "889      0            111369  30.0000  C148        C  \n",
       "890      0            370376   7.7500   NaN        Q  \n",
       "\n",
       "[891 rows x 12 columns]"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "titanic"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a0820446",
   "metadata": {},
   "source": [
    "### Data Dictionary\n",
    "\n",
    "\n",
    "survival:\tSurvival\t0 = No, 1 = Yes\n",
    "\n",
    "pclass:\tTicket class\t1 = 1st, 2 = 2nd, 3 = 3rd\n",
    "\n",
    "sex:\tSex\t\n",
    "\n",
    "Age:\tAge in years\t\n",
    "\n",
    "sibsp:\t# of siblings / spouses aboard the Titanic\t\n",
    "\n",
    "parch:\t# of parents / children aboard the Titanic\t\n",
    "\n",
    "ticket:\tTicket number\t\n",
    "\n",
    "fare:\tPassenger fare\t\n",
    "\n",
    "cabin:\tCabin number\t\n",
    "\n",
    "embarked:\tPort of Embarkation\tC = Cherbourg, Q = Queenstown, S = Southampton"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a5829749",
   "metadata": {},
   "source": [
    "### Dropping some columns"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4be8a31e",
   "metadata": {},
   "source": [
    "When we look at the table, I don't think, the Passanger ID, Name, Ticket no or cabin info will have an effect on survival. So we can drop them."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "80f2e1f0",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Embarked</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>male</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>female</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>C</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>female</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>female</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>male</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>886</th>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>male</td>\n",
       "      <td>27.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>13.0000</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>887</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>female</td>\n",
       "      <td>19.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>30.0000</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>888</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>female</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>23.4500</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>889</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>male</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>30.0000</td>\n",
       "      <td>C</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>890</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>male</td>\n",
       "      <td>32.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7.7500</td>\n",
       "      <td>Q</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>891 rows × 8 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     Survived  Pclass     Sex   Age  SibSp  Parch     Fare Embarked\n",
       "0           0       3    male  22.0      1      0   7.2500        S\n",
       "1           1       1  female  38.0      1      0  71.2833        C\n",
       "2           1       3  female  26.0      0      0   7.9250        S\n",
       "3           1       1  female  35.0      1      0  53.1000        S\n",
       "4           0       3    male  35.0      0      0   8.0500        S\n",
       "..        ...     ...     ...   ...    ...    ...      ...      ...\n",
       "886         0       2    male  27.0      0      0  13.0000        S\n",
       "887         1       1  female  19.0      0      0  30.0000        S\n",
       "888         0       3  female   NaN      1      2  23.4500        S\n",
       "889         1       1    male  26.0      0      0  30.0000        C\n",
       "890         0       3    male  32.0      0      0   7.7500        Q\n",
       "\n",
       "[891 rows x 8 columns]"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "titanic_new = titanic.drop(columns=['PassengerId','Name', 'Ticket', 'Cabin'])\n",
    "titanic_new"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7e47ef44",
   "metadata": {},
   "source": [
    "### Converting Verbal Categorical Values to Numeric Ones"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d8a807d3",
   "metadata": {},
   "source": [
    "Let's replace males with \"0\" and females with \"1\"."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "008a5c2b",
   "metadata": {},
   "outputs": [],
   "source": [
    "titanic_new['Sex'] = titanic_new['Sex'].map({'female': 1, 'male': 0})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "9976f0aa",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Embarked</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>C</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>886</th>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>27.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>13.0000</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>887</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>19.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>30.0000</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>888</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>23.4500</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>889</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>30.0000</td>\n",
       "      <td>C</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>890</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>32.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7.7500</td>\n",
       "      <td>Q</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>891 rows × 8 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     Survived  Pclass  Sex   Age  SibSp  Parch     Fare Embarked\n",
       "0           0       3    0  22.0      1      0   7.2500        S\n",
       "1           1       1    1  38.0      1      0  71.2833        C\n",
       "2           1       3    1  26.0      0      0   7.9250        S\n",
       "3           1       1    1  35.0      1      0  53.1000        S\n",
       "4           0       3    0  35.0      0      0   8.0500        S\n",
       "..        ...     ...  ...   ...    ...    ...      ...      ...\n",
       "886         0       2    0  27.0      0      0  13.0000        S\n",
       "887         1       1    1  19.0      0      0  30.0000        S\n",
       "888         0       3    1   NaN      1      2  23.4500        S\n",
       "889         1       1    0  26.0      0      0  30.0000        C\n",
       "890         0       3    0  32.0      0      0   7.7500        Q\n",
       "\n",
       "[891 rows x 8 columns]"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "titanic_new"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f67ad80e",
   "metadata": {},
   "source": [
    "Let's do the similar thing for \"Embarked\". New values would be C=0, Q=1, S=2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "37950821",
   "metadata": {},
   "outputs": [],
   "source": [
    "titanic_new['Embarked'] = titanic_new['Embarked'].map({'C':0, 'Q':1, 'S': 2})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "5724e0a4",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Embarked</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>2.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>2.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>2.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>2.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>886</th>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>27.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>13.0000</td>\n",
       "      <td>2.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>887</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>19.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>30.0000</td>\n",
       "      <td>2.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>888</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>23.4500</td>\n",
       "      <td>2.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>889</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>30.0000</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>890</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>32.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7.7500</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>891 rows × 8 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     Survived  Pclass  Sex   Age  SibSp  Parch     Fare  Embarked\n",
       "0           0       3    0  22.0      1      0   7.2500       2.0\n",
       "1           1       1    1  38.0      1      0  71.2833       0.0\n",
       "2           1       3    1  26.0      0      0   7.9250       2.0\n",
       "3           1       1    1  35.0      1      0  53.1000       2.0\n",
       "4           0       3    0  35.0      0      0   8.0500       2.0\n",
       "..        ...     ...  ...   ...    ...    ...      ...       ...\n",
       "886         0       2    0  27.0      0      0  13.0000       2.0\n",
       "887         1       1    1  19.0      0      0  30.0000       2.0\n",
       "888         0       3    1   NaN      1      2  23.4500       2.0\n",
       "889         1       1    0  26.0      0      0  30.0000       0.0\n",
       "890         0       3    0  32.0      0      0   7.7500       1.0\n",
       "\n",
       "[891 rows x 8 columns]"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "titanic_new"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c2fda1bb",
   "metadata": {},
   "source": [
    "### Rearranging Columns"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2c3b169c",
   "metadata": {},
   "source": [
    "Let's rearrange the column values so that Survived would be the last column."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "c3b9c507",
   "metadata": {},
   "outputs": [],
   "source": [
    "titanic_new = titanic_new[['Pclass',  'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked',  'Survived']]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "7524272d",
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Embarked</th>\n",
       "      <th>Survived</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>2.0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>2.0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>2.0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>2.0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>886</th>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>27.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>13.0000</td>\n",
       "      <td>2.0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>887</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>19.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>30.0000</td>\n",
       "      <td>2.0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>888</th>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>23.4500</td>\n",
       "      <td>2.0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>889</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>30.0000</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>890</th>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>32.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7.7500</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>891 rows × 8 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     Pclass  Sex   Age  SibSp  Parch     Fare  Embarked  Survived\n",
       "0         3    0  22.0      1      0   7.2500       2.0         0\n",
       "1         1    1  38.0      1      0  71.2833       0.0         1\n",
       "2         3    1  26.0      0      0   7.9250       2.0         1\n",
       "3         1    1  35.0      1      0  53.1000       2.0         1\n",
       "4         3    0  35.0      0      0   8.0500       2.0         0\n",
       "..      ...  ...   ...    ...    ...      ...       ...       ...\n",
       "886       2    0  27.0      0      0  13.0000       2.0         0\n",
       "887       1    1  19.0      0      0  30.0000       2.0         1\n",
       "888       3    1   NaN      1      2  23.4500       2.0         0\n",
       "889       1    0  26.0      0      0  30.0000       0.0         1\n",
       "890       3    0  32.0      0      0   7.7500       1.0         0\n",
       "\n",
       "[891 rows x 8 columns]"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "titanic_new"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6066cee1",
   "metadata": {},
   "source": [
    "### Handling Missing Values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "b546e028",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Embarked</th>\n",
       "      <th>Survived</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>714.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>889.000000</td>\n",
       "      <td>891.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>2.308642</td>\n",
       "      <td>0.352413</td>\n",
       "      <td>29.699118</td>\n",
       "      <td>0.523008</td>\n",
       "      <td>0.381594</td>\n",
       "      <td>32.204208</td>\n",
       "      <td>1.535433</td>\n",
       "      <td>0.383838</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>0.836071</td>\n",
       "      <td>0.477990</td>\n",
       "      <td>14.526497</td>\n",
       "      <td>1.102743</td>\n",
       "      <td>0.806057</td>\n",
       "      <td>49.693429</td>\n",
       "      <td>0.792088</td>\n",
       "      <td>0.486592</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.420000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>2.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>20.125000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>7.910400</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>3.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>28.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>14.454200</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>3.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>38.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>31.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>3.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>80.000000</td>\n",
       "      <td>8.000000</td>\n",
       "      <td>6.000000</td>\n",
       "      <td>512.329200</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "           Pclass         Sex         Age       SibSp       Parch        Fare  \\\n",
       "count  891.000000  891.000000  714.000000  891.000000  891.000000  891.000000   \n",
       "mean     2.308642    0.352413   29.699118    0.523008    0.381594   32.204208   \n",
       "std      0.836071    0.477990   14.526497    1.102743    0.806057   49.693429   \n",
       "min      1.000000    0.000000    0.420000    0.000000    0.000000    0.000000   \n",
       "25%      2.000000    0.000000   20.125000    0.000000    0.000000    7.910400   \n",
       "50%      3.000000    0.000000   28.000000    0.000000    0.000000   14.454200   \n",
       "75%      3.000000    1.000000   38.000000    1.000000    0.000000   31.000000   \n",
       "max      3.000000    1.000000   80.000000    8.000000    6.000000  512.329200   \n",
       "\n",
       "         Embarked    Survived  \n",
       "count  889.000000  891.000000  \n",
       "mean     1.535433    0.383838  \n",
       "std      0.792088    0.486592  \n",
       "min      0.000000    0.000000  \n",
       "25%      1.000000    0.000000  \n",
       "50%      2.000000    0.000000  \n",
       "75%      2.000000    1.000000  \n",
       "max      2.000000    1.000000  "
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# descriptions\n",
    "titanic_new.describe()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4b5e6cda",
   "metadata": {},
   "source": [
    "As you see, age count is significantly less than other values. There must be several NaN values. Similarly, Embarked count is 2 less than other columns, so there must be 2 NaN values.\n",
    "Let's replace NaN values. Let's start with Embarked column. Since it has categorical values, we can use \"Mode\" to replace NaN values."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "2d0b7d3d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    2.0\n",
       "dtype: float64"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "titanic_new.Embarked.mode()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "abbcabed",
   "metadata": {},
   "outputs": [],
   "source": [
    "titanic_new['Embarked'] = titanic_new['Embarked'].fillna(2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "bc8ee2dd",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Embarked</th>\n",
       "      <th>Survived</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>714.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>2.308642</td>\n",
       "      <td>0.352413</td>\n",
       "      <td>29.699118</td>\n",
       "      <td>0.523008</td>\n",
       "      <td>0.381594</td>\n",
       "      <td>32.204208</td>\n",
       "      <td>1.536476</td>\n",
       "      <td>0.383838</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>0.836071</td>\n",
       "      <td>0.477990</td>\n",
       "      <td>14.526497</td>\n",
       "      <td>1.102743</td>\n",
       "      <td>0.806057</td>\n",
       "      <td>49.693429</td>\n",
       "      <td>0.791503</td>\n",
       "      <td>0.486592</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.420000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>2.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>20.125000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>7.910400</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>3.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>28.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>14.454200</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>3.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>38.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>31.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>3.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>80.000000</td>\n",
       "      <td>8.000000</td>\n",
       "      <td>6.000000</td>\n",
       "      <td>512.329200</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "           Pclass         Sex         Age       SibSp       Parch        Fare  \\\n",
       "count  891.000000  891.000000  714.000000  891.000000  891.000000  891.000000   \n",
       "mean     2.308642    0.352413   29.699118    0.523008    0.381594   32.204208   \n",
       "std      0.836071    0.477990   14.526497    1.102743    0.806057   49.693429   \n",
       "min      1.000000    0.000000    0.420000    0.000000    0.000000    0.000000   \n",
       "25%      2.000000    0.000000   20.125000    0.000000    0.000000    7.910400   \n",
       "50%      3.000000    0.000000   28.000000    0.000000    0.000000   14.454200   \n",
       "75%      3.000000    1.000000   38.000000    1.000000    0.000000   31.000000   \n",
       "max      3.000000    1.000000   80.000000    8.000000    6.000000  512.329200   \n",
       "\n",
       "         Embarked    Survived  \n",
       "count  891.000000  891.000000  \n",
       "mean     1.536476    0.383838  \n",
       "std      0.791503    0.486592  \n",
       "min      0.000000    0.000000  \n",
       "25%      1.000000    0.000000  \n",
       "50%      2.000000    0.000000  \n",
       "75%      2.000000    1.000000  \n",
       "max      2.000000    1.000000  "
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "titanic_new.describe()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "78680f32",
   "metadata": {},
   "source": [
    "Now Embarked count is 891. Let's check the Age column.\n",
    "\n",
    "### Handling Missing Age Values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "c7d620be",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Embarked</th>\n",
       "      <th>Survived</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>8.4583</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>17</th>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>13.0000</td>\n",
       "      <td>2.0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>19</th>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7.2250</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>26</th>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7.2250</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>28</th>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7.8792</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>859</th>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7.2292</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>863</th>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>NaN</td>\n",
       "      <td>8</td>\n",
       "      <td>2</td>\n",
       "      <td>69.5500</td>\n",
       "      <td>2.0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>868</th>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>9.5000</td>\n",
       "      <td>2.0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>878</th>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7.8958</td>\n",
       "      <td>2.0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>888</th>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>23.4500</td>\n",
       "      <td>2.0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>177 rows × 8 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     Pclass  Sex  Age  SibSp  Parch     Fare  Embarked  Survived\n",
       "5         3    0  NaN      0      0   8.4583       1.0         0\n",
       "17        2    0  NaN      0      0  13.0000       2.0         1\n",
       "19        3    1  NaN      0      0   7.2250       0.0         1\n",
       "26        3    0  NaN      0      0   7.2250       0.0         0\n",
       "28        3    1  NaN      0      0   7.8792       1.0         1\n",
       "..      ...  ...  ...    ...    ...      ...       ...       ...\n",
       "859       3    0  NaN      0      0   7.2292       0.0         0\n",
       "863       3    1  NaN      8      2  69.5500       2.0         0\n",
       "868       3    0  NaN      0      0   9.5000       2.0         0\n",
       "878       3    0  NaN      0      0   7.8958       2.0         0\n",
       "888       3    1  NaN      1      2  23.4500       2.0         0\n",
       "\n",
       "[177 rows x 8 columns]"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#listing the Null Age values\n",
    "titanic_new[pd.isnull(titanic_new['Age'])]"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1801464b",
   "metadata": {},
   "source": [
    "Since age values are numerical, we can use Mean value to replace all NaN values. However, that might be the smartest way. Instead, we can try to guess ages based on the other given values like Sex, Pclass, Embarked info. Then, we can calculate the mean values of each subcategory and replace it with the corresponing null age values.\n",
    "\n",
    "Let's create a DataFrame to fill missing age values as described above."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "19a762ed",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Embarked</th>\n",
       "      <th>Age</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>40.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>44.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>42.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0.0</td>\n",
       "      <td>36.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1.0</td>\n",
       "      <td>33.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2.0</td>\n",
       "      <td>33.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>26.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>57.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>31.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>0.0</td>\n",
       "      <td>19.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10</th>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1.0</td>\n",
       "      <td>30.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11</th>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>2.0</td>\n",
       "      <td>30.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>12</th>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>25.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>13</th>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>28.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>14</th>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>27.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>15</th>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>0.0</td>\n",
       "      <td>14.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>16</th>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>1.0</td>\n",
       "      <td>23.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>17</th>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>2.0</td>\n",
       "      <td>23.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "    Pclass  Sex  Embarked   Age\n",
       "0        1    0       0.0  40.0\n",
       "1        1    0       1.0  44.0\n",
       "2        1    0       2.0  42.0\n",
       "3        1    1       0.0  36.0\n",
       "4        1    1       1.0  33.0\n",
       "5        1    1       2.0  33.0\n",
       "6        2    0       0.0  26.0\n",
       "7        2    0       1.0  57.0\n",
       "8        2    0       2.0  31.0\n",
       "9        2    1       0.0  19.0\n",
       "10       2    1       1.0  30.0\n",
       "11       2    1       2.0  30.0\n",
       "12       3    0       0.0  25.0\n",
       "13       3    0       1.0  28.0\n",
       "14       3    0       2.0  27.0\n",
       "15       3    1       0.0  14.0\n",
       "16       3    1       1.0  23.0\n",
       "17       3    1       2.0  23.0"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "fill_age = titanic_new[[\"Pclass\", \"Sex\", \"Embarked\", \"Age\"]].groupby([\"Pclass\", \"Sex\", \"Embarked\"], as_index=False).mean().round(0)\n",
    "fill_age"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "412cccff",
   "metadata": {},
   "source": [
    "We have 18 different subcategories to fill null age values."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "0e46c0a3",
   "metadata": {},
   "outputs": [],
   "source": [
    "for p in [1,2,3]:\n",
    "    for s in [0,1]:    \n",
    "        for e in [0,1,2]:\n",
    "            titanic_new.loc[(titanic_new['Pclass']==p) & (titanic_new['Sex']==s) & (titanic_new['Embarked']==e) & (pd.isnull(titanic_new['Age'])), ['Age']] = fill_age.loc[(fill_age['Pclass']==p) & (fill_age['Sex']==s) & (fill_age['Embarked']==e), ['Age']].values"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "726734ee",
   "metadata": {},
   "source": [
    "Let's see if it worked."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "cb49d4f5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Embarked</th>\n",
       "      <th>Survived</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "Empty DataFrame\n",
       "Columns: [Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, Survived]\n",
       "Index: []"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "titanic_new[pd.isnull(titanic_new['Age'])]"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1997a771",
   "metadata": {},
   "source": [
    "Now, we do not have any Null values, so it worked!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "d9eb7afb",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Embarked</th>\n",
       "      <th>Survived</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>2.308642</td>\n",
       "      <td>0.352413</td>\n",
       "      <td>29.344747</td>\n",
       "      <td>0.523008</td>\n",
       "      <td>0.381594</td>\n",
       "      <td>32.204208</td>\n",
       "      <td>1.536476</td>\n",
       "      <td>0.383838</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>0.836071</td>\n",
       "      <td>0.477990</td>\n",
       "      <td>13.310033</td>\n",
       "      <td>1.102743</td>\n",
       "      <td>0.806057</td>\n",
       "      <td>49.693429</td>\n",
       "      <td>0.791503</td>\n",
       "      <td>0.486592</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.420000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>2.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>22.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>7.910400</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>3.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>27.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>14.454200</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>3.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>36.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>31.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>3.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>80.000000</td>\n",
       "      <td>8.000000</td>\n",
       "      <td>6.000000</td>\n",
       "      <td>512.329200</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "           Pclass         Sex         Age       SibSp       Parch        Fare  \\\n",
       "count  891.000000  891.000000  891.000000  891.000000  891.000000  891.000000   \n",
       "mean     2.308642    0.352413   29.344747    0.523008    0.381594   32.204208   \n",
       "std      0.836071    0.477990   13.310033    1.102743    0.806057   49.693429   \n",
       "min      1.000000    0.000000    0.420000    0.000000    0.000000    0.000000   \n",
       "25%      2.000000    0.000000   22.000000    0.000000    0.000000    7.910400   \n",
       "50%      3.000000    0.000000   27.000000    0.000000    0.000000   14.454200   \n",
       "75%      3.000000    1.000000   36.000000    1.000000    0.000000   31.000000   \n",
       "max      3.000000    1.000000   80.000000    8.000000    6.000000  512.329200   \n",
       "\n",
       "         Embarked    Survived  \n",
       "count  891.000000  891.000000  \n",
       "mean     1.536476    0.383838  \n",
       "std      0.791503    0.486592  \n",
       "min      0.000000    0.000000  \n",
       "25%      1.000000    0.000000  \n",
       "50%      2.000000    0.000000  \n",
       "75%      2.000000    1.000000  \n",
       "max      2.000000    1.000000  "
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "titanic_new.describe()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "610a951d",
   "metadata": {},
   "source": [
    "Great! All of the features have 891 counts!"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a420a321",
   "metadata": {},
   "source": [
    "## B. Visualization"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2bab62a0",
   "metadata": {},
   "source": [
    "Let's see the correlation matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "6b103a99",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Embarked</th>\n",
       "      <th>Survived</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>Pclass</th>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.131900</td>\n",
       "      <td>-0.404714</td>\n",
       "      <td>0.083081</td>\n",
       "      <td>0.018443</td>\n",
       "      <td>-0.549500</td>\n",
       "      <td>0.162098</td>\n",
       "      <td>-0.338481</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Sex</th>\n",
       "      <td>-0.131900</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.115370</td>\n",
       "      <td>0.114631</td>\n",
       "      <td>0.245489</td>\n",
       "      <td>0.182333</td>\n",
       "      <td>-0.108262</td>\n",
       "      <td>0.543351</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Age</th>\n",
       "      <td>-0.404714</td>\n",
       "      <td>-0.115370</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.247817</td>\n",
       "      <td>-0.183515</td>\n",
       "      <td>0.117867</td>\n",
       "      <td>0.002351</td>\n",
       "      <td>-0.069609</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>SibSp</th>\n",
       "      <td>0.083081</td>\n",
       "      <td>0.114631</td>\n",
       "      <td>-0.247817</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.414838</td>\n",
       "      <td>0.159651</td>\n",
       "      <td>0.068230</td>\n",
       "      <td>-0.035322</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Parch</th>\n",
       "      <td>0.018443</td>\n",
       "      <td>0.245489</td>\n",
       "      <td>-0.183515</td>\n",
       "      <td>0.414838</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.216225</td>\n",
       "      <td>0.039798</td>\n",
       "      <td>0.081629</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Fare</th>\n",
       "      <td>-0.549500</td>\n",
       "      <td>0.182333</td>\n",
       "      <td>0.117867</td>\n",
       "      <td>0.159651</td>\n",
       "      <td>0.216225</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.224719</td>\n",
       "      <td>0.257307</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Embarked</th>\n",
       "      <td>0.162098</td>\n",
       "      <td>-0.108262</td>\n",
       "      <td>0.002351</td>\n",
       "      <td>0.068230</td>\n",
       "      <td>0.039798</td>\n",
       "      <td>-0.224719</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.167675</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Survived</th>\n",
       "      <td>-0.338481</td>\n",
       "      <td>0.543351</td>\n",
       "      <td>-0.069609</td>\n",
       "      <td>-0.035322</td>\n",
       "      <td>0.081629</td>\n",
       "      <td>0.257307</td>\n",
       "      <td>-0.167675</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "            Pclass       Sex       Age     SibSp     Parch      Fare  \\\n",
       "Pclass    1.000000 -0.131900 -0.404714  0.083081  0.018443 -0.549500   \n",
       "Sex      -0.131900  1.000000 -0.115370  0.114631  0.245489  0.182333   \n",
       "Age      -0.404714 -0.115370  1.000000 -0.247817 -0.183515  0.117867   \n",
       "SibSp     0.083081  0.114631 -0.247817  1.000000  0.414838  0.159651   \n",
       "Parch     0.018443  0.245489 -0.183515  0.414838  1.000000  0.216225   \n",
       "Fare     -0.549500  0.182333  0.117867  0.159651  0.216225  1.000000   \n",
       "Embarked  0.162098 -0.108262  0.002351  0.068230  0.039798 -0.224719   \n",
       "Survived -0.338481  0.543351 -0.069609 -0.035322  0.081629  0.257307   \n",
       "\n",
       "          Embarked  Survived  \n",
       "Pclass    0.162098 -0.338481  \n",
       "Sex      -0.108262  0.543351  \n",
       "Age       0.002351 -0.069609  \n",
       "SibSp     0.068230 -0.035322  \n",
       "Parch     0.039798  0.081629  \n",
       "Fare     -0.224719  0.257307  \n",
       "Embarked  1.000000 -0.167675  \n",
       "Survived -0.167675  1.000000  "
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#correlation matrix\n",
    "titanic_new.corr()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "ec0e6b3b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0.5, 1.0, 'Correlation Matrix')"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 864x576 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#seaborn package\n",
    "import seaborn as sns\n",
    "plt.rcParams[\"figure.figsize\"]=(12,8)# Custom figure size in inches\n",
    "sns.heatmap(titanic_new.corr(), annot=True)\n",
    "plt.title(\"Correlation Matrix\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "eee6bc44",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 1152x720 with 8 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# box and whisker plots\n",
    "titanic_new.plot(kind='box', subplots=True, layout=(2,4), sharex=False, sharey=False, figsize=[16,10])\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "e324be03",
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 1080x1080 with 9 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# histogram\n",
    "titanic_new.hist(alpha=0.8, rwidth=0.9, figsize=[15,15])\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "a01f2112",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAA3wAAANjCAYAAADxlBhsAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjQuMywgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/MnkTPAAAACXBIWXMAAAsTAAALEwEAmpwYAAEAAElEQVR4nOz9d5xb13ng/3/OveiY3siZ4ZDDTlEkVTiqVrXkKttxdyw7sRPHTnG8iZPNRtnkG8ebXzZO2dhxHBet07zuvclFlmw1q1KiWCSKfTicXoFBxy3n9wcwQ7AOBjNDDMDn/XrxRZR7Lw7mAgf3OeU5SmuNEEIIIYQQQojqY5S7AEIIIYQQQgghloYEfEIIIYQQQghRpSTgE0IIIYQQQogqJQGfEEIIIYQQQlQpCfiEEEIIIYQQokpJwCeEEEIIIYQQVeqiBnxKqW1KqceVUo8qpf5DKaUKnutQSv08//ydF7NcQgghhBBCCFGN1MVch08p5dVaW/nb/wF8Wmv9TP7+J4GvAnuBH2qtb7vQsVpaWnR3d/fSFlgsSG9vL93d3ewbiLK9s77cxRHAwYEom/PnYub8ADhurh4wDTV738i3x8w0y2gNrs49fqqpJsd23Nl97fy+jutiKnD0zP4aDeDYpDWEvV7SjoPfNMmmUyQ1+D0eQj4fCjAUaNTsaxoF5ciVSwEapRSue6oeUyr3XG47jdZg5Hc+te+psmvNWe9nuSj1OzQwGSWkoLGx+H2ytoPPYxa9veW4eM35tRlmLBu/11P09q6rZ8/dcqG1xtWnviu9vb3EvI0AC6rnxuIJWmvCCyrbVDJFYyhY8v6L8V1Y6DG0Pv07u1Az3yGtNbFYjBMxzco6k6DXi2kYoBQBjzlbNxiGwnH1aXWhucw+g9Vm5hyNTMdYUVdb7uIs2HB0mpX1deUuxoLsG4gCuTqt8Hdo5jFRfseGoqxrz52LZ599dlxr3Xqh7S9qwHfaCyv1WeCvtNbD+fsPAbdrrbVS6gfA3Vrr2Pn27+np0bt27bo4hRUl8bdvpP09n5i93/uxu8pXGEH3Pfeddr/lgY/w7999gLd99gkytnvW9gZQE/TgNw1AM5GwmImrwj4Tv8dgy8pa9g5EiWecpX8DZ/AY4LhwZg02c2lmKHB17vmAR9FW62cyaVEX9LKqMUTIa/LiUBS/x+AP7tzEW3d2AXBoJMZP9g/TGPbx9p5VuC587Zk+YmmbtS1hjozFaasNkLUdYmmbdW01HB6J0dkQ5M1Xrzrt4vCxw+M80zvJ5pW1vHZ7O8PRNN/e3Y/XMHhbzyoaQr4Lvseenh7G7/zo7P1ivkNnnue59ukbj3LLPz4GgKng6N/O/Ro3/u8HGJzOAPCpX72C11256oLb3/d8Hx/86r55vcbvfvFZfv7SKCGfyc8+fBMttaE591lq39h1gnu+tR8NvPOa1fzNm7cvSj1XeM7ee10Lf/Wm6+a1/19/Zzf/9tTggsrww70D/MNPDxHwGPzr3VezYcX8L7w/+KXneP7kFDvXNPLJd1497/2fODLGb33hWRxX88Hb1/OhOzbN+xj7+qP8/KVRVtb7ecvVq7j+umu5455/42u7+os+hiJX/2mVqz9etr6ZL/7W9fMuiyjOinWbCb79n2bvV+q1wnzr3uXqXNcKhb9DULnvrVoUnqPej92FUupZrXXPhfa56HP4lFJvUErtB9qAiYKnTH0q+owCjefY9wNKqV1KqV1jY2MXobRCVLdHDo2fM9gDcIFE2iaRsYlnbAo60UhbDinL4ehYoizBHoB9jmAPco9pcj2LM8+nbc1EIkvWdokkspycTHB8IkEy65CxXR48MDq7/0vDMRxXMx7LMDqdYTCaYippYbuaJ45NoDW8NDzNcDSN7WoePzKO1tA/lSKWtk4ry4tDuRbRg8MxbMfl6FicjOUSz9icmEjO+R5nWlSX0qcePDZ72ymy/W8oH+wBfOKBw3Nu/7c/OTTv13jm+CRaaxIZm/tfHJ17h4vgS0+dxNG5hoSfHRhZktf4z6fG571PYbBXqp+9OIrt5D6bDx0q7fd130AEgD0nIyXt/9VnTmI5Lq7W3LdvqKRjHBiextWawUiaSCr3fXzi6MQce51OAw75BiMNu09ESiqLKE4kde7fICHE3M4M0M/nogd8Wuvva623AQPA6wqeKrxqrAMi59j3Xq11j9a6p7X1gj2XQogivPryFdT4zz2Mz+uBlho/LTV+WmsC+PKbGQrqgl4aQl6u7GqgrfbCvVRLJeg18J5jpJUJ+EyFz8j1JimgPmCypilEbcBLR0OQyzvquaKznqaQj7qAl7fu7Jzd/4pV9dT4PaxpDtFeH2BVY5DOxiC1AQ+vvHwlIZ/JNWsaWd9Wk39sBSGfyZaVtdQHvaeVZeeaJkI+k6tWN+Axcz2ijSEvbXV+1rfVzPkeL8bQmT9/9brZ235PcUPXtqw8NfTwY2+8bM7t/+mtp3p7AkW+xqu2rcRjGrTW+nnj9s65d7gIPnznRrymwlRw9zVdS/IaX3tTy7z3+ftf7V7w6775qk5q/V7aagO8bkd7Sce4eWMLAa/JLZtK+33+wC3rCfs9+DwG775+TUnHuKqrgbDfZOOKGpryPeiv3V78+zHI1RlBD/jMXH13x2VtJZVFFKe7rjryB5Y+mFqI0hXb23qx5/D5tdaZ/O2/AR7VWv8kf/+TwFfIzeG7b645fDKkc/nr6elBztHyJedn+ZNztPzJOVre5Pwsf3KOljc5P8tfMUM6i589vzherZT6o/ztw8D9Sql/0Vp/CPh74AvkGkk+cpHLJeZQbJexjOsWQgghhBBi+bioAZ/W+nvA9854+EP55/qBl1/M8gghhBBCCCFENauOgdNCCCGEEEIIIc4iAZ8QQgghhBBCVCkJ+IQQQgghhBCiSknAJ4QQQgghhBBVSgI+IYQQQgghhKhSEvAJIYQQQgghRJWSgE8IIYQQQgghqpQEfEIIIYQQQghRpSTgE0IIIYQQQogqJQGfEEIIIYQQQlQpCfiEEEIIIYQQokpJwCeEEEIIIYQQVUoCPiGEEEIIIYSoUhLwCSGEEEIIIUSVkoBPCCGEEEIIIaqUBHxCCCGEEEIIUaUuasCnlLpOKfW4UupRpdTHz3jur5RSe5RSDyml/uhilksIIYQQQgghqpHnIr/eCeDlWuu0UupLSqntWut9Bc//sdb6gYtcJiGEEEIIIYSoShe1h09rPay1Tufv2oBzxiZ/p5R6QCl15cUslxBCCCGEEEJUo7LM4VNK7QBatNYvFjz8Sa31TuB3gX8pR7mEEEIIIYQQoppc9IBPKdUEfAp4X+HjWuvJ/P+HL7DvB5RSu5RSu8bGxpa2oEIIIYQQQghR4S520hYP8EXgT7TWw2c8V5f/v4XzzC3UWt+rte7RWve0trYueXmFEEIIIYQQopJd7KQtbwOuITdXD+DPgLu11h8C/kEptY1cEHrPRS6XEEIIIYQQQlSdixrwaa2/AnzljIefyD/32xezLEIIIYQQQghR7WThdSGEEEIIIYSoUhLwCSGEEEIIIUSVkoBPCCGEEEIIIaqUBHxCCCGEEEIIUaUk4BNCCCGEEEKIKiUBnxBCCCGEEEJUKQn4hBBCCCGEEKJKlRzwKaXWK6X8+du3KaX+m1KqYdFKJoQQQgghhBBiQRbSw/ctwFFKbQD+DVgLfHlRSiWEEEIIIYQQYsEWEvC5WmsbeBPwCa31h4H2xSmWEEIIIYQQQoiFWkjAZyml3gm8B/hh/jHvwoskhBBCCCGEEGIxLCTg+w3gBuBvtNbHlVJrgS8uTrGEEEIIIYQQQiyUp9QdtdYvAv8NQCnVCNRqrT+2WAUTQgghhBBCCLEwC8nS+ZBSqk4p1QTsAf5DKfVPi1c0IYQQQgghhBALsZAhnfVa62ngzcB/aK13AncuTrGEEEIIIYQQQizUQgI+j1KqHXg7p5K2CCGEEEIIIYRYJhYS8P0v4KfAEa31M0qpdcDhxSmWEEIIIYQQQoiFKjng01p/Q2u9Q2v9e/n7x7TWb7nQPkqp65RSjyulHlVKffyM5zqUUj/PPy9DQ4UQQgghhBBigUrO0qmUCgDvAy4HAjOPa61/8wK7nQBerrVOK6W+pJTarrXel3/uHuAvgL3khog+UGrZxPLRfc99s7d7P3ZXGUsiXv7X93EskbtdeC7uffgIDxwY5fKOWuIZB61hbWuY9S1hPB4D14XxeAbLcXn9jk6aanyz+0aTFi8MRUlmHLpbwkSSWR47Mk4ik+Xhg+NMJbOsaa7ht29eyzef7aexxsuzfRH6J5NkLU1N0KQl7MdxNVOJLBnbIeiBpK3IOhqPAe1hk0DQQ/+URdBr0F4fZCSW5hVbV3Db5jaePxnhxESSsN9ka3sdP31hhIaQl5dvWcFgJA24rKwP0hjyEU1ZxNM2N21o4bOPHEUpxfXrmoilLG7a0Dr72DuvXU17vZ+fvTjKts566oMeeseThPwml62s48RkEr/H4LL2OgBcV7N3IIrPNNjaceqxfQNRvAWPlWK+36G/+s5T/OdT40VvD/DHX93NY8fG+bPXbOGNV3XNuX3Wdtk3EKEh5GN9a01Rr1HNZs7RZcCPS6jnXv+Jh9g3nPtyllpP7vzr+5lMWPzpqzbwO7dvnvf+WdvlO7v7aQj6eNW2lSWVYbl66tg4/+uHLxJLJOmLOmc97zfBZxr87m0b+b2XbwDg2RNT7Omb4JHDEwR9Hj75jh14vbLU8FK55q/vZyxh8VsvW81fvH57uYtTsi88fpyh6QwfvHUdNUHf3DssQ5ffcx/5S4XZ+ui/3Xsf3z92+mOifEq5ti454AP+H/AS8CpywzvfBRy40A5a6+GCuzZQWPPuAP5Aa62VUjGlVK3WOraA8oky2zcQpb3chRCzZoI9yFUWLcCe/ik+/sBhMrbLM71TGIBhKoJek+YaPw3B3AXO8HSaxpCXY+NJPvqGy2eP8/29g+zqnWQslmFrey3Pn4xybDxOJGHh5rcZi0/x/MkIhoKso08rUzTlEE0lT3ss7QDktrNc6Is5EMtVFUnLZSKZqxa+/HQ/jx6eYDKRJWU5mAq+zQCum9v7scMTGCp3zJDfQ1utj9FYlvqgl3sfPcp0yiZrO/zsxWHa64P830ePEU3ZWI7LYCSJ32OSzDp8f88AV3Y1cGAoRlutnxV1AQJeEwCfx2B9aw27T07xyKHx2cc2tNWw+2SERw6N5R9TbGirnf85G4oy31BxJtiD3Hme68fgscMjfPv5QTTwJ9/YW1TA98sj4zx/MoJScPe1q2mrC8y5T7UqrOcu+AN4oWMMn/pyvv4TD/GDP7xtXvv/3hefZiJhAfCxnx4pKeD7f0/0cv+LIwDUBjzcuKFl3sdYrn77/z3HdNrC1ed+PuNAxnH5x/sPsnNtI+tbw/zT/Qd5cXCaSMrCayr++zf38c/vvPriFvwSMRhJ4ct/fj//y76KDfh+uHeAzz2Si4oiiSx/+5YdZS5RaQouFWavFWaCvZnHJOhbPu7++H1zb8TC5vBt0Fr/f0BCa/1fwF1AUd9SpdQOoCW/lt8MU2s9Ux1HgcZz7PcBpdQupdSusbGxBRRdCAFgos75uJr5p/L/AIU6a2tDnXu7s4537pdZsMLDKpV/ZXXGaxaWcbbcp/acKa8qfEydum8ohcHp7/Ps1z9939Ne/6ySLi8+0zx1p9hiqvPcFgs205gwHz5j/vucqfDzbxrVdVLn825MwMDIfX8LdjSWqhITVcNUpy6pjYVcXQuxBBbSw2fl/48opbYBw0D3XDvl1+37FLnsnoUKe/vqgMiZ+2qt7wXuBejp6TlPW51YLrZ31jPT1yB1X/ld2QbPj+Zu937sLnp6PsK2VQ3c89rN3P/CKNs66khkHVxXs64lTHdrCI9h4mqYTGTIWC537Ti9z/b1V3SwsbWGpOWwpjnM63Zk+eXRcdJpmwcPjjGZyNLdUsOH7ljLl57qpynoY3ffFCcmE6QzLvVhD621AWxXMxXPkLYcgl5NwlKkLY3HhFU1HmpCfnon0wS8Jp1NQYYiKe7a1s7LNraytz9K70SCcMDgivYGfrh/iMaQn9s3tzIUTQGKFfX+/JBOm1ja4paNLXz24aOYpkHP6iZiGZtbNjRz76PHQWne1rOajvoAPzswwraOOmoDXvomk4S8JptX1tI3mcLvNViXH854VVcDfo8x2+MHcOWqBnymgdfM9fiVYl37qe9Qse55XQsf+2HxQzqvXdfCr/as4pHD43zk9ZcV9Ro3bWihIeilMeSjrfbS7d2D0+u54v56Z7umq45nTk6jgG988OZ57/+Ju3fy5PGfMRrL8pevm3/vHsC7r19DfdBDQ8jHdeuaSzrGcvVfv9nDX37vRSwnywtDqbOeD3sVHsPgD+7cSM/a3Hv/k1dtZu/JKR49OkHY6+Hjv3rVxS72JaOjIYi31sdwLMsf3bmu3MUp2Wu2txNJZRmKpPnd2zaUuzgluwl4LH975lrhDeuQIZ3L1Jc/fBdf+aO5t1OnOtXmRyn1W8C3yA3F/A+gBvhLrfVnL7CPB/g+8FGt9VNnPPdJ4Cvk5vDdp7W+7UKv39PTo3ft2lVS2cX8FY4XvpDCiqCnpwc5R8uXnJ/lT87R8ifnaHmT87P8yTla3uT8LH9KqWe11j0X2qbkHj6t9efzNx8Gim2SeRtwDfB3+eEjfwbcrbX+EPD3wBeAIPCRUsslhBBCCCGEECJn3gGfUuqCHYda63+6wHNfIdeLV+iJ/HP9wMvnWx4hhBBCCCGEEOdWSg/f/FPNCSGEEEIIIYS46OYd8GmtP7oUBRFCCCGEEEIIsbhKTp6olPovpVRDwf1GpdS/L0qphBBCCCGEEEIs2EKWZdihtY7M3NFaTymlypK3uJQMkkIIIYQQQghR7RayPJqhlJpdHD2/vt5CAkghhBBCCCGEEItoIQHa/wGeUEp9A9DkFlL/m0UplRBCCCGEEEKIBVvIOnxfUErtIreUggLerLV+cdFKJoQQQgghhBBiQUpZhy8A/A6wAdgHfFZrbS92wYQQQgghhBBCLEwpc/j+C+ghF+y9BvjHRS2REEIIIYQQQohFUcqQzq1a6+0ASql/A55e3CIJIYQQQgghhFgMpQR81swNrbWtlFrE4ly6ZGkJIYQQQgghxGIrJeC7Qik1nb+tgGD+vgK01rpu0UonhBBCCCGEEKJk8w74tNbmUhRECCGEEEIIIcTiWsjC60IIIYQQQgghljEJ+IQQQgghhBCiSknAJ4QQQgghhBBV6qIGfEqpDqXUc0qptFLKc8Zzf6WU2qOUekgp9UcXs1xCCCGEEEIIUY1KydK5EJPAHcB3zvP8H2utH7iI5RFCCCGEEEKIqnVRe/i01mmt9dQFNvk7pdQDSqkrL1aZhBBCCCGEEKJaLac5fJ/UWu8Efhf4l3NtoJT6gFJql1Jq19jY2MUtnRBCCCGEEEJUmGUT8GmtJ/P/H77ANvdqrXu01j2tra0Xr3BCCCGEEEIIUYGWTcCnlKrL/9/CxZ9bKIQQQgghhBBV52Jn6fQqpR4ArgB+qpS6Tik1M3zzH5RSvwR+ANxzMcslhBBCCCGEENXoovakaa0t4M4zHn4q/9xvX8yyCCGEEEIIIUS1WzZDOoUQQgghhBBCLC4J+IQQQgghhBCiSknAJ4QQQgghhBBVSgI+IYQQQgghhKhSEvAJIYQQQgghRJWSgE8IIYQQQgghqpQEfEIIIYQQQghRpSTgE0IIIYQQQogqJQGfEEIIIYQQQlQpCfiEEEIIIYQQokpJwCeEEEIIIYQQVUoCPiGEEEIIIYSoUhLwCSGEEEIIIUSVkoBPCCGEEEIIIaqUBHxCCCGEEEIIUaUk4BNCCCGEEEKIKnVRAz6lVIdS6jmlVFop5TnHcz9XSj2ulLrzYpZLCCGEEEIIIarRxe7hmwTuAJ48x3P3AH8BvDL/vxBCCCGEEEKIBfDMvcni0VqngbRS6lxP7wD+QGutlVIxpVSt1jp2McsnFte+gSjd99w3e7/3Y3eVsTSi8FwAtOT//8VLozx+dJxk1mH7qnp8hsFwLM3Lt7SxZWXdWceJpS2+u3uAjO3yK1d2kso6fPf5fg6NxHhhcJqxWAbb0ViOi6sh6DW4anUD+weniaVtXH3qWKYBdX4PacslY7soBYYCy80/rwANDrnWKXce77cuYJCxNFkn94KGAkefvo0CPKbCVNBc4yOacjCUxu8xydgu3c1Bjo4l8RiKrR21HB5NoLXGcSHsM2mrCxBNWxgKbtrQyl+89jI8nlw72lgsw9d3neSBF0eo8Zt88PaNXL+++YJlfujgKC8MTrNzTSPXr8ttO9/v0Jnnea59LMvi5f/0KGPxDLdtauOzv7Zzztf4znP9fOoXR2gM+fi/v9ZDY41vzn2q1WLUc4X7f+G3tnHLhjXz2v+xF8d49xeeBnLfk2NlqmsfOzzOnv4IV3Y18LINLXPvcJF8+Ku7uW//EH7T5K9et5U396xi3Z/9CD33rufUGvaxY3UdDx4YB3J1S1PYx45VdTx8aBxDKbauCLN/KI6rYU1TiDUtYTrqA3zk9VsJ+M6+9Lr34aN89uGjBH0m2zvqeOzoBDV+D597904yjstnHjpKY8iHAr6/dxCfafD937+R9W1n19HLmeM4vO8Lz9I7keBtO1fxwds3MjBZHdcKkWSWv/r+i0yns3zw9o3sXNNY7iKV5FzXCvP9XRFLq5Tvy3Kaw2dqrWfq3yhQmd8UISqI42qePxmhfyrFiYkkR0fj7BuIkrFc9p6MnnOf4+MJxuNZYmmbg8MxXhiMMhhJc3w8wch0hmTWIW27OBo0kLZddvVOET8j2Mu9PkRSNmnbxSUXkFkFUZ2TD/ZgfsEewHTaJeNoNLlynBnskX/ccjQZWzMUzZC2HOJph/F4lqzt8sJgDMtxSWRsdvdFSGRsppIW8YzNVDLL4dEYY9NpxmMZDgxNMxBNzR774HCMo6NxRmNpJpMWD7w0csHyuvlzkbVdnj8ZAWD/wLnPwWJ68vgUI9NpXFfzxNHxovb54d4h0pbDUDTFQ4dGl7iEl5Y/+NIL897nj7+ze/b2fL8ni2l33xRZ22V331QZS3G2hw+P4TiaRNbmgZdGmE7ZJQd7AGOJLI8fmZy972pIZGwePTyB4+bqlBeG47N14ImpJKmsxbHxBAdHzt2O/b09A1iOy0Q8y5O9k2Rtl+m0zVef6ePBA6Mksw4DkRT3HxjGcTUpy+E/ftm7gHdRHgeH4xweiWHZLj97IVd3TKbm2KlC7OqdZCiaIpFx+MVLUi+Ki+PMYPx8llPA5xTcrgMiZ26glPqAUmqXUmrX2NjYRSuYENXKNBRbO+poq/PTUR9gdXOIzStrZx8/lzXNYeqCXoI+k40ratjSXkdbrZ/OhiDNYR8Bj4HfVChO9aBtW1VP0GdyZt++oaDGb+Lz5LY3yPfqFTw/c/ec4wIuIOgFrzn3/h5D4TUVLWEffo9ByGfQEPLiMQ02rKjBNBQBn8nW9joCXpO6gIeA16Am4GFNU5jGsI+GkI/1rWE664Ozx924oobVTSEaQj5q/SY3z9HjYRiKyzvqMZRiW0c9ANs66+f5rudv56pGmsI+lFJcXWSL9Cu3rsBjKJrDfl624cK9lmJ+/vldl897n4++Yf77LIVtnfnP70X43M7Htd3NGIYi6DW5eVMbtYGFDW6qD5rsXNMwe18BAa/Jtd2NGCpXh61rCc3Wge21frymSWdDkI0ras95zNdsa8c0DOqDXq7qbMBjGIR8Jm/auYqbN7bg8xi01vq5ZUMrhlL4PQbvum5+PcHLwYbWGtY0hzGU4pZNuTqxxlvmQi2SK1c30Fzjw+cxuHnj8unhFtWt2B4+dapTbXEppTqAHwJbgRqttV3w3ONABvADf6m1fkAp9RlgO2ACDVrryy50/J6eHr1r1y6g+Oh2OXdBL/f3UEr5enp62LVrF//7f9/H//yfy/dvfyn5whfu49d/PXcuZs4PwEw9MDPcWmvNeYZen1Ph9o7jzD7m8Zy6sHIcB9d18Xpzv+6ZTAaPx4NpmrPb2LaNUuq0x1zXJZvN4vf7Z19jenqacDh82naJRIJQKHRauWde0+PxoJTCsqzZ52bK4TgOWmtM0zxtX9u2Tyv/zLZKKQzDmH2PM3+7mcfOxXXdCz5fqPBvOXOO/vSe+/i7Jf7uW5Y1+zcphuM4p/39L1WF36OFOHHiBGvWLOwCfmxsjNbW1gWXZSHmW3cstZnzY1nWbD0wY3BwEJ/PR01NDaZp4vV6iUaj+Hw+vF4vhmHM1iG2bRMIBHAcB58vN4TZdV0sy8Lv988es/C7Xvh8MXXATJ1zrr9h4f7ZbHa2DJWqsP6YOUff+MZ9vO1tlX+tMJ/6vhIsVh0nFtc999zHx/LXBUqpZ7XWPRfafikDvgAQBL4D3Km1tpVSXuDHwE3AfuC/A1/UWq9SSn0UeCswDYSBHq119nzHb2lp0d3d3ewbiNIRgObm5dWiOB9DkSjjCdi+zFpF52tffvjZzPvo7e2lu7u7jCUSFzJzfrSGjJ0LegxDoVA4Wud65wyFUgqlcsOWbMfFZxqgQJF73HY08YyFaSi8poHj5oZRZiwn33tm4POaJLM2tu0ync5i2S61QR/1QR/kB1e5Gtx8fZSxHDymgdc08JgGiYxNwGuA1liuxnI0TWEfWdvFzY8T1eR6LI18eVX+mJbjEvSaubK6GsfVeEwD23ExDIVHKVzyc/zc3IWWUXCt5bgaQylMQ+Hq3G2t88Fe/jWX8hw53kaSzK9+2DcQRVF8D6Flu0RSFq21/rk3zoskswS8JgFvcUGf7WqytkPoHPOXzsfVsIR/3pI4bi7I95i5C7re3l5i3lzP6ELq8P0DUdrqTNpqa0rafySaZDJpcVl76WXI2vnvxAL+6Laj8ZgL2N/VuK7G5yn9grnwc1P4O3RyIk4knWuQCppga/CZiqawH4/HxO8xSGZsXCDs8+C4bq7OcDSOdvF5TDyGwu81MZTCclwcV+M1jfz8Y0XGdjFUru608vVN0GtiuxozXzedj+W4mEphGArbzdXBM/WL1uTrXcjYLh5DLWnds9Rm6lLInSN/Q2NVXAcNTiVJWQ7r287dk1spCq/nZr5D+weiGMDWCj9H1aLwHD377LNaa33BSnPJAr7ZF1DqIfIB3xmP3Z5P0PID4G7g34EPaq1HlVL/Atyrtd53vuP29PTo8Ts/Ont/OffeXcgPf3gfv//YqfuV+j7ONaFXWoWWt56eHh545HHe/OlfcmIiOTvHrTA5SsBj0NkYBDTD0TRZRxP2mVy9ppFtHfVEUxZffqoP68zJeWVkKHJDSpUiZblocoljerrqeKZvGttxUerUBdWqpmA+UISJuIVCs3llHbUBDxOJLMfHE9QHPLzhyk6UUnQ0Bjg2muDBAyM0hHy8/+Z1vGrbyiV5Lz09Pcy3npvv5PqDgzFe9clHgNxQtKN/O/drXPc3P2MklmuP+5s3buVd16+94PaHh2O87lOPYTkuV69u5Ju/e+Ocr/HsiSkeOTRGS42Pd1yzekEBwGJ5vm+KD399D47r8uE7N/Gmq1fhb99I+3s+MbvNQpO23L4pyH/85svntf87P/1znug7NRGqlDJ86sHD3PvoMTym4jN37+S6ORIMnctf//BF9g9E2bGqnj+/a+u893/8yBi//+Xd2K7md25dz+/dvmHex3j2xCSPHBqnpcbHr167mhuuu5Zdu3ax7Z77iM/7aOfWUR/gFZe38Y1d/aStXIPS9lX1tNb4uP/FUQwFnXUBjk+lcnVMY5CV9QEaQz4+9ubt1IfO7p379EOH+fJTJwn7TN517Ro+/8tjGIbiH9+yg9qgj5/sHybsNzk6GuOHe4cJ+gy+9oEbWNtaWgNBuTiu5mvPnGRkOs0N65u5fl0zjas3Un/3J2a3qdTroKs/et9p8xEr9X2clbTlgY+c9jsElfveqsWZSVuK6eEr1y/ouRK0NJDr3St87DQXmsN3y18UN+RwuSkM9oS42I6OJZhIZE9LaFKY9CHruEynLCJJi7SVa82OZ2yiySzTaYu9/dFlFexBrnU/62jS+WAPIGW5nIxkyDpuPjFMrnfOJZdNM5GxGZnOPZ91XAYiKYan0wxMpbAdl7TtsOtELknD830RRmNpUpZD2nJ4bgkTVOy7CElbvvZM7+ztcyW2OZex2KnBF196sm/O7X/y4hCWk/tkHRyZnmPrnGNjucvz8XiWaMqaY+uL45dHx8naDo6r+eXRiSV5jV8cmn8Gi8Jgr1SPHR1Ha41luzw4R4Kh83lpKHduDwwVd47P9LMXR7AcF601jx0pLoHQmY6OJYCzPzeLFewBTCWzPH0sl6BGa3LJnjI2Tx2fzP0NHZfBfCIk19UMRlK4rmYykaV3InHOYz59fAqtc/XrT14YwnFz5+KxI+McH0/gak0sbfPU8SlAk8o6PFZkkqXlJJ62GZlOA3Asf66Sy+PrvWDVknxGVJblnrTlXAlaIvnbhY+dRmt9r9a6R2vdc+Y8hUf+f5XZ2lAtrSSVO7Dk0nZFZz1XdTUS9pu5YUgKwj4DjwEBU9FW6+ey9jp2rKqnvT5AbcDDhrYaLu9sYNOKOt7e00VrQUr+830OAksw1cvnAf8ZNZhp5MrfVhugoyFAyGviMaCrMcAtm9torfFT4zdprfHh9yjqgx6u7W5iXWsN169tYmWdn47GEDdvaOaa7iZu2thCa22Ada21/Nr1a2gK+3jjlZ1cvbqR9fnkA79yZcfiv7m8izG86S9/ZfvsD0FLuLg5fDflE7UYwL/efcFGRQB+47puWmv8+EyDN1/VWdRrXLe2mZYaH9s762lZJss+vG3nKtY0h2mrDfCeG5YmYUYpvwnfecfCP4Pvv2kt9UEf7Q1BfuNlF+6xPZ/XXdFBc42P119RWnk+cMt6VtYHaQj5+MDN60o6xnVrm2Y/N83hU5+bd10x/8Tf56rPDOCKrgZ+82VraKsNEPLlRkFsXlHL7922gbDfQ3PYz22bW6jxmwT9JndsaWVFfYAruxq4/DxDbn/t+tU0h/1sWVnHH79iMyvqAnQ1hnhbz2quXt1AW52fDW01vO/mbmqDXta21PCmJax7lkp9yDub3OS6dU0ArK6sTsrzesv2ttnbMrtZXCxlT9oy+wLnHtL5SeArwF7gPq31bUqpPwKGgK8DvwBeobXOnO+4hUlbxPIkQzqXNzk/y5+co+VPztHyJudn+ZNztLzJ+Vn+ihnSuWQLrxckaLkC+KlS6n8C79Zafwj4e+AL5JK6fCS/y+eBLwMfIjd/77zBnqgsyz0DqRBCCCGEENVqyQI+rbUF3HnGw0/ln+sHXn7G9tPA65aqPEIIIYQQQghxqSl/2jMhhBBCCCGEEEtCAj4hhBBCCCGEqFIS8AkhhBBCCCFElZKATwghhBBCCCGqlAR8QgghhBBCCFGlJOATQgghhBBCiColAZ8QQgghhBBCVCkJ+IQQQgghhBCiSknAJ4QQQgghhBBVSgI+IYQQQgghhKhSEvAJIYQQQgghRJWSgE8IIYQQQgghqpQEfEIIIYQQQghRpSTgE0IIIYQQQogqJQGfEEIIIYQQQlQpCfiEEEIIIYQQokotacCnlPq4UupRpdQ/n/H4J5RSD+X/TeUfe69S6mD+sb9fynIJIYQQQgghxKXAs1QHVkpdDYS11jcrpT6jlLpGa/0MgNb6D/PbXAX8ccFu/6C1/vxSlUkIIYQQQgghLiVL2cN3A/BA/vYDwPXn2OZNwLcL7v+hUuoRpdQdS1guIYQQQgghhLgkLGXA1wBM529HgcZzbPNq4Cf5298FdgBvAf5RKWUuYdmEEEIIIYQQouotZcAXAeryt+vy92cppTYCA1rrJIDWOqK1drXWY8AhYMWZB1RKfUAptUsptWtsbGwJiy6EEEIIIYQQlW8pA74ngJmhmXcCT57x/JuA78zcUUrV5f8PAhuBsyI6rfW9WuserXVPa2vrkhRaCCGEEEIIIarFkgV8WuvngLRS6lHA1Vo/rZT6l4JNXgf8oOD+h5VSTwAPAR/TWltLVTYhhBBCCCGEuBQsWZZOAK31H5xx/0MFt28547mPAh9dyvIIIYQQQgghxKVEFl4XQgghhBBCiColAZ8QQgghhBBCVCkJ+IQQQgghhBCiSknAJ4QQQgghhBBVSgI+IYQQQgghhKhSEvAJIYQQQgghRJWSgE8IIYQQQgghqpQEfEIIIYQQQghRpSTgE0IIIYQQQogqJQGfEEIIIYQQQlQpCfiEEEIIIYQQokpJwCeEEEIIIYQQVUoCPiGEEEIIIYSoUkUFfEqp951x31RKfWRpiiSEEEIIIYQQYjEU28N3h1LqR0qpdqXUNuBJoHYJyyWEEEIIIYQQYoE8xWyktb5bKfUOYB+QBN6ptf7lkpZMCCGEEEIIIcSCFDukcyPwB8C3gF7g15RSoSUslxBCCCGEEEKIBSp2SOcPgL/UWv82cCtwGHhmrp2UUh9XSj2qlPrnMx7/K6XUHqXUQ0qpP8o/VquU+oFS6pdKqV+f5/sQQgghhBBCCHGGooZ0AtdqracBtNYa+D9Kqe9faAel1NVAWGt9s1LqM0qpa7TWhUHiH2utHyi4/37gK8DXgF8opb6qtc4W/1aEuLR133NfUdv1fuyuJS6JEEIIIYRYLort4Qsqpf5NKfUTAKXUVuCWOfa5AZgJ6B4Arj/j+b9TSj2glLqycHuttQPsATYXWTYhhBBCCCGEEOdQbMD3n8BPgfb8/UPAH86xTwMwnb8dBRoLnvuk1non8LvAvxSxPQBKqQ8opXYppXaNjY0VWXQhhBBCCCGEuDQVG/C1aK2/DrgAWmsbcObYJwLU5W/X5e+T338y///hYrYv2O9erXWP1rqntbW1yKILIYQQQgghxKWp2IAvoZRqBjSAUup6cr1wF/IEcEf+9p3k1u4jv39d/v8WTs0jfILcen8mcCVwsMiyCSGEEEIIIYQ4h2IDvj8Cvg+sV0r9EvgC8KEL7aC1fg5IK6UeBVyt9dNKqZnhm/+QP84PgHvyj30eeBfwKPDvWuvM/N6KEEIIIYQQQohCF8zSqZS6BjiptX5OKXUr8NvAW4D7gf65Dq61/oMz7n8o//9vn2PbaeB1xRddCCGEEEIIIcSFzNXD9zlgZmmEG4E/B/4VmALuXcJyCSGEEEIIIYRYoLnW4TNnEqwA7wDu1Vp/C/iWUur5JS2ZEEIIIYQQQogFmauHz1RKzQSFdwA/L3iu2EXbhRBCCCGEEEKUwVxB21eAh5VS40CKXEIVlFIbmDtLpxBCCCGEEEKIMrpgwKe1/hul1IPkFly/X2ut808ZzJGlUwghhBBCCCFEec05LFNr/eQ5Hju0NMURQgghhBBCCLFYil2HTwghhBBCCCFEhZGATwghhBBCCCGqlAR8QgghhBBCCFGlJOATQgghhBBCiColAZ8QQgghhBBCVCkJ+IQQQgghhBCiSknAJ4QQQgghhBBVSgI+IYQQQgghhKhSEvAJIYQQQgghRJWSgE8IIYQQQgghqtSSBnxKqY8rpR5VSv3zGY9/RCn1RP7fHfnH3quUOqiUekgp9fdLWS4hhBBCCCGEuBQsWcCnlLoaCGutbwZ8SqlrCp7+gtb6BuA1wEcKHv8HrfVtWuv/sVTlEkIIIYQQQohLxVL28N0APJC//QBw/cwTWuvj+ZsZQBfs84dKqUdmev2EEEIIIYQQQpRuKQO+BmA6fzsKNJ5jm78CPpe//V1gB/AW4B+VUuaZGyulPqCU2qWU2jU2NrbY5RVCCCGEEEKIqrKUAV8EqMvfrsvfn6WUehPQrLX+MoDWOqK1drXWY8AhYMWZB9Ra36u17tFa97S2ti5h0YUQQgghhBCi8i1lwPcEMDM0807gyZknlFI7gA/m/808Vpf/PwhsBKQLTwghhBBCCCEWYMkCPq31c0BaKfUo4Gqtn1ZK/Uv+6X8g14P3U6XU9/KPfVgp9QTwEPAxrbW1VGUTQgghhBBCiEuBZykPrrX+gzPufyj//6vOse1HgY8uZXmEEEIIIYQQ4lIiC68LIYQQQgghRJWSgE8IIYQQQgghqpQEfEIIIYQQQghRpSTgE0IIIYQQQogqJQGfEEIIIYQQQlQpCfiEEEIIIYQQokpJwCeEEEIIIYQQVUoCPiGEEEIIIYSoUhLwCSGEEEIIIUSVkoBPCCGEEEIIIaqUBHxCCCGEEEIIUaUk4BNCCCGEEEKIKiUBnxBCCCGEEEJUKQn4hBBCCCGEEKJKScAnhBBCCCGEEFVKAj4hhBBCCCGEqFJLGvAppT6ulHpUKfXPZzzeoZT6uVLqcaXUnfnHapVSP1BK/VIp9etLWS4hhBBCCCGEuBQsWcCnlLoaCGutbwZ8SqlrCp6+B/gL4JX5/wHeD3wFuAX4LaWUr5jXue6e+xav0GX0vo9Xx/vorpLzcSlxHJeRSIp02sa2bdJpm2TSAsB1Na7rYtu5f/G0jetqHMchm3VwXQ2AbbtorXFdzfHRBNPxDIOTMSYnJzkyMkn/xDRHRybpG4+STCYZjSR44WSUH794mL7xKA8//DCfeOBhPn3/Pg70R+mfmGZicpJjo1OcGI/QP5HbZ3hklPFokr0nIiSSWeKJLNPT00RiaQ70R0lnbZ7rG6K/v59s1mEynmUyniWZtEgmk0zHMySTFvH8e0wkEsQTWdJpG4Bk0sKyLGzbJZt1cFyNbbuzf4Ns1pn95zgOANn8466r0Vqf9rfN/d0cbNudfexc2xWa+ZsWmu/36nc+dR+f+9z89tnXF53X9jOfkfkYnorPa/uZ87KczHxHCnXfc9+C675/ffjxBe0P8MiREwvaf67PZjEKP+ulmKlHFlvfeJSf/OIXdN9zHx/52n3sPjbOZ37xGP/582foG49yYmyaRDLLwPg0R4cipDM2tuMSmU4yFY0RjaVxXZdM5tS5t+3cfdfN1REAtuOeVv75/D1m9svVu6efi1KPuVyd6z28+W8r//rhwIEDfHvvgXIXY8HOVactRj0nFs9v/tP8zoVaaOV+3gMr9UFgTGv9daXUW4AOrfW/5J97CLhda62VUj8A7gb+Hfig1npUKfUvwL1a633nO35PT48ev/Ojs/d7P3bXkryPpfapT93HP/aful+p7+PMSqD3Y3fR09PDrl27iq4gKvW9Lxfz/Tv39PTwuW/9jLd99nHOvL7yGIpr1zayZWUdjx4aI5q2SGZdMpZNbcCLYShcrbl5QwtXdDXw0/0jjEQT9E5lFvttXRQeBWuaQ/RH0riuxmcqXGBlfQBDKaZTFmnLwXI0rtYoBV2NQS7vaOCZ3kliGYv6gJd3Xread1/XTX3Iy56TU/zZt/fRO5FkRa2f/+/1l7O1vY5v7+7HYyje3tNFQ+j0dq2f7B/iwFCMq9c0cuumVnp6ephvPXeu7+KF9I1HueUfHwPAVHD0b+d+jT//zj6++Ww/Po/BN3/7eja3119w+11Hhnnr558FoCHo4fmPvGrO1/iL7+zj+3sGqQ96+cHv30hDODDnPkvt6d4Jfu+Lz2G7mv/xqi3cfd1q/O0baX/PJ2a3KaUeKzxnd/c08r/feuO89v+/vzjM3/z00ILKcHIyyff3DBLwmry9ZxW1Ae+8j/GPPz3IM72TXLe2iT965eZ57z+dtvj6MyfJ2C5vuKKDrqbQvI9xpp6eHiKv/CiLFSPV+k3edf0a6gM+PvfIEbK2S13Qy8YVtbz68pX85IVhwn4Pf/yKTfzn470cG4vz6m3tvOfG7gse9+hYnB/tHSLr5BrWTk4luXp1I3dft5rJRJb79g5RG/CwqjHEf/zyOHVBL3/9xstpqSn/92I+bNvlz7+3nxMTCX7lyk7eee1qGldtpP7dn5jdplKvBa796/sYTZy6X6nv48zfkJYHPnLa7xBU7nurFoXnqPdjd6GUelZr3XOhfZZySGcDMJ2/HQUaC54z9alIc+a5C20PgFLqA0qpXUqpXWNjY6c9t65CWx0Kgz0hLrYHDgyfFewBOK5mb3+Uk5NJoimLRMYmkbFxNESSFsmsTcZyODIa5xcHx0hkbUbjlRnsAdgaeieSOK6L5WqSlovluAxG0kSSWeLpLMmsQ8Z2sRyN5WgiSYtnT0wSTVmkLZdoyuLkRJK+ySQATxydZCyWwXZcppJZHjs8xtGxOBnLJZFxODGRPK0Mrqt5aTgGwIGhXFW4b2B+vW6l+NSDx2ZvO0W2/z1yaAytNRnL4XvPD865/T/94tRrRFLF9dr98sg4WmsiySyPHZ0ormBL7Id7BslYDo7j8qP9Q0vyGl/eNTXvff7h/kNzbzSHI6NxsrbLdMpiIJIq6Ri7+3Jlf65v/u8BoH8yRSxtk7VdjozNrzf4QhazQ8x2Nb84OMrjx8ZIWS4Z2yWSHzXwk/3DZG2XqUTu+947nsDV8Ezv5Jw9p4dHYtiu5thYgoFIkljaZmQ6Td9kkkMjcWxXM5W0ePClEWxXM5nIsn9g+oLHXI6Goml6xxNoDU8dmwQg6ZS5UIukMNgT4mIptrF/KQO+CFCXv12Xvz+j8Os989yFtgdAa32v1rpHa93T2tp62nPHKrS1QVpJRDm95apOfKaava/IVQo+U/HyzW1s7ainoyFAS02AprAXv9egoyFAc9hHfdDHVasbedNVnbTW+tnQGi7b+5gvdcb9gEdxRVcDfo9J0GvQGPYQ9nvYsqKGzoYgrXVBmsM+agMeQn6TkNekoyHIKy9fwcr6AHUBLx31QS7vqGNd/u9w52VtrG0JE/KbdDaEePW2lWxZWUtjyEtbnZ/1bTWnlcEwFDvXNBLymfSsybV3be+8cM/ZYvjzV6+bve33nPmXObe37OzAaxo0BL38+o1r5tz+f71+x+ztzobieiRed0UHPo/JqqYQd25aUdQ+S+3d162hPugj6PPwrmtWL8lrfPkNZ7V1zunz79u24Ne9vKOO+qCX9voA3c2lfZdv29JKyGfy8i2lna+1LWHa6wPUB71c3lE39w5FCnsX51LHA4R9Ht6+s4u7tnfQEPRS4zfpbAjS3hDgndd20RT2sbopxCsvX8EVXQ3U+D3cuXUFSl34u7V9VQO1AQ9XrKrnspV1rKwPsL61hnWtNWxfVZ/v3Qvy5qs6qQt4WN0cYuea+X9Wyq2zMcCVXfWE/Sav3Jr7nHRUVifleV3dsfAeaSHmq9g4YimHdF4N/LbW+reVUp8G/lNr/XT+uU8DV5AL/LYCncA3gE3Ac0AXcKfW+rxdBj09PXrXrl1LUnaxOGRI58VVypBO+Q4tb3KOlj85R8ubnJ/lT87R8ibnZ/krZkjnkgV8+QL8M3A1sEdr/ftKqX/RWn9IKfV+4M+AEeAl4CHgXUAW2Abcp7X+4IWO3dLSoru7u7FdjWmos1rsRfn19vayZk03rs6dI7G89Pb20t3dXe5iiAvo7e1lTXc3jqvxyHdoWert7WX1mjVowJyjF0dcfHPVc7Z8t8pOfouWHzcfGxhKyflZpgrjn2effVZrrS84lGFJA77zvqhSVwBv11r/uVLqH4A4kNJa/71Saidwt9b6jy90jJ6eHv3X//EDXhqO0V4f4FevXZrhNaJ0V+/cyQf+6euksg53XNbGjlUN5S6SKCCtdstfT08Pf/KZbzMYSbN5ZS2v3d5e7iKJM1x59U5+6x+/iuPC669oZ11rzdw7iYvmQvXc9/cMcnQ0TldTiLfuXHWRSyZmyG/R8nJiIsF3dw9iKHhbTxevu+MmOT/LzE/2D3NgaJoVdbmh5IZhlDVpy4UcBq5TSr0A9ABHmCNhC5ydtGVmYvnwdBrbqfw0xdXGdjSpbG665sBUaUkAhLjUDUXTgHyHlivLcWezt86cK1EZZr5Tg5HUgpejEKJaDEbSuFpju5qRaanTlqOZ+Gc0lsYqMttauQK+9wA/1VpfDtxHbi70BRO2wNlJW27b3EpnQ5DbNrfhMcv1VsT5+DwGO1bV09UU4tq1TeUujhAV6fbNbfl6rnXujcVFF/CYbF5Zy7rWMFd0NZS7OGIebt+Su4Z4+Za2OZOqCHGp2LGqnnWtYTatqGVLe225iyPO4dZNubrr1k2t+DzFxT+eJS7T+ShgMn97HOgGrgX+HrgTeLKYg2xoq2VDm3wYl7M7Llse2fWEqFRXdDVIILGMKYUMta1QW1bWsWXl4mUDFaIahP0efuXKznIXQ1zAhrYaNrTNb/pAuQK+LwNfU0r9GmAB7wDer5R6DOgDPlGmcgkhLiLJ4CqEEEIIsbTKEvBprSPAq854+O/y/4QQQgghhBBCLAKZ+CaEEEIIIYQQVapcQzpFlXFczU9fGGYikeXOy9porw+igR/uHWQqafGKy1awsj5Q7mIKUXGePDbBweEYO9c0sq2zvtzFEWfQGr71bD+W4/Kabe3Uh7zlLpI4B8fV/GT/MJPJU79RQoizRVMWP9k/hGkYvHb7ytxjSYsf7x/CaxrctaOdgNcscynFfEkPn1gUg5EUB4djjMcyPHtiCgDLdjk8Emc8luG5vqkyl1CIyvTE0QkmE1kePzpe7qKIc0hbDn2TSYaiafYNRMtdHHEeA1MpDo3kfqOeOxEpd3GEWLZeGIwyGElzcjLJoZE4APsHowxF0/RNJjk0EitzCUUpJOATi6Klxk9d0ItSsLYlDIDHVNQGPKc9JoSYn+6WEADrWmRB7+XI6zEIeE08hmJNc6jcxRHn0Vp79m+UEOJsa5rDeE2F32vQ2ZDrCV/dFMJjnP6YqCwVPaTTtl0OjcXobg4T8lX0W6l4QZ/Je2/sxnLc2a5+QynecEUH4/EMl7VL6mshSvHKrSs4OBJjR0dDuYsizsFjKN5wRTvprEtXkwR8y1XQZ/KeG9Zgu/q04WiW4zKVzNIS9mO5LtMpm9ZafxlLemmKpS0GplJskWuFsutsCPL+W9ahULNrvHU1hXjttnZ8HkVzjXw/KlFFR0l/+5OX2D8Qpb0+yCd+9cpyF+eSd2BomolElmu6Gwn5PFiOy+988TkSGYtfv6Gbd12/ptxFFKLifPhre+idSHD16kY+9pYd5S6OOEMq6/A7X3wW29X88Ss2cefWleUukjiPA0MxppJZruluIujLBX3f2NXPyHSadS1hJhJZoimLnWsauWVTa5lLe+lwteaPv76HaMrilk2tfPD2DeUuUsl2900Rz9hc091U0fPc/J7Ty/7Ai8P8n58dwlSKv33zdravaihPwUTJKjrge75vipHpDFPJLOmsTaACe/m0zk0kPzIa58YNLexc01juIpVkOJrmZy+OALk5La+6fCXxtM2BoSiuhm8/d1ICPiHmSWt47sQUGcflsSPFzeEbiqb4/vODBH0mb7l6FWF/5dWLlWQ6bdE3mURr+NG+4ZICvv0DUR46OEpXU4jX7+jAMNQSlPTSNhRN8dVn+jg2nuDnL43yF3ddBsBoLA1A31QS29Gz2wI8fGiMvScjXNHVIAHgErIdTTRlAXBsLF7m0pSudzzBQwfHAHA13Fqhn5mXhqb5h58exGMo/uL1WwH49u4Bjo0lUAp+sGdQAr4KVNFz+Da01VAf8rK2OYzHrMy3krIcXhqOYbuaff2RchenZH6PgZm/SAnlW04NQ+ExFKYBYb9krhOiFM01PgIeg5V1xWW5fWk4RjLrMBHP0jeZXOLSiYDXJOT1EPSZbGgrbW7Y3v4olqM5NpZgOm0tcgkF5HosJuJZXDcXXEwksnhMgzu2rMgNV9vezo3rm+lqCnHzxtyF+t6TEWxXs+dkpLyFr3I+j8Grt62kuyXMuyu4YTjoM1H5tpqZ66BK9OiRMeIZm0jK4omjEwDU+D14DIVH5XIziMpT0WftTVevYlXjJFva6yo24At6TbasrOXoWLyiW0wawz5+9douIkmLDa255BK1AQ/XbWxlLJbm/TevK3MJhag8SsHvv3wDz/ROcedlK4raZ8vKWg4Nxwj6TFbLnLIlVxPw8I5ru0hlHd66s6ukY+xYVc9DBzN0NYWoC0jj2FJoCvt4zw1ruP/ACFvb62gO+wDYvqqe7atyy52sb63huoJ9ruhqYE++h08srd942dpyF2HBVtQFeMc1XSQyDutbKzcx0M0bWnnq2CQeQ3HD+mYA3vuybk5OpvB5DN68c1WZSyhKUdEB3/Xrmrl+XXO5i7EgSiles7293MVYFG21AdpqT/VCGErxmXfvLGOJhKh8r7+ik9df0Vn09u31QX771vVLWCJRyFSKP7xz04KOsa2zXtZYvAhu2NDCDRtait7+lk2tMpRTzEs1rO+4pb2Oz7/nmtMe29pez1c+cH2ZSiQWQ2V2i1UZ19Uks3a5i7EkbMclbTnlLoYQFUvr6q0fqoXluGRsqecqQSrr4Li63MUQYlnL2A5Z253zMVE5KrqHrxporfnmc/0MTKWqLjOYqzX/+Xgv8YzNqy5fKUszCFGC7z4/QO94ku2d9dy5tbhhneLisV3N5x89ju24vPGqTlmaYRnb3TfFQwfHaAr7eOe1q2dTzgshThmIpPjOc/0YhuJt+WHqJyeTfHf3AKapeEdPlyzNUIEqOuD71rMnuf+FEXrWNvL+mytzCFPachmYymUEOzYWr9iAz3Zcfrx/mMlElju3rqCzIUjWdvnCE70ksw4Z25GAT4gS/PujxxmMptnaXisB3zKUzjr8cM8gtqtZWR8oKeA7Mhrj0cPjdDWGuOOyNpSSLJ2LzXZcvvVsP/1TKda1hjkyGuPZE1PUBry8dnv7acFf30SSn780woq6AK+6fKVkTb0IfvbiCP1TSW7Z1Mr6fB6ASpO1Xe7bN0g8bfOqbStPm+JSSV4YjPJcXwRDwdWrc5njHzo4yhefPIFpKDa3hblja3VMRbqUVHTz1lee6mNPf4RvPNNPukKHPAV9Jtd0N9EU9nH9+sqdjzgUTXNkNM5kIsvuvikgl4F0LJZhOm3xTO9UmUsoROVxNZycSpHK2hweLS5dedpyePDACI8fGceVoWtLLuu4TCSyRJJZTownSjrGM71TRJIW+wais+npK1HveIIf7xvixERpf4elNBhJ4zUNlAKPaXBwOMaTxyZ54ugEfZOnl/fZvkmmkhYvDccYjWXKVOJzS2RsfvbiCE8dm0Dr6vh+O65m/0CUSNLimeOT5S5OyU5MJOgdTzIez7L3ZLTcxSlZOuvQOxandzyBkx/C+fSxCSIpi8lElmdORMpbQFGSig74JpJZplMWUykLw6jct3LTxhbec2M3W1ZWbg9Ya60f0IzGMnQ357JTGTOt1Br8FZpFtZpkbZfvPDfA40Wu5ybKz1C5dNiQy4RbjKeOTfDggRF+8sIwRyt4TatK4TEUrutiu5rmmuLO0Zk2tOV6NFbWB6it4CydP9o/xEvDMX60b7jcRTlLa62ftlo/K+oC3HFZG+OJLBOJDIPR1FlZvje01qJUbkmUuqCH/QNReksM5hfbk8cm2D8Q5fGjE5yYqI5lVwxDsSK/7MzMd6ES5b6/HkxDsbaCs3QeG4sTSVlMJS2O55f2CfhM0BoNhCpwzetqMzKd5mvP9HFkJFb0PhV91hqDXqYSWer8JgaV29KVsR2mEhZttf6KHTqSyNiAojnsZSyeaxENek3Wr6wllrZkKNoy8O+PHef+F4cxDUVdsHIvKi813S1BDKXZVOQFxNB0mqNjCRQQT1fmyIdK4mqNjYt2IVriGnrXdDexvbMev8eo6OGcDUEfI1aahtDyq1+CPpPagJeg12BX7yTbOuo5PhanLuCl9Yz5SNtX1bNpZQ1ew+DJ4xM8dnicrO3yvpvXlj0L48zf1mNUz3poCnjntV1kbJeAt3LXr6sNePmNl63FcXVFzw+NpmySGQulYDqdBeDll63gxcFpDFNxYwWPRqsWf3vfixweTfCDvYP8+3uuLWqfstUWSqlfB94DmMC7gLuBXwFOAO/VWs/5yxlJ2aQth1jGqdgePsfVfO2Zk0zEs1zWXsert60sd5FK4uSHlpiGwnFyt12tGYtlSWatZTcs5lLUN5lkZDqNUoqpRLbcxRFF2nNympRlk7aLGxa9pinM1o46PIaiocheQVG6lOWgE7nA+vm+SMnHqeQL3RlvvrqT4Wia9oblOXfpyFiMA0MxghNJmsM+XhicJugzGYmmWNdWe9q2fk/ufCQyNnv7o1iOyzO9k7xhHkukLIWda5poqw1Q4/cU3etfCZRSVfEdMA2FWaEN9zP6p5Jk88k4BydzOSb8HhOlFAaqooPZarF3YJqxWJqRmIltF9ewW5azppTqBG7VWt+htb4NyAK3a61vAvYCbyzmONGUhe3mKuRkpjLnPViOy2T+4ns0li5zaUpXH/SigOFomo6GXAtoMuswPJ0mmrJ56KWR8hZQcP26ZjauqOXKVQ2SSbBCuFqTtBxsl6Lndl3T3cjrdrTzxqs6WdtSucOKKoXlaBwNjma2Lr9UBbwm3S3h2WBpORmMpBiMpElZDqsbgxweiaMUHByO8Qdfe57n+s7doHJZex3t9QE2tNWwXKbMdTWFqirYE8vLRCKDq3NzyEfzI7bu3z/EkdE4B0diPHZ4rMwlFJ2NQRrDPlbWBaDIDq9yhemvAkyl1INKqX8BrgUeyj/3AFDU6o4elbsgmpmEXYkCXpOVdQFGplNsqNDMVACj0xk0sLI+yLHxmXlDGlfnLoaqZXJ5Jduxqp5VDUE2tNWwqqHyF4e9NCi8Rm7Ik7/IVtWjY3H+z/2H+MQDh4imLu0A5GLwGgqD3DlqCpV2Ef708Qn+5Jt7+LdHjy1q2cQph0ZirG4K0V4fYGtHPW/euYpU1kEpyFgOn33oKEPR1Fn7HR9PEElZJDI2O1bVl6Hkl4b/++hR/uSbe9jVW7lJW6pFczjXgG8ALfnhzpFkFttxcRzNVKIyO1iqybXdTZiG4vLOuqLnVJYrSloB+LTWdwBJoAGYzj8XBRqLOYiLwmMolFLYTmUuBpm2HF4ajpG2XPb0R8pdnJKtqAtgqNxE0lOBq0KRuxCq5Hkp1eLgSIzmGj8aOBk5+8JGLD+Ggra6AD6Pweoie2U/+/AxTkwkeHFwmm/s6l/iEgqPaWCaCtOAFfWlDWX86tMn6ZtIcv+LI5yskkQcy81l7XU4WlPj91Af8nJ0LM5NG1tpCPpIZl2awz4ePzJx2j7RpMWP9w0znbKIpW36JqXeXApZ2+WBF0fpm0jylaf7yl2cS97KulBuaKqpcj1IkJsekL+gawwvvzm6l5p9A1HaagOcmEgWPVe/XHP4osDD+ds/B3qAmSaDOiByrp2UUh8APgCwevVqbmwLs+ekxco6PzXByhzeoLWmfzJJLGNXdHa2qWQWV+cCv77JJFva6zANhTIUaE3Yv/yG+Fxq2uv8fOOZKM21vpKzCRar+577lvT4lwqtwXFygV+xjVpXddXzTO8kpqG4UnoklpxSuQQaGoW3xLktM+vCtdUFaK4t7bs5lc86ubalpuLnEC2FWNrG7zEYmErxsxdHWNscJm057FzTSGdjEK1zw6QKhfwmHQ0Bjk8kqAt66TxjbqLjao6Px2kO+88aYjkWyzAWS+MxFKuaQpLZ8AI8psJQMBhNc+M6SQhSbl7PqXl6nnxd4vMY+PJDtT0VmjOjmqysD3BweJqNK2oJ+Yo7H+WqgR4H3p+/fSVwEngH8PfAncCT59pJa30vcC9AT0+PPjgcJWO7DEwlsSwLr7fyAiaPaXBFVz2DkTRXVPDFmdc0MJTC1Rp/fuK1ocBjgKmhJlCZAXk1+fbuAfYNRDANxYGrp+feQSwDmoH8MLMjRa7D9+s3rqVnbRP1Ad9ZF7Bi8dmuJmu5aGC8xORU61truLKrkeYa/6nlbOYhkbH58tN9ZG2X7Z31khX5HPye3G+UrTWumztfx8cT2LbL2pYQ77x2zVlBm9c0+MAt63nTVauoDXqoO6NR9qGDo+ztj+LzGLz3xm7C+SVUJhNZvvJ0H3tORqgPednRWc97X7b2Yr3ViuO4moFomnTW5oWhyv5tytoututWdIDvasB1ANDq1GPadfLjtmSKTrm9NBgllV8v0baLOx9l+URqrZ9XSqWUUg8B4+QydLYrpR4D+oBPFHOcrJPPCqkVSQvqKy/eQ2k4NBLn+HiCtjr/3DssU01hH++4poupZJZNK3LZzpRS1AR9WLZLi0wwL7vjYwmSlosC+pbJmlLiwlydazhxNbmupCJtba/cxqNKo7XGMBSa3JzyUsQzNo1hHxqN5cw/NX3WdsnmF0hOZGUpjnPpagqxtiXMI4fH6J/KNbBqQBmKeMY5bxIUn8c4b8NJPJP7W8/8/cP5n/Bk1sZxNVnHxbJdElkHrbVMbTgPx9W4rsZrGkSKTE61HEVTFl99uo+05fLa7SvZuKJ27p2Wocawj6Dfh6GYbeQwlQKVWzZGPsflF886eD0GWVeTcpyi9ilbE4TW+r+f8dDf5f8VbX1LmH2D03TU+6lfhuv+FGM6bTMayxD2ezg0UtmLJAe9Jpbfw8xoItNQpC2HtOUSKLLLWSyd69e3cGB4mrDPw46uhnIXRxTBUAozH/AFvPIdWo48hoHOL0hcX+L6lmubw+zqnWTLyjpq/PP/WW4M+3jl5SsYjqbp6W4qqQyXghMTSUyVm2/pakXAozg+keL4eIKHXhrlti1t59xvdDrNj/cPE/Z7eP0V7bNZSG/b3EaNf5KV9YHTAsZVjSFu29xKV1OQoNdke2eDXCRfQMBrEvKaDEZT3LWjrtzFKdnodJpkNnfxfWIiWbEB37qWMDV+E8NQdDfnMj0rQ+caHtHIiPHyu3ljC9/ZPcj2znrqi5zSVrl9zkDfZAqfaTCZzBJJpGkIL8+1fy6kqcbH9s56dp+c4rbNreUuTsmiKYsvPnWCrO1y/bpmbljfTDJj4806uBqe6y1uDTGxdMJ+kxV1AUI+E6mvK4POZ7kFyFiVmZiq2mUdFyN/jgamSkvqcWg0zsr6IJGURTRl0VBCts/LO+q5vEN6ds8nkbFJ2zbpfE/oRCJDxnaJp2yOjMZ58vgE169vPmfv6t7+KJOJLJOJLH0FF/L1QS93XHbu4bNXrW7kqtVF5Z+75CWzDv6sTX3QyzPHI+UuTsm6W8JsXFFDPG1z1eqGchenZEORJNGUhVKK0enccmGxlI3WGheIpSu3F7ZavDAYoy7gYSCSWvZJWxZFwJfr/q/xmxU7RyxtOQxGksTSNr1jldvDl8jYs0OKIslcKniPaeBojetCqIRWa7G4MlZuzceM7cm31IllT6nZ4LzYVtV02uavf3SAphoff/zKzUtWNJFz2owWt7SgvLs5xPMnp1jfVlO25F2Oq3lhMErI52FDW+UuEXQ+yaxD2OelqyFALONwZCTB4ZE4luNgOS5rW8IoYHffFBnLxec12NZRj+24HB6N0T+VzK3JJ0vaLLqA1yTs9zAyneaVlzeUuzgl85oGr9vRUe5iLNhLg1FiKRsUHB7LTf8Iew1cV4OCmgpOMFgtbMdhKJpmRV0Af5Gjfyr6KjyVdVCA7WisrIMnWHlDnqYSWX5+cAzX1XwzmeVPX7u13EUqSUdDkJs3tjAez3LjhlyWLctxUW7uYmhkurRkBmLxPHFsnKlElkgyy+GRWLmLI4oxM+9H61zG2yL80Tf38Gh+YdyGoJf33bxuzn3SlsNkIsvKugCGjNeZl4ztMtPcODhd2rqHY/EMbbUBrPxcsKDv4mc13tU7yeNHc8sSvHXnKrqKXAakUrTW+rlpYwvPnpgimbF5aXgaRa5h8rXb2nnL1at46OAovzg4xqGRGFd1NTAey3BoOMajR8YBza/fuKakIbfiwlxX01rjp8ZvSmPkMrBnII4DoGH3idy6iC8MxXKjTTS8MFDZiXWqwWA0jatdIsksll1cQ2OF11zqtP8qkakUfo+BZbsE5zlRf7k5c+6IkZ8r4bi59NaivLSr0WgMZeBKlq2KoBTU+E1SGZuGIueH2fkECIahsIrocbIcly891cd0ymJbZz2vkAyP81I4NctX4jxLrUGh0Rp0mb6bhRfapSafWe6u6mrghvXN9I7FmEpauFoT8Bq84aoOPKbBvoFpDgxNMx7PYLkurgZH54ZW5yhJvrJENLn5sK5EfGVX2GPkzy/PEPZ7Z9dVLkeDlDhDPviez7el8rrECoR8JqhcN7q3Qj+ALbU+tnXUEfB6uPM8cwEqVY3fg9c0UIaStXWWgc3tdXgMk6DXZE1TuNzFEUVQSrGqMYQyDLasLC4BwGsuXwlK4TEN7twyd52SsV2m85nxxkpcVuBSVhfwYqjchdCNa0tLmBLPWDz4Uq5nyVemNa6uXdvErZtbee32dtY0V2f9kHVcXhiM8sJwnF+5sp2r1zSwY1UDB4djaK1prw+wvrWGnasbuWVjK7dtbuWOy9qYTtukLJeR6TSffugo//7Y8dkMnWLhPKbiyq56Qn4Pd249d+IccfFcuepU4pyeNbl5we+8pgu/1yDoN3n3dWvKVTSRF/ab2Bp8ZvHrv1Z0wDeZyKJ1Lg11PF3aUJpyiyRtJpMW9UEPh4tcZ2s5ymYdPvil53jrZx7nsSO54WSRZJak5eK4mh/tHy5zCUU0aRH0Gfg8BpMJubCvBK6Gg8PTpG2Xp49PFrXPz14awWvmqvYHDozOuX2N38PLt7SxrjXM7VsqN3FUucTTdm6NKuDp46Ulp3r40DiprM3h0fjsuosXm2korl7dyOYiGxYqTTJj8Zv/8Qw/PzDKVCJD70QCj2FgOZojI3FSlsPLL2vj5k2t3L6ljeHpDM+fjPDM8Umytkvacrh/3zBZ2yWaskpO0CPO5riaoWgGUykZLrgMPHn0VD32yOHcMO+vP3uStOWSzDp8/dmT5SqayJtIZFFAIuMQTxXX+FTRAd9MKmx0bihAJQr6TKZTFoPRNJZd3Foay9H9Lw3z9PEJjo8nuPeRY0Cu52BGRAKMsgt4FZGkxXTKorW28jLaXopcrZlJzpksMkvnxtYaso6D7WouW1lc8o3agIe6oLfih5WXQ6ag3h6JJUs6RtBjMJW0yNoOdTJHbEnct2+YFwYipG2XsViGrK3pn0pxYHia2qCHkM9DW22A69c18XTvJAeHYzxxdALTVHhMhUZz3fpm6oJe2usDoDW/ODjKRPzUb1s0ZfHQwVGOjC7uHOm05fDo4TH29kcW9bjLhQYOjcTYOxBhIFLad0gsnunMqc/0zLXb0dF4fq1ReGkoWqaSiRmuhqyjsV236B6+iv5lsfL5yh0Njq7MMfWT8QwTiSxa64ru4VtRG8gtMuu41OaH18o0h+XlwQOjWI7Gchx+tH+g3MURRSjlK5S2HTyGgWFApIh0zcmszQ/2DOFqzdh0hrdf01XCq166nII5R/FsaY12kZSFz6OwXE3CcpBk/ovLdlwUzC7JADCVzNI3mcJjKoyCH6sf7hlkIJJiOmnzK1d1sGNVPdeubaI57OfyjjrCfg8r6wN87uFjxNIWu/um+IM7NmEaigcPjHBiIsnzJyO876bAomVcffzoOHtO5i6yG0O+qkuoYzsuveNxMrbL08cmyl2cS16mYNWFhJWr32autwHSskRQ2c1kw8/YmmQmXdQ+ldktllf4kfNSmeuChP0GQY+Bx4DGsL/cxSlZS42fta0huhpDXL6qAcgNEZrh80jPQbnNDDvTgN+o6LaeS4YqIYHHM8enmE7bRJI2Lw3NPTzKNBS+fAthoELnQpdTYQIPU5X29/OYRm5BcKXwmaW1lFmOy/QC18dKZGzSVuWONDmfBw6McHQsTmPIS9CjqAt4qQ96AQ2ui4Emmsz97Y6NJYinbfwexS0bmjk+nsBrGERTGX68f5hvPzfAkdE4aM3+gWleGorx8KHc0OmZHnKvacwOq14MM2sDGkqdc53ASue4mmTWxXZhOFaZ03OqSXPtqWXOOvPLkBSOClotOQDKbqbtSgOeIn93quaqz6Iy1wWp8fvY1lnP4dE4t26s3MQmKctlImaRsZ3ZhTpVQf9EKReuYnE1hbwMRNIoBW11ldu4cClRysBjKGxXE/IVdwE5PH1qbtGJicSc2/s9Jr96TRfD02nWt1bf+mtLzTgtS2dpwdqdl7UR9Bq01PgJ+ef/W5axHb78VB+RpMXLNrRwbQnJY46Mxrhv7zBej+Kd16ymMVyZa9ueSzLroJRidXOY4ek0qxvDeEwD14W44/CT/SPs6Z/mytX1jMUz2I6LxzT50tMn6Z1IMJ2fI9OzpgHDNMnYLq+/opORWIbGkJdkvmf3zq0rWNsapq02sKiB2Q3rmmmt8VMb8NJaW511t5HP6O2RZWHKLpcROvc70hj20gusbwuzfzCKAta2SMBXboXrv9pOcQ191RPwORZUYNA3nbYZiWXwmgYHhip3bbSR6RQTiSy243JkNHeR6bh69gOWdSTgK7fC4WaD08UNARDlpdHY+SGDqWxxw2jMgvnMusj0+o1hX1Vd4F9MTsHfOJstLXPjprZanj8ZYU1zmHAJvazTKZtIvoeqbzJZUsB3cjKFqzUZSzMSS1fV5+GOLStIZx1+ksgyEc8STVmE/R40mqyteXE4ytB0mvF4hsFIiqzjsr2znpDPgwJW1gWoD3m5bl0zXtNkR2c9HtPgXdetYTSWnl2SyGsabFlZd+HClEApxcYV1ZlMB3I93A21fuJpm22di//3E/NzeOTUyJDn+yIAHBiIzi7dcmAwcvELJU5TGPAl7UsgS2dFFz4v6DOp8XuwXU1bXeUm0oikLAyVW1cwbeUueuyCNcAcWVun7DauqMXnUQS8Bpe315e7OKIIhcvoFfsN2t5Zh6HAY8D160pbJkDMQ+H6dSUe4tBonPb6INGURTQ1/2GZLTU+rlzdwMr6ADesL22kyFWrG1jVGGTTitqq6+mtD3m5vLMeM99b7ri5xb5Nw8A0ckMxPaaB5bizSz0ppXC15u5r19AY9tFS46enu4kb1jfjyQ/X3NpRx22b22Qx9gXyewxevrmNjStquftaSflfbk5BRWbl7/RHTo0ceaGIqQJiaW1aWYPHgLYaHx2Nxc3preiYaWaIvEFuaGRl0vi9Bj6PwreIY/4vtqtXN+BqTdbVbM5nBvQVZA7yljgvRSyeV25pwWsa1Ae8XLe2cocPX0rMEoY3be2oI+A1Cfs8rG+T1vKlVljP1QVKu/BvCnt49PAYU8lsSYk+lFLcvrmNd167enbOzXw1hHy8raeLu3a0lzz/bDKR5ZneSSYTy28eVlPYSyLjgAavoWgM+UhkbBxXU+P3cmVXPW/vWUVHQ5CA16SzPoBpKJpr/NQFPAxGUnx3d2nJrjK2w7MnJosaYn0psl3NsYkEacvhub7SljYRi6e9/tSw4e2rco3D4YIpBe0V3DlRLd54ZScr6wLcvqXttN+gC1lwhKGUWqGU+jel1I/z97cqpd630OMWY2aUoAukM5WZtCVrawamUiTSdkX/GPxo7zDJbG7NvZ8fyK3Dly3IiCZDOsvvk784TjzjMDyd4T9+eazcxRFFcNz59xk9fmSCjOWQyNg8U+Tafamsw8nJJK70xM9b4fIz0VRpCU/+6/ETnJhI8OCLIwxWcFr6bz/Xz2OHx/lOiYHRUvrZgREm42ksN5epeCiaJJF1SNu5VFbrW2t441Wr+K/3Xsu7r1uNygeFrTU+RmMZjo8neK5viieOjs/7tR86OMYjh8b57u5BppZhMFxujqs5OZFkNJbhQIX3HkWTFsPRyp4yMRA9tSzD3oFcdthY5lQ9NzQty2yV22cePspgJM13nx+cTTg1l8XoUvpP4KdAR/7+IeAPF+G4cyu4NimlJXw5cFxNPOPkkp4UedKWo+lkwbot5xiSVORUIrGEpvLr6WhgLFbZP0iXClXC2ibxbG4hcFdDvIiGMMtx+dJTJ/jms/387MBIKcUUeU6JFV3/ZIrplMVEMlsVKc/L/WtsO2f/Db1Kkc7H42lbnzYn1qMMDEPhuJrjEwnG4lkcV/OyDc2sbAjymu0r2byihoFImh/vH+aZ3uIaUmYU+/ewHbfoebfVxFCQdVwyllPR7388nuELT/Tylaf72HMyUu7ilKzw6zPTcF94Wir5HFWLRMbGJXd+HKe4hsbFGHjeorX+ulLqzwC01rZS6qLkdW4Me5lKWAS8BqpCF143lCbkM3Fcl/oShwMtBw0FS0rUBHJJBwqDcH+RXc5i6XQ3h9k7OI0CtrbX80C5CyTmVMrP6sbWGl4cnMY0ikv0kLVd4pncvNuJuPQ+zJenYLh6bYl1eF3QZDim8BnqtONVmrdcvYqjY/GyzgHc2x/h5y+NsqIuwNt2rpp9POT34MkHdSGvQW3AZDRuYSjFyza2sHNNI5975BhTySxBr0ltID8EFLh5QyuWrXH1KI0h32mLrRfj1s2tNNf4aa3xnzcZzqGRGD/eN0xDyMs7rumqyuUXzkdr6GgIkrVd2upKG5K8HERT1mySreU4rLlYdQGTqfxohVUNIaaAsN8klQ/+KnkJsWrREPQymbDwew3Mi7jwekIp1Uz+2kQpdT0QXYTjzmkikWu9TlouAV9lBkt+r4dY2iaWtkmUmOFtOWiuOfUjFsyvuVf4EUxWQat1pWuq8WGqXCDeEq68jLaXIqOEHr605eSyqbkay577exf2e7hjywp6JxJc0y1JXuar8AzZVml1eE3AR8CbJuQzSzrny0Vj2EdPuLyfoZeGY2gNw9H07GiTJ49O8MUnT+C4Gk1uioHHNHFcCwfND/b0c2Bomum0Tf9Ukuawnz999Wa2duTmwBqG4s6tKwj7PUwls9ywvmVeZfJ7THauabzgNodGYrha0z+V5NMPHSGStLiqq4HXbG+v+uDP5zHoWdPEi0PTvP6K9nIXp2Rrm8Nc091EPGNxTQmZcpeLWMHQ9IlYAgPIWqdGi7h25Y5GqxbJrJPr4bNcgt7i4p/FiJL+CPg+sF4p9UugFXhrMTsqpf4IeLPW+ial1J8AvwKcAN6rtZ7XJ+qRQyO8/LLKqyiOjcWJpy20hhMTlTt3YzJx6nQl8wv3TiQtKu+MVK/JRHY2EBiv4NbHS0mihIyN/ZF0vrFFcXC4uKVegj6DsN8sevK3OCVluTTkb8dKvA5a3RSkdyJBU9hHXYkZHw8MTTMcTXP1msb8ouKVR2vNsyemSFkO165twu+Zf6Bz9epGplMWHQ1BmkK5hshvPHuSiXh2tsc862hOTp0a1j6RdMgORsm6uTJMJDJE0/ZZyWvmyoCathyeOj5JbcDD1asvHOCd6YpVDYxMZ5hOW/RNJBnKzwNb31bDjlUN8zpWpXFcTdpyaKnx0TteubkMDENx08b5NQYsR4XNVpNpaAGmCy4ZDozJlJBym+lEcYC9J4sbYr7ggE9r/ZxS6lZgM7nGzoPFBGtKKT9wRf52K3B7PvD7U+CNwDfmU45bN6+cb9GXhY56P6ahcBxN2F+5rXjxzKnaIJ1f7y0gF4/LykQsjSaX7GhYJl1XhHDAw3zP1K0bm2aHdL5y29z1YjJrc9/eYVytGYtleMc1q0sr7CVqMfrjQj4PXlMR8JqoEuajj8XS/NP9B4lnbG7a2MoHb9+wCKWav3gml3xsTXO4pKUKjo7FefRwLimKoRQv2zD/i+cNbTVsaDt9SOmW9lqeOT5x4R2VotZvkLZzv8VXr26Y92s/cXScB18axWcaNId9tNb6OT6eYHVTaM7sq11NId5301peHJzmG7tO0j+VxFCK9vrKHeJYLA08eGCE8XgGX+V2cJPM2nz+0eNEkll+42XddDVV/gLlM6fD4NSyM7WVfJKq0LrG4lYpWIwsnW8G3kAu4NsEvF4pdYdSqm2OXX8L+K/87WuBh/K3HwCun285UsnKbBVSCvyGxqugtkKHpQIMFLSWposYRiYuvkj6VLvdSIVnEbtUFDsZu9Dz/TEsV5Ox3aISB3gMA78391MQlvXE5m0xRmA+e2KKiViGIyMxosn5974nsjbxbG4o73isfI0533q2n/tfGOHbz/WXtH/I5yGWtpiIZxb1s7iqIcTKCwROCgj5TDatqOWrH7iWr7z/eta3nZr/mrYc+qeSpK3Tv48Z2yFScL56J5IcHonz4tA0iYzNd3YPcP8LI3x9V/F/j60ddbxm+0p61jQR9pskMpU71aNYsbTF0HQGy4VvPz9U7uKUbM/JCI8eGmNPf4Qf7Knc9xHwnKrUms8x/UOSri8zRnEjOhajRn0fcAPwi/z924AngU1Kqf+ltf5/Z+6glPICt2qt/1Up9b+ABmAmF28UOOdYCKXUB4APAKxevfq0ltWxpKKmAteKTdkuaUdha81kCcO3lguv51QNoPPxniy2vrxkT0sfLz18lUCXEE3s6Y/k9gUePzLOh+7YdMHtfR6Dd16zmpFYmrUtld8ifbEtRjU3HE2RtjWW6zKdmn/A19UY5rXbV3J8PMFbr+5aeIFKNBMQnRkYFWs4mub+F4axXM3mlbVc2dWwKOXaNxC5YBINDdT6vXQ0hjg5meal4TE6G4K8/ZouYmmLjz9wiBcGptmyspYPv2ITDSEfacvhi0+eIJa2uXljCz3dTXQ3h9i8ohafx6DG753NuJqxc9kni826q5TCn5+3l7YvSg68srIdd/Z6rpKbi32mYiKRxXY1bgVnsrTsU2WPZyxqOP28WFblvrdqFE0Xdz23GAGfC1ymtR6B3Lp8wGeA64BHgLMCPuDXgC8X3I8Anfnbdfn7Z9Fa3wvcC9DT06MLV8OpDVZmF7NC4fMYaNshVME9fEZBZ7E33zokmTmXF9NQWPmrU1+Rk3xFeSk1/+9Q4YhAo8jhgfUhL/Whypz3VW6FSTVLHZRfE/ASSVn4TIVRQsZp01D82vXdJb764nnDlR0cHI6xZWVdSfs/cXScWMZGa3j86Di/dkP3IpVMkT1jqYb6gMl02kEDXgN61jbywds38pP9wwAMRlM8eWyCB14c4eGDY2Rsl0gyy6cfOsK7r+vG7zWI5UdN9E+l6OmGG9a34POY1AY8rG4O0dkQ4KWhaW5Y3zyvJVZWNwaZSmbxeww66qt/keu6oI8EuYvJzvrKzQBZH/Jx7domsrZ71rDiSlLYxDCzFKxXwUycVxOQa7vlxOe5SEM6ge6ZYC9vFNiktZ4EztdltRn4XaXUT4DLgR7g1vxzd5LrIZwXu4QFipeD+qCPzoYgYb+Hy9vnTqG+XHU1nPpRCvlylz2F6cUl9iu/wkxvtYHKnS8qLqzwgumyIpZlANjdN8V3dw8wGEktVbGqljrP7fnY2FqDzzRpCPnobAyVdAzH1WUf/tdeH+S2zW2sLDFIuaa7kaawj9qAhxvWXThBynzUBTzUFSyZ4TMVN25ooT7owWcq1rbU8Lu3rifsNzk0EmPPyQjXdjey52QEx3VBQ8Bj4jUN/KbJvoEobbV+rlrdQGdDkOvzZQ14TV62oWU2ycrx8SQdDUH6JpPzWrvsyFiCxpCPkM9DbwUnc5uPllofNX6TVU2lff6Xg+7mMLdtbuNlG1pmPxOVqPByzZuv1IL+U7VbY500GJdbKav3LMZZe1Qp9UNOJVl5C/CIUirM+Xvq/nTmtlLqMa31R5VSf6qUegzoAz4x30LU+IqLcJcb13UZj2eIpW2Gpyt3XlV3W/3s7Zba3LkobNGUaX3lt641xHN9uZHTr7q8gwfLXB4xN08JCTy0PvVlK6YnN56xeejgGJCbC/au69bM+zUvZb6CTJKlNnwnLAevR6G1Jpq2aAjN7/csa7t88MvPcmwswR/csYE3XLlq7p2Wocs66vm167tJ2w53bF28RGw7uxvZdWKSA0MxHA22o+mbSMwGyI0hL31TKfbsHaJvMhdgDUbTdDWFODAUpSHoxUWzos7PZDLLppU1KKVY0xwm4DF4+NAoSinefGUnB0djZGyXsM/EayqSWc32jvp59fBtaKthb38Ur6lYUwWJP850cjLJQCTF9s56wn4PpqGIp21SlosuYd7ycuExDV5dRKKs5c4DzAyAnqnTVMHY9YBZub2w1cLkVE9sW5ENbIsR8H0QeDNwU/7+00C71joB3D7Xzlrrm/L//x3wd6UWIpJMUxOsvKDv8GiMyWSuI/SFweJSqC9HLwxEZm8PR3NVRTxtU/2DUSrH833Ts7c/8bODZSzJ/HXfc19R2/V+7K4lLsnFlS3h4ufg6Kleuh/sGeB/vGbLBbcPeAzqg16iKYsVtfKNna+k5TDT3BUvsWErlrKwHJcUqqRJTA8fHOWRQ+NorfnYjw9WbMBXG/Dymzetndd8t2IMRtLUB7yzySZc4IWh+OzzE8kMu3qnGI2liWcsPIZibXOY3Scj2I6mdzKXFG4sluW2zQFS2Vyylu89P8DB4RhHRmM0hf2MTqeIpmziGRvb0XQ0BLmqq4GXX7ZiXuVdURfgd25dt6h/g+UimbX54pMnGI9n6J9M8taeLqaTWdz8fMenC36nRHkUznadsHLLMsQKHuydqMwkidUkWzBgYNfxsaL2WYxlGbRS6ii5OXtvB44D31rocecrlUmRm/5XWQIF/bKVPA12KHrqIjORn9dQwXOWq1LhdeREQpK2VALXnX/AVzi63XLmjh48psHd161mKpllZZ0EfOXQXONjMGoS8pn4vPPvJmyq8eaW93E1tYHKH2612IGOoRQDFxiu3BLONRZnbRePYWAYCsNQKKVm/64ZyyWRcdg7EOXyznrWNIVRqPyc2Vx5E1mXff1RMo5LW42fjoYgpllat281BnsAyYzDz14cJmO7RFMWb+3pquhELZeKwnNUOIpElJ8ucih/yb8MSqlNwK8C7wQmgK8BSms9Z6/eUggFKi/YAwiY1REVWQVjNmd+p0oYjSYuks6GEMfKXQgxJ1VCGpCQCfF8nLhhRXGJAwJe85JY72u5Stsu8XQuWYlZwoX+zjXNfOR1W3m6d5L//srNS1DCyvaKrSv4r8ePn/f5oM/D9lX1KJVLwAK5jKFvubqTq1Y3sKI+wDPHJ0lkbTobgjQEc0mO3rKzk/F4K/1TSdDQ3RwinrZxXE1PdwPrW2u5vKMyr02WiuW4NIR8ZCx3dq1Gv8eYXW+01i8T/pejwiGEAU/lNypVk1CRCdcWctZeAh4FXq+1PgKglPrwAo43bx4jNzdMASsqNLPT3qHqmJBdmGhgttNSAr5lJWBCOl9jb2wLSsBXAYrNslnI4zMglWuAUUvUnjQyneaBAyM0hny86vKVmNK6syBDkTQuueUMRqbTtM2zp9V1XU5Opcg6Ln2TSdobJHgvFPCarG8OsetE5KznFBAwDeoCXi5vr+W/Hu/FUFAX9NAQ8nH1ah9b2+v4wi97ue+FIWxHs2llLhnSqsYQqxpDs8tHHB2LYyhoqvXx2u0dpyXKEjntDUF+/YZuXhiM8parc0OPa/xeGmr9TCWz3LZ5riWcly/Lcfnx/mHiaZtXXr6ClprKvC49l8KxJpZV/WtDVpKGmuISHS2kKeUtwDDwC6XU/1VK3cFFvsSf6VTSQH8kfsFtl6sdXedccrDiHBo5Ne4+kz8vRpUOSalU6YIa+4ljkbKVQxSvlHhtZu0vgLFYcYmgDo3E+NmLI4zHixvq++yJKUanMxwcjuV6N8SChGczGxvU+ucfJBwaifNM7ySj0xm+vXtgsYtXFWIXWBuwdzLF5pW1fGv3IBnLYTpl8w8/eQnbcfnu8wN87uGjHB2PE/CYxDIWB4fPPd/+meOT+YQ7ilS2cpOPLLU3XtXJn9+1lS3tud7PjO1gOQ4eQ3FysnLrk2OjMb79XD/fe36Axw6Nz71DBSm8mpMllpeXsXhx2bVLDvi01t/RWr8D2AI8BHwYWKGU+oxS6pWlHrdUHfWVuebJVLJyM3MWOtck3mLmD4mLp7A7f0Nb9WV+q0ZuCb+svoJf5ra6uXt6klmbH+0bYv9AlAdeHJlze8ilH1cKagMeWmurpxW7XDoaAngNgxq/hxr//NdDXNUYosbvIZGx2S5DCM/SOxbnl4dGz/mcAWQch7HpFFd21pGxXWxXM57I8uBLI3xrVz+PH53g5FQKn8egMeSjuzlXf1qOy/6BKKP5DNtrW3KPt9T6T5tLaTsuLwxGGY5Wx+/9YnM1JLIOGdtlPJ6de4dlqncyycHhaY6Px3nu5GS5i7OoCn+JtDTmLys6W1y9shhJWxLAl4AvKaWagLcB9wD3L/TYc/EaMNOYrR0LPJV34RGJV8cPQJ3fC5z+XiRpy/Li8RjY+W7xcAkXleLiM5l/L4FjnJptEcucbynUUzyGQdBrksw61AaK+1xs7aijIeSlLuAh5JP5HAt1YDhG1nGYSGgmkxla5ps8R8G2zjrWtYZpkUyrZ3nLZ58gmjn7B2km34rS8L09QwxMpagJeEhmHNAKv2ni8xhkHZeruup5885V+E2TphofluPy3d0DnJxM4jUN3vuybq5b18y2znoCXvO0Yc4PHxpjb38U01C854Zu6oucc3OpMA2Fi0ZrqCuyDlqOfKZB2nKxHbeqe8HSdhW/uQrU1FBcA/6i/lLnF1v/XP7fkisYuUTaMai8cA/GqyTgyzhnVwBSJSwv6YLEOruOT5SxJKJYrpr/IAxdcKVhFzHVwucxeOd1qxmdztDdXNxcgOdPRvjFS6MEfSbvum510YGiODfH1cwMmspY8x8Z4Ti5ZQxCPg9ZGVlxlvONNtGAo2EqZTMez5DKOtT4PWRtl6YaL2tbw/z5XZcxPJ2ifzLNvz/Wy1gsw45V9XhNg90nI3gMxYa2mvw5hLD/7MuqbL7udbXGduX8nMl2XPLJvTk2Xrkp/8djWbK2g6thqoJ7Kuci13bLSypV3GetatIhpazKTDMfrJJsR/H02R84yeOwfCVkfklFsEpoSW0KnZoDtqXI4X11AS8b2mrwFJlCfiif4j6VdZhKzN2LKC7spg0t1PhNVjcF6Sphoe2agIeA12AommJ1Y3FB+1J45NAY//qLIzx2eHnNX/r426/gQlMjtdZ0NgT5s7u2cGVXA1evbqS9PsiXn+7jwFCMG9a1MBbPEEvbxNIWqaxD/1SSdS1hmsM+XrejIz9379xu3dzKdWubuGt7O83nSeRxfDzBZx46ytefOTkbIF4qUgXzKzMV/N6V0hiGgVIKo2qursVyFyky/KnoaGNmSKcCanyVt+g6QHtTbbmLsCimImcHfOfo9BPLxIZGj2TprABe81S68mKNxU4FYC+cLO7C+0tPnWB/f5S37FxFT3fTnNtfu7aJlOXQGPLR1SQZIRfK7zEJ+z25odYlNJRNJDKksi7t9UEOj8a4cnXDopexGHtORrBdzfMnp7hpY0tZynAud2xdyY42D88MnbvLO2u7dDUG2d0XYWd3I4eG40RTFgGPwaGRGJtW1KC1prXGx+YVtWxcUUNj2Mfx8QTXdjfR3XLhID3k83Djhgv/PV4YjJK2HAYiKUam03Q1lS9wv9iCXpOZtBOVvIxkNJlrFAB48mhxi2ELsVDNNcUN46/gr9apOWIKKnaYxMR05Q5fKHTunGViOTE4tXiqEbh0LiYqWbyIOXhnyhRUhdH03K0uI9Npvv/8IAD/78kTRQV8NQEPHQ1BmsK+ql0g+nz+4jv7iKYs/vpXttIQXpz5cscnEsTTNq5OM52yLthbdC5NIR8/PzDMiYkk/+PV5VuHb/uqevb2R9m+qqFsZTif8wV7ACGfyVee7mPnmkYeOjTGdMqmvS7Ao9Mp0pbLUCRJ2O+lLujl3TesoT6YG8J8TRHflWJtaKvhscPjrKgL0FZXiRNUSqdUbkSQqyHsr8zGe4C//tGh2dtjCRlFIy4Og+Lin8oO+PL/u4DHrMy30tIgE+zFxWEYMNMu4pXxthVBK2feHT6KU3WjU0RDWH3AQ13Aw2gsQ1djcb11jx0eZ29/NLf/dV5WzDfJSIX6xM8O8b3nc8se/M/vaD797p2LctxY2sJydX5o2/yHRnxz10n2D8bQwN/++CDvvmHtopRrvm7b3Fax66itagozmbQ4PpYg67iMxtIYSuFqzcBUivVtNbxmWzsh39KsrTcUTc9+jybiWTouobUUtT7VgF/J2b07an0Mxqp37p5YnsaixS1lUtGjjP2e/NpFhsKo0B6+QJFzZoRYKE9BkFcjSTYqgt+dfyux1zx1novpKTIMg22ddWxeWcOWlcUNMZ+Z66cUl9Si6+GCiWA+z+LV3U1hLwFTUeNV+EqY1+3xqFMNoJIeed5WNQT5s1dv4Q07Ogj6TBQKRa7XSevc39Q0FD1rGrDnMVfBcTUZu7jvsNc49Z3ymJfOdwpyf6eZv2oiW7mLej/+56/gxrUNrAh76P3YXeUujrhEmGZxveKV2S2WF/AapCwHn2ngqdDU4P4quVg6lQheLFdGQc9BwLM0rdRicfkCYebbXmwVXJDaReQGT1sO+wamiaVt9gSned0VnXPu87L1zTSHfdQHvbScJwlFNXr/LevJWC6RZJZ7XrN4QydHY1miaZu0Y6JLCNiuX9dCfcBDPGtz6zKaO7dc7OqdxAOcK5TwGKBRfOHJXhpCPravqufFgRhrW0K86vJ2njg2To3fpLnGz4MvjfLQoXGuXtPIrZtaL/iaiYzNV57uI5FxeO32lWxcceHGlBvWN9MQ8lIf9NJ2iS2tkczYsxejFZyzhVja4vbL2rl+g8PRsTjrWytzfWhRWTqaLoE5fLG0hQaSlkPWsvF5Km/s90i8OsKk6ngX1S1TcLUznazMrLaXmow1/29WYbgwEkudd7tZCvb2R4im7KIz647FM+zum6Ih5KO9PlB0ds9q8Pt3bFz0Y45Mp9E6lzykbyJJ+zyH80Xi6Vx2Qw0nS1zc+8hIjE/+/Ai1AQ/3vGZLVS21cXQsfs5gD2aGEzporXmub4pYyuayjjoUmvv2DRJNWwS9Jn2TKWoDXrZ11nN0ND5nwDcynZ5N4HF0LDFnwGcaim2d9ec91v0vDFMX9NKzppGfHxyjLuDhtdvb8VbBdy+atmgudyEWwXA0TTz/Q3t8LCEBn7gooqniesUrOuArXK4oks5QE6y8gG88Xh0X3iGguFHEolwKQ4cnjk0gaVuWP9MwmG/alrBPkcjmwr4r2uceonlkJMZU0sJxNQeHi0u/9L3d/fzXEyeoC3jZsaqeNc3zX0qgEmWzDh/94QvEsw5//tottNUtzjwrn6lwAaU1bbXzD7SOTSRn19nsm4iXVIYf7htiZDrNyDQ8dmSc12xrL+k4y9E13U3Ucu7kYq6GG9a3EPJ7cBxNd0uYWNpmKJripeEYrqsJ+wxWNYXxe01crbl+XS48iSSzPHV8kvb6ADvOSFTT1RRiXWuY6bTNVQvMmrq7L8J4PMt4PMtYLMOekxH8XpNtnfVVEVTUV3JqzgJBJ84/P3gYgNdubeHOrSvKXCJxKRicLK7Or/ymobzmcGVevraGquMUnCvY81TJcNVqlKzcaRKXFMuZfw9f4RyjkcTcJ7ox5Jvt2St2Xtp3nx9iIp7l+HiCB18cmXcZK9UXnjrBQ4fG2NU7yT8/cHjRjhvN9wS5wMmpInplzzAYObVPKlPamLidaxoxDUXIZ563p6lSrWutOW+DpAZu2tDK9Wtb6G4O47qa1c1B0lkHy9E4GmydW1rhyq4Gfve29WzNr2/50MExXhyc5sEDo0wmcoOvj47FOTA0jcdQXL26kau6GhY87HldaxhDKWoDHhQwFsswMp0mW+T8wOXO762OgO+WTz43e/tHLy6vtShF9RqMFDeqoyzfMqXUdcDHyXU67NJaf1gp9SfArwAngPdqreds2C5MM29bFixR9qyl1Np4qmW8OkI/Icqn+577itquUibU+5TNfC//PYZBJp/prraIlvPW2gBXddUxPJ3l5UVmWFzdFKJ/KomhFFuK6EWsFmuaQiil0FrTUWRG02J4VP7XTEN9CSNVCrOrBryl/ZLcvLGVK7sa8BqKQIXOib+Q84VGHgO+v3eQlhofz/VNYTmaiUQWx9X4TIWhYOfqRv7mTdvpaAieNnx5ZnkGv9cg4DU4Pp6YXeJkMJJi30AUrSGasnjZHOvwXcimFbWsaQ7hMQx2901xcipJjd9TNXP9TEPNnp9KzldTFzCIpCt4EqKoSM3h4r405arVTwAv11qnlVJfUkrdDNyutb5JKfWnwBuBb8x1EJXPP64AV1VesAfQGPDPBq5hX+WGfF44a+iZW0TCCFEenbX+eS/oLS4+wzv/i//1K2rY2z8NwJVdjXPvoF0m4jaxtM3wdHEthZ+++0r+64k+Nq6o4cYNF57LVE1ecflKgn6TWMrmNdsXb8jjjs5anj8ZpSboobWENdi6W2oJeSDrwIY5FgG/kGqat3emgAGF1+Jb2oK4mNQGPBwaiTGdCqDzi5oEfSaXddSxsj7A8ycjPHtikv/2td189l07WVl/Kri+dVMra1vCNIZ9hHwebOdU80zGdhd1qYGZrORZ20Wh8JoGYX91BOaGoWbX4aup4OGdn373Ndz9+acA+OBt68tcGnGpWLeioajtyvLN0loPF9y1gR3AQ/n7DwB3U0TA11rrZyKeIeTzYKrKDC5iWRu/18BxdUW3qp4rVK3MM1K9vAqs/EnZuqqO3eUtjiiC486/ubuzIcj+/mmUgrVFzO8ZjWdJWg4Br8lAsUNDPB7ed/O6eZetGty0BAHuppV19EcytNb6CJbwO+D1GHQ115DKOnS3XTo9rvPhPSPg8/t9/LfbN3Df/mGawz4cV/PO61YTTVqEfR42t9fy+UePcWgkTsZ2iCYs+iOp0wI+w1B0FwTYG1fUcudlLlnH4cquRta1hokmLa5aXUTDS5GGomnqgl5cDfGMTcBbmY3dhQIeA4/PJJl12LSicuckpiyHbR11OFqjVAV3VYqK0lZfXCNfWbuUlFI7gBYgAkznH44C56wdlVIfUErtUkrtGhsbo7s5hAZaav0VmbAFoLulhoDXwHI0nRW80Oq5RtNWbn9ldVpVMOzrjq2Sur0SlJKBb3Q6jUuutXwkMveA0HWtNdy1o4OuxhC/c2txrdIT8Qzffq6fX7w0WtIyAuJ0Pq9JTcBD2J+bozVfm1bUsqoxN9zwts2XTo9rsfqnkqcleQNIZSz+7qcH2dcfoW8yyaYVtVy7pgnTMEhaDs1hP3945yZu29JGV1OI11/ZgQK+vuskx8cT532t7avq2bmmCdNQbFlZx3Xrmhd1zcYdq+oZi6fxeRRNRayzWQmyjsZUioDHKDrj4HJ03dpmAl4Tx9HcIsujiCVUmCLDbxbXSFi2LiWlVBPwKeDtwE5gZvGnOnIB4Fm01vcC9wL09PTo3X1RbBd6J5KMx5K01FZe4pa9JyeZymfQ2NcfLXNpSuf1cNYkCRnJvrz0TZ66+H/ghbEylkQUyy5hKNhAJJeeQgOHx4rL3nXbplaaQj62rSouWcfTxyc5MZHkxESS9a01rG6uvLp3ORmOpOgdTxDPLzU0X4dGYhwdTZDI2jzwwjBvvGrVopexkj1+ZOKsISeHRpP4PTMLrCsGV6R48KVRDo/mvjMr6gL0rGnkndd20TveTHPYy88PjOIxDRKZUTob1vBc3xRhn4ftRX5vimU7LrtPRvAYiiu7Gk7rLXppOEYq6zIczdA7kWBdFWTp9HkUrsoFfk2hyh1W/NTxCdKWg2koHj40xrXrqmGxCbEcFc6Y6lvOWTqVUh7gi8Cf5Id3PgP/f/buO06uqzz4+O/c6Tuzve9qm7rVJa/k3is2zQZjGzDgJJhAIPELKSYQiCEkJoROQoeE3sHYAhvjLnfJ6r2tyvY6s7PT7z3vHzO7WsmSdna1qyl6vp+P7Nk7tzwzZ2557jn3HK5IvX0t8GI66xltF29aGvcUnnXJBta4HvVyub8t8yT3paVFQ3YZ//va1p67NxfOJfYptNYKjeulsX9k4ic1B4IxvvbkPp7c3cNXHk+v58m6VGsEj9NGqTe9C7TBkRj7e4PybO9JPLW7l1DM5OhQhL1dgYkXOMFgMEpnIMJAKMa2zvSG1jiX1JV4cNpfe0KKJZJN7yoLXbidNuZW+zAU+ENx3HaDLe1+1m7p5Bfrj/Do9m4CqTHW6ko8vHSwn99tPMr/vXCQg31TGwrjVDYeGWLd3j6e2t3Lnu7j1314IETHUJgDfcG8GIMPks8lhmMmpqVp68/dAZ48DhuBSJxAZLKD6QgxdXYjvewhUzV8twGrgc+m7lx9FHhGKbUOOAx8KZ2VGAZYqWsbhzLJxWEFCz3Hruhyu3cqJ/7o8ReXDiM/Tkb5yO2wMb2XKGImGFOo77GNO5C402hK5rArbEby4tedZoa5vKGExrICPE5bWs8QBaMJfvTSIcIxk9UtZVyVZm+goVgCl92GLc+HeDEMhQa01nim0HmXPVXmydqqqccxWjuRL4nEqEvnVVBf4ibQe3wT5wqvnQ9ePY/L5ldiNwwqC93Mr/bx6uEhntjdw8VzKjBSdy5thuLiOeW0NpVS4XPxnWcPsG5fP1prFtcV01IxfTVtrnH77YnNQedU+giE47jsNkq9uXmj+0QJSxNP3fweHd4iF7VUern2vCrCMZPL5+dv0+r8OjrkvkJ3eseBTHXa8lPgpydMfgH47GTW47QbJGIWDkNhqdxL9gBKvG58ThuhuHncM1a5przQyRH/8QmfPZcz2Dw3t9LLpkwHISZmTP64Vl/sZjCUTOdnV07cgUeB005rUxk7OgNcNonnTiZzsRkIx1nfNkgk1TlMOgnfhkMDPLOnj3KfkztWN07rc1DTaTqOcuVeB4OhOA5DUTqF57IK3U6K3XbCCYum8qn10nmgN8jDWzpx2g3uXN1IcQ43rTuZhfUl7ByX8NkNKPO5eengAD99+SiRhEldiYdij4P+YBS7zeC6RdXcsaaRq4cjbG/3s7MjQMLUvGF5HVVFbkoLHBhKYU5zrfXS+mI8Dht2m0HLCb2uXnNeFbNKPVQVuvDlSS+dKvVPA84cvm6wtGZn5zAjMZOrz8vdWr7RsgBwneR+XrE79zsKyieGkV555PTRwuO0EUtYOO0K0zSB3PsRKgzqSj0EIwlmleXuczDB8GsPbtN9EhRnxm1XRBLJMvHkyUC3+W4qzR/nVReysyuIYSgW1U5c6xCKJbA0LKwpomd4ZgbrsBmKuhI3wUiCsjQTxYN9yaZd/cEY/nCcysIzG7x6pkzH5enASBwDMLXmaH+YWWWTqy0q9TpZ2VTKwEiMi6b43NCh/hCmpQnHTDoD4bxL+Kp8x//uitwOEpamfSjCUDg57l5PIEKFz0mh206Fz00sYbG4rpi5VT42Hh5CKTXWYcsNi2sYGIkRiZvcdv6ZPzPZPhTmse1dlPlc3LSkhnnVJ79Z47AZLKmf3mcGM83jsGF3GoTjFgtrcreX2S1H/HQHopha88L+fq5bVJPpkKZk/FknkWotaOPYYyFTebZczJxEms+DZect0zTNqyrE47Qzq7QgZ4c0KPM5uXx+JTUlbt64fPrGdTrbrlxQPfZ69EdVO6776vmVuVt7mT+OXZoWunL3Luq5xDGFWq2eQARTQ9zUaTWPKnQ7OL+plAqfkwtnqJOBqkIXTeVevC47KxpK0lpmfpWP7kAEn8tGhS97m64VjOuiuNo3tSSppMABCuw2g9op3Pir8DkJRuK0D4YpnWKitryhhJpiN7MrvcyexuaJ2aLcd/wg5cOROIaCmmI31UUuijwOyrxOZpUWcP3iGta0lLGwpmhs/kvnVlDuc3Jpqhbc7bBx9yUtvPey2Ww8MsQTu7qJJaZ+Ibzx8CCDoTj7e4J0+tMbHiVfRBMWIzELS8Ourtx92GBWiZtCtx2Pw0ZTDt/Av3R2ydjrt52fvC5dVn8sEX/TirqzHZI4wfgruII085/czJJSWip8aA3lXmfOjnkSiSe7O5lb6WM4krvdET+zv2/s9fhTXlOZh4SlWdEkvVVlWmTcxcjT+/ozGIlI1/ghD9JN/bZ1HOu044ldvXzg6gUTLpN83mTmnjnpC8YIx0yqi9zs7w2mVUOxvy9IdZGbYNRkMBRPu2bwbIuN63hrJDa1Y3ihy4HbbuBy2LCmMMzF07t72ZLq5flbzx7kttWNk15HmdfJnWsmv1yuiJrm2ODeAAkLRqIJQjETr9NBQ2kBw1ETh83g4jkVNJxwwb68oYTlJ7lZsaMzwMbDQ0Dy5snq5rIpxTe3ysf+nhGKPfasrc2eKcOR+NjFaDCWu93XtVQV8obldYRiJpfl8DN8Nrsdh6FAQcxKnnk6ho/dPNwknb5l3PizRCiW3nOvOZ3wrW4upczroKrQfUYPqmfa/t4RRqKJnH5Q/qLZZezpTjZ1caQKw2k3qC32EEmYLK7LryYouchhMDYW1eLaYl4d917zfWszEpM4PQXYDDCtZOcq6TivtpAXDw6igMuzZEy2AqcNl8MgGrfSfkatpMAJhHA7bHiyeHBpz7gavoayqT0/V1Xkomc4gsth4J1Ca5WGsgLsNoOEaWV1bWgmNZR5cdkU4VSzdkMpInGLIpcdDbiddhIWOGyKQnf6ZVDicaIUaM2Ua1ch2aR6doUPu6EwcvmCZgoKnHZGL1lz+BE+fC47f3FJC6bWOX09V+VzkUwpFNXFyZsPi2qL6An0ohRc2CJjDGaaTcHovcbyNI87OZ3wvWF5He2DIWqKPTlbw2coxdL6YvzhGPOqcrft+nsuns1jO3oYHInxd9fOB5LP7SydVUz/cCTvnjnIRX937Xy+/tQ+fC477796Pu/9WqYjEhNRSrFiVgkdQ2EWpXnT5JNvXMK//HYzRQVO7lzTPLMBpsnrsvPOC5sYGom9pubkVK6cX8nsCi9lXudxSVW2Kfc6KS/1kEiYvC/NgetPdMPiGoo8Dsq9LnyTSDZGzany8b/vWc1LB/v4i0taphRDvls2q4TL5leyqzOAacGyWUXUlBTw7ouaKPY4CUTiWNqitMCVutmQnsbyAt5xQROmZVFTfGaPLmRrx0QzrcBlp6nSS1cgzJtW1E+8QBYzDIUxLU/2Zs7bL2xiKBTD4bDxxhX1/AL4/t1r+MKfdlLqdXO3HGMy7rzaIvZ2D1Pmc2K3nwNNOl85OMDLbQMsrCnkxiW5+fyb025wx5oGDvWHWFRXNPECWWo4kiASt1CGwf7UwLXRhMWDGzuIJkw8LjvnN5VmOMpz29ULq9h0eIiqIvdxz6aI7KUU3Li4hnX7+7l5aXodAOzsDOB0OEiYcHQwlBXNw+KmxR+3dtIdiHLdomrOq53497ejM8DjO3uoKXJz66p67Fl8x7zS58K0rClf6L1uaS0NZQXUlXjSfh5jvGA4xqfX7qQ7ECGh4QNXzptSHPlsdqWPS+ZVUuxx0h2IYKKYW+nloc2dlPtcnN9UwmM7uikucHLb+bPSGm4EkuNL/m5jO6bW3LqqnqpC98QLideoLvbgsBl4XfnVWVAuGo7E2d41jM1QxOLJJrZP7urmoS3dOAzFBc1lLJKb+Bk1ryr5GNisUg/2NIdTyumEb3tHAK1hZ+cw155XndUXBKdTW+w5roOTXLS/dzg5cKrW7O0dTfhMbPEEpob9PSMZjlDs6BxGKQhE4rQP5e7gttMh3SasbQ/cPMORTGxfbxC7odjZld6A2gVOGyUeBw6bgXOGjol9wShP7uqhtMDJ1QurJmyCNjASo2Mo2RHFzs5AeglfRwDT0rQPhRkIxbL2QlopxaxSD5bWeKdQOwfJGtALzqDDnE3tfnZ3BUhYmoc2dUjCdwq3rqynwGFj3b4+bAZ877k2aordLKkrxrQs4qambzhKlz9Cc0V6zXMP9o8QTA3IfqB3JGt/p9lMAXUlbnwuG+XSJDnjfrexnaFQHAU8tKUjNa2Dbn8EpeDhrR2S8GVYXamHUNykzOsk3ce+czNDSlnZWILLYbCioSRnk718UVrgIJKwiJmaRKqPWLfdhqk1sYRFQ6mcBDOteyjEK4cG2Xh4EOlVOXcMhGJ0+MMMhtIb12l+dSEx08IwkgMBp+Ng3whP7+llKJTew9/P7evj+f39PL6rm6OD4Qnnr/C5CEbibDwySH1Jeje35lX56PJHKHDaKPdmvpbyVNwOgzWzy1nVVMayWSVTWseRwRBfe2IvT+3umdLyhU4H4bhJOG6dUU+R+a7Q7eDSuRWE4ybP7e1nOBxn4+FBnt/fx/xqHz3DyZsSdWn+RiE5EHq5z0lJgYP5pxhKQZyezVDMry6k0O1gVaO0BMo0hSYUMxmJmWPPItqUJhQ3CcVMPLn8oGWeWFBVyEjUpKG0IO2m4Dldw9faXEbrFHvEyibtQyG2twe4eE7FlJ7fyAYPbe4c6zVoV3eyhi8UT2BLWGgNW6RXp4z74/ZugpE4I5EEz+yd2oWlOPu0Bruh0FZ6F/KP7+ymZzhKfzDGywcHuHKCQc7DMZOHNndgWppuf4S3rW6YcBs9gQj7e4K4nQYxc+Je9dr6R9h0dIh4QvPHbZ1p1WYdGghRU+wmHDcZjsQn9VzV2WQoxV0XNp3ROr74pz1sODTAo9u6OK+2iOqiyd0g294xyGie136Odek/WV9/ej/7uocZjsSJxC1QcLA3yLeeOcCS+hIgWYOdbtJX7HHwrouaZy7gc4ClNSPR5MD3hwdCOX1d1x2IMBJN0FLhzdm+JZ7fPzj2+sld3QBsSPVEq4F1+wf40HUZCEyMWbu1g+5AmCd29/AXlzSntUxuZhd5JBJL8PHfbiMQifPMnl7+7ZalmQ5pSkq8x9rd24zk3QaFwtLJC9Z47va0nDdipollgVIaxeS7fheZobXGUCrt7vr7R2L0B5MDqEfTqO0xDLDbFKalcTnSu1No6WRvhkorVBrPrWlL0xOIEjct+obTq0UcfYbKbqi8b8Gxvm2ATn+EbiNKIByddMI3fow5V55/V2fK67ITiVuYOjWEkIaoqen0R2ipSFDoduA6RztPyRyF3aaIJXTaz05mo55AhB+9eAjT0ly9sCqHE9dj5xpbKml1jHtObCrPGYvptbs7SF8wysBI/Pix0E5DSi3DogmLw/0hIgkTu5G7J5kr5lfzg+cPkbBgdWPy+RyHTWHYDRKWpqkit59RzAcrZxXTPhjBaTOYUylNj3JHMhkz0jw+XDKnnI6hMC67jTlpNOl02W3csbqRLn+EOVXpNQFtKi9gYW0RbrstrR40bUbyFkM0YaV9QXfNwioaywqo8LnwufL7VGVaFpbWGEDCmvzNmBUNpcyuKKB/JMZbV+V2L4cz7UNXzyUaN3lmTy+HBkJorSlw2Chw2vC57Ny+uoFyX/Y2Ic5HhoI7VjfQHYgyr9qX6XCmrCsQ4dXDg5impr7Uk7MJ3/nNpTy8pQsFnN9cxlqS55W23iBKwcVzZFzlTCspcOAPx/G57GiV3jkjv8+iOcBpt1FZ6KLdH6ahLHeTIkOB22EnZlq4HMnaPoXC57ITS1gUSc9bGVdVnGzr7XHYsqLnRpGeoVCM3uEoVWmW2ZrZ5aAUHoeNOZXpXTyVeZ2TGtj88vmVuB02SgucaTV9C0QSKJK1dcOR9J5FtNuMtDp3yQclBU66h2O47AYFaXaxfRwFdSUFeJx2vG451p5KJG7y8OZOaorcrGwsYSgcIxhNdjZWU+yh0OOgplieN8+Ecp8r5xNtj8NGfYmHWMKiLEuboKej2G1ntJK7LDXGW4HLjs/tSF7rOXO3ciJfFHscWFpT4DSw2c6BXjrzhddlo8jlmNKAu9nCIHWX2tIkUs8aOWwGjWUFjMRMFkuPThm3v3eYkWiCaNyka2jijjZE5mkNg6EYltb0B9NrCmlayQfudao5dTqPkWw96qetf4TVzWVpXfAWOO0TPhs4XlWhE5/bjmlpGsvTG4cvV2jgqd09xBLWWCI8WR6nDbfDSC47hUG3Fcmu1APhOJGYtJ8/lQN9QZ7f38fASIzaYjcep52RqInPZcdhM7hsngwonSmbjgxxdDDEmpaynO3ptLnCyxXzqwhG42fU626mDYzEsHTyuDIwkrxBt7DGR12JG7thpH0jUcycw/0hQlGT7kA07Y66cjfDAPzhOHu7h2kq9+ZwjUWy56ORWJxIPJHpYKasazhKOG6hgc5UpwFOu4HTYdA/EmNJnTQhzLQd7QFipiZuarZKJzppyfTwDUoln4kNxUwcaT5X9PzePr637iAeh0FNsXvCngNHogke3NSOPxynPxjjPWk8AB43Lba1+ykpcNKSRvf1GoXbbiPiMPPuGbNI3GRjqkMDn9vOxXMmnzR4HTaiCQuXzcA7hearA8EoRwZChOMmOzsDk17+XBGOmuzuGqZ9KMzBPgdDoSimpYnETVbMKqax7OQ3I4LRBLs6AzSWFVA1yecrxcRMS/PgpnaCkQRDoTjvPMNOkDLFYTNYUFNIMJqg2JO7Ne0j0WTCBxBNJF8sry+hzOvC5TBYVCc38DOtfTBE3ILBUJxIIr3cIafPvA9t7uDZvX38asNRrCk895ANYglN+2AYfyjOwb7cHRvt1bbBscd8Dw0kP8dgKMb6tkGODIT41MM7MxecAJLlAckaib40a4tEZmmdvNuq0HQF0quVfXBzB5uPDPLSwQFe2Nc34fyW1uzrCXJ4IMShgfTGy1y3r4+HN3fyi/WHx7qyP53hSJzeYJRQNMH+vmBa28gVdkNhpKpRpzp8xCuHBombmoFQnAPd6Y23OF5XIII/1evk7iksf65o6w/RF4wyEk3QPhhmOJocey+asNjW4ecPWztPutwftnTy7N4+frnhKHEZ02baaWBP9zB7e4Y51J+7Y/YeHQzx0OYOntzVw0sH+jMdzpRt6zxWBi8eTJ5Dfr7hKPt6htne7uehzR2ZCk2kxFKHIQ3E02zVkdM1fJDswS6XxU2LvmCMWMLk8EDuJnwnE02YOMxk+bSnMVaXmFlelx1/JHlgKCrI3buP5xqf247NgJI07xjv6xkmktCAZn/vxMmVw2awvKGEQDjO4jSfmdvbPcyWo4M47DaGw/G0m2BpSKNPz9zisBncdVETCdOacu1PJNUkRwOD4ckPqzDaL7JGIx3wnlpxQbIHTq1f27Hdrq5kC4jrFlVTeJrnICdzyaG1RmswptBM91yjUKBz+/iQ/G3oSf1GslE0fmzviEST1wwDI1GGw3FQydZ1InukmwfldMJX4XOy6cgQqxpLc/aAGksksHRyrLpEmuNsZaP60mMPKDtTRWGMe3gozd7exQzy2I+VR1VBrjaBPrcoBYtrC3n18BDnN6bX41tiXA1EJD7xMcXtsHHHmgaODoZZlGbCNxCM0jYQpsBpEEljzBW7oRiJJgjFzCn1QpntJtPhzcmM/0qcUzgtl3id2JLpHqVyM+eUPA4bpqVRBmMZn079NxSzQMNwJPGahO91S2vY1TVMY1n6gxyHYyY/f+Uww5EENy+rZbY893RKCphX5WM4mqCxLL2egrNRmddJLKEJhGNUF+fuOdbQx47pbqeNIDA4HGEkdT4ZieTu40f5yEjzmJRVl+FKqS8qpZ5VSn05nfm3HPEzHImzrd1/3EVOLnE7HGPj1bnS7GknG+3qONaMKBQfvXo5dhVj5votrzzQGzx2V25z+0AGIxHp0ho2Hx1iOBrnxYPpNREqdh87jlT60ktEHtnaxf8+d5DtHek92zkSNyl22/E47Aylcbf3yEAI09LYFHT606vtPzoY4qcvH+ap3T0535JjMtqHJ98aYnAkQtxK5jADIbn7firP7OnBQnPivdVYQtMzHGVPd4Ct7X7ME25KFLodrG4um9T4iB1DITYdGWJHZ4DNR+SZ6dOxGYrDgyO8sL8Phz03b94DdAyFcdoNKgrdHOzN3RZb0XH53OBIckzXV8f9hp9IDcYuMmd88uaypXeTMGsSPqXUKsCrtb4McCqlVk+0TChuMhJNEM7hUb0HQ8d62Bl9xioXRccld6OvRqLHymU4mpsJeT4pGNcZRK10PZ4zBkbiRGIWXf70mvr1h47td0fT6I31yMAI33vuIJuODPHZR3altY03LKujpcLLyoYSVjaWTjh/c4WPkgIHboct7aEWXjwwQJc/wsbDQ+fUM6cFtsknt3u7Q2PH3XSHvTjXmJbmyGCY/uHYca1ebSpZw5QwLQZDcXZ0BDg6eOYX66bWjEQTyZ6RE7l7jXI2jEQTrNvbT+9wlC/9eU+mw5myhrIC6ks8FHscLMnlnsnHZwap/HvJuI5aLppCx1Rieo2/om4bSO+57Wxq0nkR8OfU6z8DFwKvnG6BlY0lFHscVBS6sOVok87SAgcep41YwqJ0ig/8Z4MLmsp4ZGsPAJWFyVqFgnHDTPhcuVt7mS8uaCnjiV092AyDNbMr+WWmA8oj6fTmOaWePBUUeRyEognK06ytW1RXTPtQGKUU56eRjPmcdpz2ZNPMojTHcFvTUs6alvS7Ha8pdvOei1vwh+NcPj+9i4XGsgKODIQoKXBQ5MmmU9X089gNwgkLBdSUTr5H44vmVOCwKUxL05DDTeJm0kgsQVmBkzKvg95gHKWg0GXnhiU1PLu3D8vSNJYV4HXZpmU8uKpCNysaS4klLBaeI+NJTpWhklm3qcGVZvO0bOR22Hjb6oZMh3HGzqspYnN7srffNc0VbATuuWIOMUtjMxR3rGnMbIDiODW+9M7b2XQWLQH2p177gcUnzqCUuge4B6CxsZHXL6ujOxCh3OdEpTPYVBYq87n59rtaeWxHN395aUumw5myd108m6ODEfZ0D3P/m5YAyYu8y1bUsq9nhK/euSKzAQo+d9sKfvLSIRpKPVy1sDrT4Yg0KOAztyzh6d29vGlFXVrLfOmOFXz76f1UFLp42+qJT8ylPhdffNtyNhwe4k3L688w4pNzO2zcdVET4UkllWUsrC3E47DhyLOhHE70rXe38m9rd3BRSykXz6uc9PKLZxXzu7++iFcOD+Vsl/Yzrcjt4LrFNZT7XAyHo0QTmjtWN+Jy2vj76+YzHE1QWejC7bCn/Zze6ZQUOHnXRU2EY6YM5TABj9PGPdfM5+VDA/zt1XMzHc457zd/cyn/+uBWvE4b/3TTIlZ/DS6eW8H8Gh92w6AkhweVzxcPvPk8vvj4Pm5aUkttWXo3lFS2PBuhlLoXeBcQAAqAn2qtv3iq+VtbW/X69evPUnRiKlpbW5Eyyl7jyyfd8ebEzBtfEyj7UPaTMspuUj7ZT8oou0n5ZD+l1Aatdetp58mihO8jwB1a69VKqReBX2mt/+tU81dUVOjm5uazFp+YmD8UZySWoK7EA0BbWxtVdbOIJSy5I5QlhiNxHDYDt8NGW1sbmdiHwjETU2t8454pDEYTmJbO6cFqZ0KmykikL1vKKBCJE4wkqC1252yLl5lwJuVz4jlNzIxs2YfEMT3DUQwFFT7XWPl0ByLYDEXFNDR5FtNrw4YNWmt92qYJ2dSk88/A7UqpZwEn8OLpZm5ubpY7Dlnkd68e4cO/3IJTg7fcw9P/cDXnLVuB8y3/ic2yuGhBFV99+6pMh3lO+9yju/jdxnbsNoOv3bmSu9987Vnfh57e3cPHf7cNS2vefVEz91wxh4c2t/Ox325Da80bzp/FJ9+45KzGlM3kzmr2y4YyeuVgL7d/62WcGmyFLl7+2LUZjSebTLV8Ht7Uzt/+fBNODQVlHp75x6tnIDoB2bEPiWP+4vsv8+SeXgCuPX8Wf37gL1n0/v+mb28fSsFNaxr51C1LMxylGE8p9epE82TTgxF7STbnLCP5DN/zJ86glLpHKbVeKbW+t7f3bMcnTuOVtqGxsaT6U93/h2MmZqr/60MDI5kKTaQc7Ev2PJcwLXZ3p9er03Tb3TWMlWpVcKA3+ZvYetQ/1u3+3l75nQgxWa+OO/4GIudOj6Yz6eWDA2Pf6cCI9Hwqzh37eoNonRwWaGtqqJ59PUEgOW1TuwwzkouyKeF7N/Co1noxsBZ454kzaK2/pbVu1Vq3VlZO/sF2MXM+c+tS6ordeJ027rtxPgDlXidrWspoKvfy8dcvynCE4u+umcvCmiIum1vJLWl2ADLd7rqomdUtZSyqK+Jvrp4DwN9eM5+ls0poqfTyiZvPy0hcQuSy9101j9nlBXidNj541bxMh5MXPnXLUmaVHH9OE+Jc8JU7VlDstlPmdfDfqQ73vnzHsrFp//OO5ZkNUExJNjXpVMDoaNB9QA4PYnJuev6j1xz3t1KK775nTYaiESdaUFPEd98z4fCWM8rjtPGl21ceN83ntvOT916YoYiEyA9P/MNVmQ4h76y775qJZxIiz6xoLGPzv95w3LTVLZWvmSZySzYlfD8Bfq6UuguIA7dnOB4hhMgK6faiOqWx/oQQQgiR17Im4dNaDwFy+0AIIYQQQgghpkk2PcMnhBBCCCGEEGIaScInhBBCCCGEEHlKEj4hhBBCCCGEyFOS8AkhhBBCCCFEnpKETwghhBBCCCHylCR8QgghhBBCCJGnJOETQgghhBBCiDwlCZ8QQgghhBBC5ClJ+IQQQgghhBAiT0nCJ4QQQgghhBB5ShI+IYQQQgghhMhTkvAJIYQQQgghRJ6ShE8IIYQQQggh8pQkfEIIIYQQQgiRpyThE0IIIYQQQog8JQmfEEIIIYQQQuSprEr4lFLvUko9rpR6SilVn+l4hBBCCCGEECKX2TMdwKhUgneF1vqaTMcihBBCCCGEEPkgm2r4bgBsqRq+ryqlbJkOSAghhBBCCCFyWTYlfNWAM1XDFwLedOIMSql7lFLrlVLre3t7z3qAQgghhBBCCJFLsinh8wNPp14/AZx34gxa629prVu11q2VlZVnNTghhBBCCCGEyDXZlPA9DyxLvV4BHMxcKEIIIYQQQgiR+7Km0xat9SalVFgp9RTQB3wxwyEJIYQQQgghRE7LmoQPQGv995mOQQghhBBCCCHyRTY16RRCCCGEEEIIMY0k4RNCCCGEEEKIPCUJnxBCCCGEEELkKUn4hBBCCCGEECJPScInhBBCCCGEEHlKEj4hhBBCCCGEyFNZNSyDyF3+UJw3fO1ZAuE4f3PVXN57+RwsrXn9V55lMBTjH25YwJtXzsp0mOe0nuEIj+3optjj4MbFNWPTtx71s/HIIOfVFrG6ueyUy29r9/Pq4UFmV3jpGY4SNy1uXFxLJGHy553dlHicuOwGHf4wl86tYHal72x8LCHOefF4nNd95Tl6hqO866Im/v6GhZkOKef83/MH+eJjeyh0O/jtBy6korAg0yEJkRHP7unh3p9vxqbg+3evAeDxHd38w683YzcUP/yLNSyoLc5wlGKypIZPTIvvrttPlz9CKGbyv8+3ATAwEuNQ/wiBcJzvPHswswEKXj00RE8gyt7uIIcHQmPTn9vfR38wxnP7+rAsfcrln0/Nt3ZLJwf7RugYirCtw8/Gw4P0BKJsOTrEs3t76Q/GeOngwNn4SEII4DevdnJ4IEQkbvKzV45kOpyc9N11BwnFTLoDEb637lCmwxEiY77w2B6GI3GGwnG+8NgeAL7yxF6CkQRDoTifT00TuWVGEz6lVI1S6o1KqTcopWomXkLkqhsW1+KwGSilWNVUAoDPZcftsKGUYnVzaWYDFLRUeDGUwueyU1XkPm762PuGOuXysyuSNXaL6orwOG3YDUVjWQHNFV6UgjKvk8bygtS83hn8JEKI8S6dV4HLYQNgUW1RhqPJTec3lqKUwmEzuOa86kyHI0TGXL2wCkMpDENxfWpfuGJ+BSo17aYlcjmfi2asSadS6q+ATwBPAAr4qlLqU1rr783UNkXmLK4v5ul/uJzuQJzF9cmqfrfDxmP/7zL6RxLMqZLmfZm2oKaQpvIC7IbCbjt2r+f6RdVcOreCAqfttMtfu6iai+aUU+C0ETc1lta4UxeZTWVeHDaFUopowqTAKa3FhThb6ssKWPePV9PWH2R5g9xcm4ov3rGSv2r3U13kkOac4pz2oWvm87qlNXjsdurLCvgc8OHrF/LGFXVj00Tumcmrsn8AVmqt+wGUUuXA84AkfHmqorCAisLjp5V43ZRIZU/WGE3QxlNK4XWldygYnc9pP74m0DMuWZRkT4izr7jAwfICSfbOxOjNSiHOdXOrXttS4GTTRO6YySadR4HhcX8PA/JwgRBCCCGEEEKcJTN5K74deEkp9SCggTcBLyulPgygtf7CDG5bCCGEEEIIIc55M5nw7U/9G/Vg6v+FJ5lXCCGEEEIIIcQ0m7GET2t9/+hrpVQpMKS1PnWf70IIIYQQQgghptW0P8OnlPqEUmph6rVLKfUEyZq+bqXUtdO9PSGEEEIIIYQQJzcTnbbcDuxOvX53ahuVwBXAv8/A9oQQQgghhBBCnMRMJHyxcU03bwB+qrU2tdY7SaMJqVLqw0qpdTMQlxBCCCGEEEKcU2biGb6oUmoJ0A1cBfz9uPdOO1qjUsoFLJ+BmMQMGxgYYNV/vgDABU3F/Pz9l2JZFnM+uhZTw7subOBTb16W4SjPbY9u6+C+32yj2O3gT/deMjb9YN8ID2/u4MndPRS67NSWuAlGTVoqvLhsBvt7g5iWRWWhi6FwgivnV1HgshE3La6YX0XMtHh2by8lHicep8Gmw34cdkVrUxmL6mTcHiEm8uGfb+TBTR2U+1y8/LGpPfmw6lOP4g8nuGN1A5+5VY61J/PC/n56g1EunVtBmdfJG778DFs7h5PNkIpc3N46iw9fvzDTYYocFYkk+MsfrccfivOZNy9heWNujov5g+f284mHdgHwo/esBOC/H9/D5x7bC8Av3reGNS2VGYtPwIMb2/nlhqNcPLeMD1w5L61lZiLh+zvgVySbcX5Ra30QQCl1E7BxgmX/Cvg/4FMzEJeYQdd+9ZWx1y8d8gOwv3eE0lRd7w9fPCIJX4Z94sEd+MNx/OE4n3ho59j0x3Z08euNR+nxRwGNy2HDYTPYdGQIt90gGE2gtSZuakoKnOztDnL5/EpshqLI42A4kmBvd5BowiQStzjQG8RhNxgciTO/2ofdNpPDfQqR+367sQMN9AxH+f6z+7n7sjmTWv4zD21jIJQA4GevHJGE7yQ6/WFePNAPgALmVHrZ2pkcKtgCugNRvvdcmyR8Ysq+8ewBthwZAuDf/7iTn7/v4swGNEX3P7xr7PVf/nAjhcB/pZI9gHd+5xX2fOamDEQmRn39qf0MR+O09Y/wjgua0lpm2q/EtNYvaa0Xaq3LtdafHjf9D1rrO0+1nFLKAVyhtX7iNPPco5Rar5Ra39vbO82RizNx8eyy10zzuY/dT3DZ5aI/0+pLPQAYSnFB87Hyqix0UeJxYjPAblMUOG247AYlbjvFHgcOm4HDZlDkceCwKaqKXDjtBkpBpc9Fhc8FQIHTTrnPSYHLjtdpp8zrwGaojHxWIXLJ+OPjqqbJ1wpcNLd87LXbYZuWmPJNodsx9t1UFrqoK3Yx/uhkAMUeR0ZiE/lhaX3R2DlvbmXujkBWXHBsP5hVmmyY53MeO640pK4lROZUF7sBKPbYjyub05mxYRmUUuXAJ4FLSQ68vg74lNa6/xSL3AX85HTr1Fp/C/gWQGtrqwzxkEW+dtdqnD9bz8ttQ6y7L9kkqbbYw51XzmbDkUF+8t7cvNOVT37zgUv4+lN7mV9VyDWLavhMavobltVxQUs5OzsDlPucFLocDIWjVBe60SiC0TimqfG57AyEYsyvLsK0NHHLGkv2Gko9eJw27IZB/0gUBVQUulBKEj4hJrLxY1fyD7/ewa3n17G88bU3zyZy9Xm1fPWOZTy6rYfPvWXRDESY+3wuO++6qInhSIKa1MXSk39/Bf/4y81cs6AcDDt3rEnvTrkQJ3PNohq+9x4nHf4wr19Wn+lwpuzVf7me93znBQrdDr76zlZafwZbP3Ujb/36c5R7HHzzPWsyHeI57xvvPJ91e3tZ1VSKzZbhhA/4GfAM8JbU3+8Afg6c6gGFBcAKpdRfA4uVUh/SWn91BuMT0+wLd7S+ZtpHbjwvA5GIU3n/Sdp6220GdSUe6krG37XznXT55lOst6rIPfZ6lvO0j+oKIU7g8Xj42jvPP6N1vGFFA29Y0TBNEeUnr8uO13Xssqe5wscv3n/JaZYQYnJWNZWxKtNBTIP//auLXjPtV7KvZA2P08Z1i2smtcxMJnxl45t0Av+mlHrzqWbWWv/T6Gul1DpJ9oQQQgghhBDizMzkg1VPKqXuUEoZqX9vA9ams6DW+tIZjEsIIYQQQgghzgnTXsOnlBom+cyeAj4M/DD1lg0IknyuTwghhBBCCCHEDJv2hE9rnbtdEwkhhBBCCCFEHpmJGr6FWutdSqmTPreqtX51urcphBBCCCGEEOK1ZqLTlg8D9wCfHzdt/BAKV8/ANoUQQgghhBBCnGAmOm35jlKqRmt9ldb6KuB/ST67tw146wxsTwghhBBCCCHEScxEDd83SI21p5S6HPgP4EPACpKDpk9b0rduXy+/2nCUaxZU8YYVuTvIZVvfCAf7R1hWX0x5aiDrXLT8Xx8lGE3w8ZsXcvelcwB4/482cHggxH++ZRmL64szHOG57fcbj/CRX27F6zTY9K83HvfeQDDGT14+RMy0aCwtoNTrZHVLGUVuBwB7u4fZ3hFAa40G5lcXsmSC8tRa8+rhQaJxi9UtZThsBolEgr//1Va6/WHefmETl82rpNjj4NXDQ0TjJq3NZVhas75tkEK3neUNJfQHo2xp99NS7qW5wjtTX48QU/bjlw4RiiV4xwVNFDgnf1r92UuH+K/H9rCsvpjv3T21QY3f/q0X2dszzCdfv4jX5/D5cKaFYgl+t/Eo33n2IO2DYepKPPzkvRdSe9w4pNmrJxDmC3/aS1WRiw9fvyDT4Yg89ML+Hu789iso4MmPJMfeW7+vizu/twFDKTZ89HJ8vpOP1SvOjj/t6OL76w5y7XnV/OVls9NaZiYSPpvWeiD1+nbgW1rrXwO/Vkptms4NfWbtTgLhOBsPDXLtoho8zvRGm88m0YTJ7zd3YFqajqEw77igKdMhTcnt33gOfyQBwKce3sXdl86hdzjK4zu7Afjbn23k8Y9cmcEIxYd/uYWEBUMRkzd/7Znj3vvOugM8s6eX/mCUMq+LRXVFDEcSvHllPYFInLVbO9l4eIhAJI7HYaO1qZQyr/OEwdqPt7cnyDN7+gAwDMWFs8v50uP7+PPObmIJi+7hKOG4xarGEp7Z0zu2XMy02Hh4CIDSAifP7uulJxBl21E/91wxG5c99/Zzkb/84Ti/39QBgN0wuPuSlkmv45MPbSea0Dy5p5dHtnVw45K6SS3/oxcO8sLBfrSGf/z1Vkn4TuOZPb18+fF9dAeiABzsD/F3P9vIL/764gxHlp5//+MuXjmYvMSaV+XL6ZvdIju949uvAMlnsa770nMUA2///qvEreTUy7/wHK9+4oYMRig++eB2gpE4OzsDvGVVeueLmWjSaVNKjSaS1wBPjHtvWhNMjyN54ee028jBXA8Am1K47MliKMjVDwE0lB678DdU8v8Om0Kp5B8+10zcWxCTYUuVBUBdScFx7/ncdgxDYRgKp11htyncqf3LYRg4bAZ2m8JpU9iM5L/R3+2pjO6f419XFjrHprkcNgqctrHtAHictrF5lQK3wxi3nxvHfQYhsoHNOPabLPE4TzPnqTlsqd88UFPknvTyNcUeRqNw2GQfOR23w4bLdvyxq9STO+en0VYXSilKfVP7vQlxOrZxl6IFqRus42+0lhXI7y7T3KnrL5th4Lald/yaiaPcT4GnlVJ9QBh4FkApNRfwT+eG/ucdq/j9pg6uWFCJzZabyZLdZnDH6kY6/GFmV+Zuc7X/un0VPcMvsKtrmN//TfJOaUmBkw/dtJCdnQE+8bpFGY5QvPrPV3DjV1+kpcLL/7yzldYvHXvvnktnM7fSh81QlHmdeJw25lYmm2x4nDbuXNPIJXPLUQpMC+pLPBM2P24oK+C21llEExZzUut698WzsSmD9qEwb1xRz5xKH067wdtWNxCJm8yp9KG1ptznxOuyU1Xk5qaltRzsG6Gu2IPdNhP3qISYOp/Lzkeun084ZnLFgqoprePhD17Efb/ZzrXnVbGisWzSy1+7qIYPXzefdXv7+K+3LZlSDOeKy+ZV8tU7V/LDl9rYcsTP4rpiPvfWpZkOK22ffP15zK300lBWwKVzKzMdjshDez9zM8s/+QhOu8Er/3I9rWs/ztb7b+CyBx7H57bxx3uvzHSI57zv372aH71wiOuX1uB2p5fKKa31xHNNklLqQqAW+JPWeiQ1bT7gm65hGVpbW/X69eunY1VihrS2tiJllL2kfLLfaBk137c2rfnbHrh5hiMSJ5L9KLtJ+WQ/KaPsJuWT/ZRSG7TWraebZ0baMWitXzzJtD0zsS0hhBBCCCGEECcn7aOEEEIIIYQQIk/lzpPKQgghTkuafgohhBDiRFLDJ4QQQgghhBB5ShI+IYQQQgghhMhTkvAJIYQQQgghRJ7KqoRPKXWBUup5pdSzSqkvZjoeIYQQQgghhMhl2dZpyyHgaq11RCn1Y6XUUq311lPN/MK+Xr7xzEFuWVHLm1c1nMUwp9dnHt7OI9u7+acbF/D65fWZDmfKLvrMY/SOxPjOO1dw5aLk57jgM48xMBLjx+9dw5oWGSQ2kzYd6efOb75EVaGLp//pmrHpT+3uYX3bAHeuaaS+tIBOf5j9vUHshsH86kLKvE4AhiNx9vYEaS73jk07UZc/Qqc/TEt5AY/v6qG22MPFcys40BvklxuOcuWCSkaiJk/u7GIonOCa86oo97loKC0gFEuwtd3PdefVUOZLrr8nEKF9KMzCmiI8TtvYdkKxBLu6hplV6qGq0D02bXfXMPWpab3DUY4MhlhQXYjXdfyhzrI0OzoDeJw25lT6CMdMdnUFqC/xUFXkntbvXeS/Rf/yR+KWxSv/cAklJSWTXv6Vg718/Hc7uHJBJR+9adGUYvj5y4dZf3iQD187n9oSz6SXP3GfyEePbOvgj1u7CEYSbO/001Lu5ZZVs2jrG6Gi0Mkbl9dzZDBMTbGb2mIPpmny/ecPYSjFX1zawuH+EEPhGItqi7DbXnu/PBI32dkZoLbYQ03xzBxHhiNx/riti3lVPlY2ls7INjLhS4/t5pm9ffzbmxezqK4k0+FM2S/XH6FzKMJ7L5993Dkrl/T397P6cy9iKNj3H8kOvvx+P2s+9xxOw2Drp27McITi84/u4FvPtHHRnHL+9y8uSGuZrEr4tNZd4/5MAObp5n//j18lFDN5+WA/1y+soaDAMbMBzoB9XUN8Z10bGvi7n23K2YTvrV9/js7hGADv+cEm2h6oZ19PkJLUtLd982XpGTDDbv3vF7GAQ4MR3vr1dQAc7A1y/0PbiSUsNh4e4ht3nc+v1h/l1cODeBw2ljeUcM/ls1FK8eCmDnqHo7ziHBibNl44ZvKrDUeIm5rv9QXpCURRCoo8Dv7ld9voGY7w21fbqfDa2dk9gkLz1O4els0qYVFdIZuPBFAKthz185lblhJNmPxyw1FiCYuDfSPcumrW2LbWbunk6GAYp93gnstn47AZ/GFrF0cGQjjtBu++qJlfrD9CLGGxvyfIba3H3xDacHiQdXv7AHjLqllsODxAW19y2b+8tAW3IzdP1OLs294RoDpuAbDqgec4MIXj3Lu/t4FQ3GRvT5Ar5lVy8bzJ3RzbfHiQz/xhJ1prdnUGeOhDl006hlfaBnh+fz8Abz1/Fg1lBZNeRzbb0zXMv/xuOwMjMUydnNYViLH+0CBOu40Cp43n9w+wuK4Yu6G4+9IW/u/5g/zkpcMABCJxALSGgZEYVy6oes02HtvRzb6eIHZD8ZeXtVDgnP5LrK88vpctR/3YDMXnb1s+peQ+24xEE3z1yf1orXnHd15m4yeuz3RIU/LIti6+9OfkkNPdw2E+c8uyDEc0Ned/LjmUtqVh3j+vpRhY9cA6TA1RTJbf/yibP3lDZoM8x331yYMAPLWnj90dw2ktk1VNOkcppZYBFVrrHSdMv0cptV4ptb63txcrddDWGYhxusROm9LmjljitR9E61wumfwWjiYvUE1LM1pMptZorbF08qLGSv0bZaVmTM732nVaqWUBEqkrKq3BtKyxZbXW6OSm0antaAANZuqNhHVsWSv12rSO3+Bx60u9ZVnHpoEe+/1ZJwl2/PpMrTGtY+uQn62YnDP/wVipdWggblmTXj46bpkT95V0meN++FNdRzZLaIvTlZXm2HEreQzUxBPHvtd4wjx2rDnFQWLsuAQzdhwxxx0fE3lysBr/KU713eaC+LjrIDNPru1OdiiwzNwto3wUNmNpzZdVNXwASqky4GvA2058T2v9LeBbAK2trfrf37qMbz5zgDeuqMvJ2j2ARfUl3Lqijmf29fHha+dlOpwp+/2HLmfZJx8hGDX53FuXADCvupCoy8ZI1OS/71ia4QjFd9+1knt+tJFCl521915O649gbnUhH7luARsODfDOixop8ji5ZWU9i+uKsBmKxXVFYzV5b1hWx66uYWZXejEM9Zr1e112bllZT/tQmHesaeSRHV3UFLlZ3lDKf9yylJ++cpirF1YxHE3w2PYuhiMJrl5YRXWxh5YKLzcsqWXLUT83L6sFwO2wccuqeo4OhllcV3Tctl63tJYdHQEaywpw2pP3rW5cWsOOjgANZQX43A5uXTWLwwMhFp2wLEBrUykOm4HHYaOlwktloYtt7X5mlXpythmOyIzFdcX4DTAt+M1fXzKldXztzpXc/9AOVjeXcsWC6kkvv6a5nA9dPZcNhwb5hxsWTimGNc1luFI1Xc0V3imtI5stqi3mH29cyB+3dTE0EmN/X5D6kgJuXVVPW1+IikInbz2/gfbBMLXFHgrdDj5wxVwsDYaCD109l4P9IQZDcZbPKjnpNq49r5qtRX7qSzyvaUY+XT541Twe3NzOgupCGkrzoxbW57Jz8wUNvHhggH95/dSaNGeDN6yop3s4Ss9wlL+5ak6mw5myP917Idd/6UUU8PLfX8gNj8GTH7mUqz6/DruhpElnFri9tZ5fbWhnWX0RKxrK01pGZVMtjFLKDvweuF9r/dLp5m1tbdXr168/O4GJKWltbUXKKHtJ+WS/0TJKd0D1dEnz6ukj+1F2k/LJflJG2U3KJ/sppTZorVtPN0+2Nem8DVgNfFYp9ZRS6qJMBySEEEIIIYQQuSqrmnRqrX8K/DTTcQghhBBCCCFEPsi2Gj4hhBBCCCGEENNEEj4hhBBCCCGEyFOS8AkhhBBCCCFEnpKETwghhBBCCCHylCR8QgghhBBCCJGnsqqXzsm65F/X0h5Jvs7lcaUW3LeWKHBJs5sf//U1mQ5nykbHCvv8anjLW24+btpHW+B978vdMsoXo+Uxfn9Zf7CftVs6qC3xEI6ZJCyL6mIPrU2lNJX78Dht+MNxDvWP0DscZXlDMZYFA6EYTptBbaGDf31oBw67wQUtFZja4kBviL3dAQxDUe5zMqfSR02Rh12dw2xrHyAYM2ko9bFubw9N5V4cdugNxFAKCt02ls4qo2MwxHAkwcL6IrYe8VNe4MDrcnB0cISuQJTXL66hyOfi8MAIPYEoNSUePnjVbB7a3IW2oLbMg2la7OkeZtmsEgrdDnZ3BdjR6afQ5aClwktzpRe04tDACDctreV3G49S7nPx+mX1r/nujgyMsG5fHzcuqqXU56Q/GKV3OIrDbtBQemwA+PGODoZw2gyqitwzV6gi4062X03Gx3/9Aj96ZeCM1nHHN9ax6Yif33/oQubXpDcQ74ke2tROhc/FRXMrprR8lz/Mk7t7uGZhFVVFnimtY92+XoZCsZPug+mwLE1b/wjlPhfFHsdx7932jXW80uZ/zTIrZhVS6Lbz5uW1fO7RvUQSFm9eWYfTbqOq0MmDmzoJReMsri8hHDf555sWUlVUwI9eaKN9KExtsYsyr5u3tjbw4v5+6ks9FHkcPL+vl4SledOKetr6Q5QWOCgpcKb1OdqHwtgNRfVJjh2H+0N4XTbKfa4pfUfZaHQf+quLqvj4m1ZnOJqp+9ufvkpbb5Df/+3lmQ7ljJzsmHamxzkxfcaPzZtueWTVwOuT0draqvuuvX/s729eZeeGG27IYERT8zc/fJa12wNjf+fqjnTiwNBtD9yMq3Yete/+0nHTROacWEYVf/4kv330aW76yjqC0QSjRwIFOGyKpvIC3nFBE3esaeT7z7Xx8JYO4gmL2hI3S+qK2HTUT1OZl2d293DUn7zz4jBAAwnr+G0bCmwK4idMn24ehwIU0YSF22EQT2gsNHalKPbY6Q3Gxz6nTUGlz0kobuG0G9gMRSiaQCnFv7z+PG5rbRxbbyxm8ob/Tn5PTeVevvb2VfzwhTY2HBqkqsjNxXPKuXXVrONi2dER4NHtXSgFb1k1i4aygkl/Hhl4PftNx3FufPk2l8BT901uHf/864385JWOM4rhC3/aza9fPYpSis+8eQlXLKia9Dre+NV19I9EqfS5+d0HL5n08o/v7OKTv9+B1prbzp/FvdctmMI6utly1I/LYXD3xS1cdvEFrF+/nrd94zlebhua9PpOxue0sbyhmBcPDGDq5DGzwGljSX0RoFAkjy1P7+3DaTM4v7mUhTVFOO0G77qoiUK347Tr39kZ4JFtJz92rG8b4Nm9fdgMxdsvaKQiD5I+X/08Ku760tjfuXp8et8PX+bR7b0AFLpsbL3/xgxHNDUnu1YYf70NuVtG+WJ8GV1RDj/4x9fn3MDrU/Zgx8TzZKONhwITzyTEDOkfiRE3j8/CNGBpTShmMhxJEImZROJmqvZPMxSKEzU1CVMTTZgEoomxZU0N1kmSOq2T7820eEKjU9tLmBpL69S2NXFLMz4ErSFuWsRNC601I6nPobXmYN/IcesNmyahmAnAYCjGSDRBwtTETIto3MQfjr8mlkAkPrad4UjiNe8LcTJTyUk2H35trdVkdaRu2mitOTwQmtI6AtHkb94fee3+kI72wTCjN6G7/NEprWN0X4zGLaIJc2z6of6pfaaTiZomvcMxrNQBRaf+9QSSMUcSFn3BGFonj0Gj02MJi3DcPPlKT/IZtD52HBk1+rdpHTtm5bqZvhF4tuzrOnbeSKechZgOT/enN19ON+kc73/uzr3aPYDnP37ztN+9z4S2B177OZbWF9OXoXjE6bU9cDOtrZ9keUMJb15Zz7N7e/E6bUTiFhqYVermivlVXL+4hhKvk+sWVeN2GHT5o1y1sAKnzUZDqYdyn4vrzqviY7/dimnBiqZSzIRJW3+I/pEIljYodNuoK/ZQ5nVxsDdAuz+GRmMAsdSJ3qaOJYQGUOhSRM1k8yyPHUIJsBkKu2Ewkkq8SjwOqnw2uoMJonGLEq+dt69p5Jk9vQxHTGZXehmJJegYitBS4aWuxMP6tkG6/GHsNkVtsZtls0pJWBad/givW1zNrzd2UOiy87dXzjvu+yr2OPnrK+bw9J5e7lzTQENZAVcsqKSqyEWRx8GalrLXfMcrG5PNv1w2g4U1hTNXmCKjCtT0rm8qd87X/r8rablvLRo4f9bUfmt/f/18/v0PJiUFTt6+pmFK6/h/18xn7dZO3rCsdkrLv+OCRvb2BAlEEtx73byJFziJqxZU8UrbAHUlnuOaTz7yd5ez8t/+fMrlnDao8do5HEgmUV47lBa6UVpzdCiKBhyAw2Hwl5e2cPm8Sv7lwW0MBKP43HZqSwv45xvm84cdPTSVeZldWcA3nz6Azab4myvn0hWIUlnooqpw4ubdKxtLiMRNHDaDhTVFx7134exytIYij4Omcu+UvqNsky/XCr//wBpa/+MZYqbFA7cszXQ4M+aNszMdgRiv7YGbUZ+deL6cbtK5fv36TIchTmO0OZrITlI+2U+adGY/2Y+ym5RP9pMyym5SPtlPKXXuNOkUQgghhBBCCHE8SfiEEEIIIYQQIk9JwieEEEIIIYQQeUoSPiGEEEIIIYTIU5LwCSGEEEIIIUSekoRPCCGEEEIIIfJUViV8SqkvKqWeVUp9OdOxCCGEEEIIIUSuy5qB15VSqwCv1voypdTXlVKrtdavnG6Z6+9by57U61weV2p0jK2/LoH77sv9zzG+LE42TWTOycqjYzDEgb4g1UUeqgpdJCzNwb4Rmiu8+MMx3HYbbocNy9IoQ4HWKKVw2A0K7DYO9I+wpyuAz2XQOTjC07t6aakp4fk93UTicZx2GyVug/Zhi5GRMMEoxDS4AMMFjaU2tnWZY/FcNqeMA92DKMOg0ufCbkCpx46vwE0wahKOJQhETT5ydTM7u+OEEwn6RyLs7vBz5bwyfrGxm7lVPu68oIlDPQHahyLMry2m0OOgczBMU3kBs8o8bDjsJxJNUOhxMrvSy8rGEjYcGUJZkLAs/OEEzeVuDvaH8Lkc1JV6KPY4OdgzzEjMpNznorbEDSgO9gxTU5IcXN7rOnZYHQrFaB8I4XbZmFOZHBA7EIljU+q4+UaZlmYwFKOswIlhTPOI3uOkO66f7LfpmY7j3Nz71nLzeRV8+d0XTGn5j/1qIw9v7WTz/TdNOQZ/OI7DpihwTu3SYPT3W1rgxDbF329fMEI4btFQWjCl5U/lzm8+xwsHh8b+nlOq2D+YHId4caWNm5a3cMGccv60o4u23iBr5pRz8exKjg6EeHZ/D4k4XLGwihWNZRR7HNhtio6hMAPBKIvqi3HZbQBsPTKEpTQLqovoGQ6zt3uEi+dUEImbeJzJY+mpPL+vl4YyD4VuJ4cHQhS57TRX+AA41D9Cscdx3GDyuWowGGN/7zCtLeVj0/LlWmHxfWsZIfc/h1zPZbdl960lkHqdbnlkzcDrSqm/AXq11r9QSr0FqNNaf/VU87e2tuq+a+8f+ztXf4C3f3otL40c+ztXP8eJF5BtD9yMq3Yete/+0nHTROacWEYVf/4k3/r1Y3zoJ6/SPxLDYVcsqy+hyx9hIBTD67RhKIXNUFQVu3HbDZRSmKaFUool9cX0DUd5bEc3g+F4hj7V1BiANe5vl00xq6yA7kCEYNRk9FLVZoCpQWso9zpw2230j0SJmRqHoZhf7WMkZtIXjOG0G9ze2sC7L2mmqtDNgd4gn1m7gw2Hhih027n32nmsairj95s6sBlwW2sD1UXu4+L6xfojtA+GmVft4/XL6mZs4PV0yT47sek4zo0v37tW1vLp21dNavl7f/wKv9vac0Yx7Oke5g9bO3HYDO5c00iZd/KJxW9ePcqh/hAtFV7evLJ+0svv6x7m/od3EDct/urSFq5dVDPpdZyotbWV8dcKZ6q60MVdFzWRsDQ/e/kIsYTJ6pYyvnzHSn796hH+84+70cDyWcVs7fATi1tUFbl504p6fC4777ywCY/ztUnfP/1qC3/Y2oFSioU1PnZ3j1DktvMfty6lLxjjpy8fxu2w8e+3LqG+ZHqT4bPJH45xx7deJBCOc9GcCv7rtuV5c61wsuugXHSya4UT96Fc/Wz5YnwZFQFbP/v6CQdez5oaPqAE2J967QcWnziDUuoe4B6AxsZGxt8/bL5vbU7+AMcne0KcbQf7RwjGEpiWxorDQDCCPxxP3amPU+CyY0PjD8UJGQqSFXwUOG2EYib7eoOMxBKZ/hiTZp3wd9zUdPrDWKn7X6O3wRLjZgzFTMJxi7ipsTSYWtMzHCWasEhYGh03GRiJ0hOIUlXoptMfoWc4imlZROImOzuHqS32YGmNZUJPIHpcwmdZms6hCADtg+GZ+/CTMN01gVKzOLHfbuvk07dPbplHd/ZMPNME2ofCaA2xhEXvcHRKCV/HUHhsXVOxpztILLXT7eoanpaEb7oFInHCMZODfSNEEyaWpekYCjMcSfDKwcHk/q3h0ECIcMxCAX3BGJZlEowma1FPlvBt7/ADye//0EAIy7KIJSxePTREOJ5sARGJmxzoHcnphK99KEwgdYNwX08ww9EIkfsCE88CZFfCN0QyUSX1/6ETZ9Bafwv4FqRq+Ma9l6sXCG0P3Jyxu/fi3DV6s+TK+ZU8NqeCHe1+yrxOrlxYRbc/zNb2AHOrvASjJk67QXOZB5vNhkGyxsvSmnlVPlY3l/CDFw6x9egQKEUodmIqlTk2wBz3t8NQJCxNocugyOOgwx/F0mBT0FBawE1La/hTqrbSspJJXHmBk75gDMNQrGgopsTjZFuHn/6RZLO1a8+rxh+Os73DT2mBk8vnVzG/Otl0c3lDCTctqeUP2zqpLnJzx+oGaks89AajOGwG82t8x8VrGIqrF1axszPAisaSs/Y9ieyy7dOTP5ft/Ldj55GpntRXNZYyFIpR4LQzp9I7pXVctbCK7e0Bls4qntLyVy6sZPPRIUaiCW5ZNfkawlNpLHVzeDAy4XxOA8YfwuqKXfQNR8emue2KK+dXsqyhhMvmlROKJegfiXF7ayOVhS7uvWYee7uHMbXmzjUNrN3SRac/wjULq2iqKKSy0EV1keuk2773uvn8x9qdlBQ4uHpBBWu3dVNT7OauC5sYCsfwh2OUeV1cNK4ZZC5aVFvMNedVs6MzwHsubgJgaX0xfRMslwvuOL+En20YynQY4hzT9sDNqM9OPF82NelcBfwXyZvrC4F7tNanzIRaW1v1+vXrz1Z4YgpGm6OJ7CTlk/0y3aQzXdNdwzfd203XVGogZT/KblI+2U/KKLtJ+WQ/pdSETTqzKeGrB/4EDACbtdYfPN38FRUVurm5+WyEJtKUsDSmpXHZk52/trW1IWWUXWIJC8NQ2A2V1+WTsDSWpXHas6oj4kmbahnFEhaGUthtM9fxi0jK5/0oH0ymfKIJC1vq+CjOHtmHsk/MTDZHdtiMsfKJJZLP7zvkvJJ1NmzYoLXWp73gyaYmnTcArwANgFZK2bTW5qlmbm5uljsOWcQfivODF9pIWJoLWsq4eG6F3BXKMi8fHOC5fX3YDMU7Lmjkhisvycvy6QlE+OnLR7C05ooFlaxqLM10SFM2lX1o85EhntjVg1Jw++oGaos9MxSdALn7ne3SLZ8X9vfz4oF+bIbirgubKJ3CM4xiamQfyi67ugL8cWsXALeuquctN1zBD37/BI9u70IpeMuqWTSU5e5zpPlIKfXqRPNk0+3vasCptb4GCAFvOnEGpdQ9Sqn1Sqn1vb29Zz1AcWqheIJEqseLQCT3OvE4FwxHkg/Km5YmFDvlvZScF4wmsFItFwI51nvodBhO7X9aw0hU9kUh0jH++JiLHVEJMV2Gx13Djb4OpPYPrY9/X+SObKrh8wNPp14/AbymLeqJnbacvdDERGqLPVy5oJKBkRgXzM7th8rz1UVzylEKij2OvL4711Lh5ZK5FYxEE1x4Dv4WW5tLiVsWBQ4bcyp9Ey8ghOCSuRXYbYpij5NZ0zwGoBC5ZEVDCeGYid1QnFeb7EtxVWMp0YSFw6ZYWFOY4QjFVGRTwvc88N7U6xXAwcyFIqZiZQ43nTsXFDjtXL2wOtNhzDilFGtayjIdRsa4HTauWlCV6TCEyCle17lxfBRiIg6bweXzK4+b5rQbXHHCNJFbsibh01pvUkqFlVJPAX3AFzMckhBCCJEWGWNQCCFEtsqahA9Aa/33mY5BCCGEEEIIIfJFNnXaIoQQQgghhBBiGknCJ4QQQgghhBB5ShI+IYQQQgghhMhTkvAJIYQQQgghRJ6ShE8IIYQQQggh8pQkfEIIIYQQQgiRpyThE0IIIYQQQog8JQmfEEIIIYQQQuQpSfiEEEIIIYQQIk9JwiemTftAiBf29R43LZowGY7EMxSRONHmw4N0DoUzHca0CUYTHBkIcqA3CIA/FCdhWhmOanrFEhZH+kNYVvqf60BvkJ5A/pSzEDMhYVpsaBugLxDFH4qjtc50SCLHReImwWgi02GcsT1dAfb3BI+btqPDT1tf8BRLiLNJa40/FMe00j9m2WcwHnEO2XxkkDu//RIJ0+LKBZV8612rsbTm/55vYyRqct2iapbUF2c6zHPa/b/fxq9fbcduU3z/3aszHc4Z29M9zPfXHeCR7d14HDauXlhFuc9FVZGLO1c3Yhgq0yFOi4/9biuH+0OsaCjhozedN+H8P37xEF9/ej92m+JLb1vBisbSsxClELnFtDTv/cF6XmkbwGYY3LqynkvmVnDtoupMhyZylD8U5ycvHyaaMHndkloW1BRmOqQp+f66A3zxz3sxFHz6zUsA+PYz+/nqE/swlOI/blnK65bVZjjKc9ufd/awrd1PXYmbt7U2pLWM1PCJafHk7u6xmpWdncMAJEzNSNQE4MhAKGOxiaRNR/xorYknLJ7b35/pcM7YkYEQHf4wCdMiYVlsOToEQE8gSiRhZja4aaL1sX1nX096d1bXHxocK+f1hwZnMjwhclYolhhrGRCJJ+gejnBYzlPiDPQMR4jETbSGo4O5+1t6fn8/WmtMS7Nub98J0yzW7evLcIRi9FjVMRQhbqZXyycJn5gW91w8h4ayAnxuO++/cjYATrvB4roi6ks8tDaXZThC8cGr5lDuczG70se7LmrOdDhnbFVjKdecV0NTuZfGMi8fuGou1UVuLppTToEzPxovKAVvWlFPbbGHt7bOSmuZ917aQn1pAQtqirgtzWWEONcUuh28/YJGyrxOFtUWc/Hsci6bV5HpsEQOa6nwsrCmkPpSD6tyuGXF3107j+oiN3WlBXzw6jkA3HvNPKoK3dSXFvDBq+dmOEJx+bwKqovcXDavAqc9vVRO5Wqb9dbWVr1+/fpMhyFOo7W1FSmj7CXlk/2kjLLfaBk137c2rfnbHrh5hiMS48k+lP2kjLKblE/2U0pt0Fq3nm4eqeETQgghhBBCiDwlCZ8QQgghhBBC5ClJ+IQQQgghhBAiT0nCJ4QQQgghhBB5ShI+IYQQQgghhMhTkvAJIYQQQgghRJ6ShE8IIYQQQggh8pQkfEIIIYQQQgiRpyThE0IIIYQQQog8JQmfEEIIIYQQQuSprEv4lFIfVkqty3QcQgghhBBCCJHrsirhU0q5gOWZjkMIIYQQQggh8oE90wGc4K+A/wM+lc7MLx8c4OWD/SysKeLaRdUzG5k4rcFgjPf+cD39IzE+cOUcbmttwNKaD/98E/0jMd53+WwunluR6TDPaS8d6OfrT++nxOPk/jcuynQ4J7Xh0CAvHuhnTqWPG5fUpLWM1ppHtnVxoG+Ei+eUs7KxdOy9zUeGWLevj5YKL69bUoNSalrjfWxHN7u7Alwwu5zVzWXTuu5Rj+/sZmdngNXNZVwwu3zC+Z/c1c1n1u7E7bDxtbevpLnCNyNxCZGrDvaN8OtXj/Dsnj5KChzce+38444b2ezIQIi1Wzspcju4dVU9boct0yGJPPOnHV187DdbMQzFV9++EoCHt7Tzr7/fgd1Q/M87VrGqaWbOd2LmZE0Nn1LKAVyhtX4i3WU2Hxkibmq2tvtJmNYMRicmsm5fL53+MLGEydqtnQCEYibtQ2EicZMndvdkOELxxK4ewjGTTn+YTUeGMh3OSW0+MkQsYbGzM0Akbqa1TChmsqtrmFjCYstR/3HvbTmaXN/urmFGYumtL11x02Jbu5+4qdk8g9/nlqOpbRxNbxsPbu4gHDcZDMX40/buGYtLiFy1rd3Pwd4R+oJRhkJxnsqh89OOzgDhmEl3IEL7UDjT4Yg89POXDxOJm4SiCX65/igAv97QTjRuMhJN8Mv1RzIcoZiKrEn4gLuAn5xuBqXUPUqp9Uqp9b29vSypL8ZmKBbXFWG3ZdNHOfdcOreSykI3dpvBDYuTta0FThu1xR6cdoMr5ldmOEJx1YIqXA6D6iI3yxtKMh3OSS2bVYzdUCysKUz7znWB08a8ah92Q7F0VvFx7y2pT65vbpUPr3N674Q7bAbn1RYlt1tfPPECU7S4rgiboVhaX5LW/DcvqcVlt1HscXD9Ymn5IMSJFtUV0VjmpczrpMjj4PL5VZkOKW3n1RThchhUFrqoL/FkOhyRh966qgGn3YbHaeetK2cBcOuq+rFpt5w/K8MRiqlQWutMxwCAUuqzwApAAxcAn9Baf/VU87e2tur169efpejEVLS2tiJllL2kfLKflFH2Gy2j5vvWpjV/2wM3z3BEYjzZh7KflFF2k/LJfkqpDVrr1tPNkzXP8Gmt/2n0tVJq3emSPSGEEEIIIYQQE8vKdpBa60szHYMQQgghhBBC5LqsTPiEEEIIIYQQQpy5GWvSmRpT7y1A8/jtaK3TGnJBCCGEEEIIIcSZmcln+B4E/MAGIDqD2xFCCCGEEEIIcRIzmfDN0lrfOIPrF0IIIYQQQghxGjP5DN/zSqmlM7h+IYQQQgghhBCnMe01fEqprSTH0rMDdyulDpBs0qkArbVeNt3bFEIIIYQQQgjxWjPRpPP1M7BOIYQQQgghhBCTNO1NOrXWh7TWh4BaYGDc3wNAzXRvTwghhBBCCCHEyc3kM3xfB4Lj/h5JTRNCCCGEEEIIcRbMZMKntNZ69A+ttcXM9goqhBBCCCGEEGKcmUzADiil/pZjtXofAA7M4PZy1g9eaGPT4SHetKKOKxZUZTqcKYnFTD7yq810BiJ85Nr5XDS3AoA/7+hmMBTjqoVVVPhcGY7y3PbUrh7uf2g7pV4nP/iLC2ZkG8/v6+PIYIiL51TQUFZwyvm+++wBdnQGuO38Bi6cUz4jsZzM+rYB9vUEWd1SxpxK33Hv+cNxHt/Zjcdh49pF1ThsM3k/LH0bDw+yu2uYVU2lzK8unHD+HZ1+Pv3QDrwuO59/23KKPc4Jl3nxQD+H+ke4cHY5TeXe6QhbiKz1f8+18cj2ThbXFVPmdbKvJ8gtq+q5bF5lpkM7Zz23r4+jaZw7xMzb0eHn7362CbtN8a27VgHQHYjwlcf34rIb/L/r5lPodmQ4ynPbD19o4zevtnNBSxn33XReWsvM5BXNXwMXA+3AUeAC4J4Z3F5OGgrFWLulk/ahMD9ffyTT4UzZn3Z18erhQTqHwnx73UEAYgmLre1+jg6GeeXgQIYjFF9+fC99wSh7u4f5wQtt075+fyjOSwcH6BiK8Ny+vlPO1z4U4k87ujk6GOZnr5y933wkbvLs3j46/RGe3dP7mvc3HRniUH+IXV3D7OsJnmQNmfH0nl46/RGe3v3amE/m288c5PBAiJ2dAX65/uiE8w9H4rywv5+OoQjP7j11uQmRD/pHovxm41EOD4T4885ufruxnfahMD97+XCmQztnmZbm5dS54/n9cgzKtC8+tocuf5ijAyG+9sR+AH6/qZ19PUG2dwT4846eDEcofvTiIXqGIzy0pYPBYCytZWYk4VNK2YAvaK3v0FpXaa2rtdZv11rLr+QEPqedmmI3AHNPqHHIJUvqivE4bAAsqisCwG5TFDiT0+pLPRmLTSQtnVWEUgqHzeDC2WXTvv4Cl43SguRdv1mlp75DW17gpMKXrHWaV332fvNOm0FVUbKW+WTx1Zd4UAqcdoOqwuypja4rTu47s9Lch1Y1loyV86rG0gnn9zhsY+WR7jaEyFXFbgf1pR4MpagsdFFXkvzNz63K3fNvrjMMlda5Q5wda1rKUEphMxQXpVrgLKwtwlAKh02xoEb2lUxrqUiWQWWhmyKPLa1l1LjH7KaVUupR4A1a6/RSz0lqbW3V69evn4lVn3WRWIJOf4Sm8gIMIzuakU1FfzBKtz/CovpiAFpbW1n3wktEExbFHqn+zwYvt/VTW+SiocxHa2sr070PxU2LYCRBqff0zQhDsQTd/ggtZ/kmR8K0GI4kKClwoJR6zfvDkTh2w8DjTO8AOtNaW1t56eVX8IfjlJ4i5pPZ3RXA53KkfaMl3XITrzW6HzXftzat+dseuHmGIxLjnew4F4mbtPUFqSn24LYbeXH+zWWtra288NLLcgzKItvah3DaDObXFI3tQ51DYRx2RYXPnenwznmmabL5aID5VV58HidKqQ1a69bTLTOTz/C1Ac8ppX5PsodOALTWX5jBbeYkt9N+1i98Z0K5z0X5Cc/puR023I7suHgWsKZ5Zp+Xc9iMtE7YBRn6zdsniC8bn0uwGYqySV4ELagpmtT86ZabEPnA7bCxsLZ47O98OP/mOjkGZZcl9SWvmVZbIi1AsoXNZmNV08QteMabyYSvI/XPACbuaUAIIYQQQgghxLSasYRPa33/TK1bCCGEEEIIIcTEZizhU0pVAv8ILAbGGvxqra+eqW0KIYQQQgghhDhmJp9Q/jGwC2gB7if5TN8rM7g9IYQQQgghhBDjzGTCV661/i4Q11o/rbX+C+DCGdyeEEIIIYQQQohxZrLTlnjq/51KqZtJduAyawa3J4QQQgghhBBinJlM+P5NKVUMfAT4KlAE/L8Z3J4QQgghhBBCiHGmPeFTSrmBvwbmAvXAd7XWV033doQQQgghhBBCnN5MPMP3f0ArsBV4HfD5GdiGEEIIIYQQQogJzESTzkVa66UASqnvAi/PwDaEEEIIIYQQQkxgJmr4RjtrQWudmMyCSqkLlFLPK6WeVUp9cfpDE0IIIYQQQohzx0zU8C1XSgVSrxXgSf2tAK21LjrNsoeAq7XWEaXUj5VSS7XWW0818/mfepT+UAKnDfZ85ubp+wRn2ZJPPkIwarKkzsfDf3tFpsOZsub71gJQ7oENn7z5uGkr6nz8Loc/Wz5Y/am19IaSr9seOLa/fPqh7Ty4uYMil536Ug/huEUwEmdulY/zaosoLnBiGPD8nj5eOthPVZGLkgIn9aUFfPTG8+gNRvjk77dTV+Lhnsvn8Py+PnZ3Bjg8GMJQiovnVbCmqYw/bu+kzx/i5UN+RmIJKrwOFtaUMLfSzS9e7cTS8IW3LSMS17RUFDAQihOOmZR6HTy6rZvW5lKuXFDFfz26i8d2dPOG5bW8/8p52AxFwrR4bEc3+3uDXLeomgU1Jz/M7O8NcrB3hOUNJVQWuib8zqIJk5cODOB22FjdXIpSalrKYjJG9yE7sO+BiY9z331qN59+ZB8AG/7hQsrLyydc5r8e3cXLBwd43+WzuWZRzYTzP727mw//cjOVPjeP3Hv5hPMDXP6fT9ATiPJPN8zn7svmTDj/k7u7+Y8/7OK8mkK+fOeqtLYx06LRKNd/5TnCMYvvv3s1i2cVZzokMQk/fekQ33vuAIf6QsSsY9MdCmZXeen0h2mu8PGDv7iAlw8OcN+vNxNNWLxl5SxKfS6uW1zNkrpjZf7igX6e2t3DhrZBfG47n7ttGRU+99j7n31kF53+MB+4Yi5Hh0JUFbpZUn/y30w4ZvLiwX5KPA5WNpae8jNsaBvgY7/bSkNpAd9+9+oz/1KyxOh10BuWVPPVd7ZmOpwp23BogOFIggtnl+N22DIdzpS84Qtr2dqTfP23VxcA8Dfff4G1uwcA+NI75vDmpQszFZ7g2HUBHH89dzrTnvBpraf8C9dad437MwGYp5u/P5SsQIyZsG5HL5cuqpzqpjPmf9ftIxhNfsxtHcEMRzN14398/eHk/7e2+6lNTduUw58tX4wme5AsrwrAH47zgxcOEbc0fcEYB/pDjKY0+3tHeOXQIPUlHiyt2XzUj9bQOxLHYUBxd5ACp41t7QF2dwXY2u5nIBjjyGCITn+EaMJEAZ3+CA9t6iAQidPlj2Dq5PoDEZPDg908vhuiieTED/z4Ve6+ZDbP7u2l0O3AZihePtiPpeHVw4NU+Zx8/7k24qbFt545yGXzqljeUMK2jgC/3HCUkWiC9qEwn3zD4tecbKMJk7VbOjEtTVcgwjsvbJrwO9twaJANhwYBKClwML+68AxLYXL8I7Gx1+k2lxhN9gAu/PyL7P33058M9nQF+MELh9Bac//DO9JK+P72p5vwRxL0B+N87Ddb+cytS087/2ce2sbhgeSB4TN/2JVWwvfRX29lYCRGW98I12/p4OZldRMuM9Pu/r+NHEod4N77w/U8/9FrMhyRSNfWo0N8e91BDow/EKbENezuHgFgy9EA/+9nG9nVNcxA6hrjhy8dZk1LGW19I/znbctw2W30BaN8/7mDbDw0SP9IDLfDxqce2slX7lwJwEOb2vn9pnYAPtq/hUvnVQJ+qovcJ73Z9Ny+Pra2+wGo8LloKCs46ef4+19tpn0wzIHeEb7zzH7+6vKJ96Vs1z4YxpW6DnpoWzdfzXA8U3Wwb4Rn9vSN/X3lgqoMRjN1o8kewFeeCFEBY8kewL0/3s+bH5CEL1t89vdb0ppvJgdenzKl1DKgQmu944Tp9yil1iul1vf29h63zILcy/UAWFB7ugrP3HGyLN84+5UhYpKcNoVtXEHZDVAq+c9mKFw2A7tN4bAZ2MaVp1IKA6gsdFFa4ADAUIoynxOXw4bdUCiV/Oe0K4oLHNgNA9sJRxybkYxhlM+VXFdRKtkDKPYkp3kcNooL7NhT8ztsCq/LnlrOhjO18kK3/bjPNLYtpfCkkkCfK717XaPzKcXYts6m1EefspI0VlDicWJPfV/pfi/ecd/LnCrvhPPPqfaNvXbY0zvtjMZiGIpZpZ60lplpjWUeRit5y30T1xCL7FHkceBO47engPrSAgrdx/ad0cNJoduOLfUDcNkNCpx2XA5j7JhZU3TsN1FV7B5rEVDqdQLJY5bLcfIYfO7k791mKAqcp75vXpSKSylFfUl27Bdnyn2K7yTXeJ02jFSZF7rP/vlCnJsWtDjTmk9prWc4lMlRSpUBvwPedkKN33FaW1v17fd/n288e5irF5Tx7bsvOmsxTrd7f7KBP+3s4bNvPo83nN+c6XCmbLSWb+1dzSxevJjW1laGr7+fqAVPv38JTU0T16iImbNx40Zu+XkHkGwC0Nrayvr163n5QC/fW9dGc1kBs8q9hGMJeoMxltQWUVvqweO0A5qjAyF+++pRljWUUOh2UOxx8qaV9SQSCf7n6QPMqyzk4rkVHOwL0jkU4uhQBJfNYG51IUtnFfPCvj7CcYuXD/TTMxyhpcLLorpCaksKePDVDvzROF+8fSWH+keoKXYzEjWJxE0qfC5ePNDH4rpiaks8vHpogN++2s7trfUsaSgb+3wH+4IcHQyzvKFk7KLoRMOpWsamci/ONBOPQ/0juOw2aordE888zVpbW6l6x+fY0R3izUur+NI70mvCtehf/oDPZeflj1+f1vwvt/XzzO5e3nNJ83FN0k4lGo3yNz/bwpLaIu69fkFa2/jsH3bwzL4+fvie1ZQVTXyh6g/F+c9Hd3HxnPKsqN0bdf+D2+gfifGVtyebmY7uR+NbOZxOus1vxPQYLR9I1mY/uKmdlw704Q/FOdwfxgQubC5lZVMJO7uGWTarlA9cNZfOoQiffXQn3f4IH7vpPAZCcVqbSynyHLu46h2OsuXoEPt7h3EYBndfOvu4ba/b10v7QJhbV9ZxaDBCmddJmffkF2daaw72jVDodpy2uXk4GueBR3azdFYRbz2/8cy/oCzQ2trKRfd+g2f39fPdu8/n/KaKTIc0ZT2BCCMxk5aKiW+EZbPR49n4a4Xx00Rmveub63jmoJ/yAtjwiZtRSm3QWp+2LXRWJXxKKTvwe+B+rfVLp5u3tbVVjx7ERXYaf6IV2UfKJ/tJGWU/Sfiym+xD2U/KKLtJ+WS/dBK+bKtHvw1YDXxWKfWUUip3q+2EEEIIIYQQIsOyqpGx1vqnwE8zHYcQQgghhBBC5INsq+ETQgghhBBCCDFNJOETQgghhBBCiDwlCZ8QQgghhBBC5ClJ+IQQQgghhBAiT0nCJ4QQQgghhBB5ShI+IYQQQgghhMhTkvAJIYQQQgghRJ6ShE8IIYQQQggh8pQkfEIIIYQQQgiRp+yZDuBMXPe5x9nbH6HIBVvuvznT4UzZgn9eS9SCpTVeHrr3ykyHM2XN960F4ObFRfz3XZcdN+26hV6+/Z4rMxWaAP7xF+v4xat+ANoeSO4vpqX5+G+38PyBXrCgwOnA47ShteadFzYRTmiCkQTDkTihaIKGCg9Xza+mpdIHQDCaYGdHgN3dw4xE4vQEogyEolwxt4KHt3dTW+zihsV12AwwLWjrH+G5vb2E4wmW1RXTNhBmRWMxhW4H/cEYBU4bTeVe7DbFU7t78YdiXLOoiq6hCAf7Q9SVuKkvKaDbH8EC6ko8HB4Yoa1vBJ/LzurmUrxuB13+CH3BKGiL3uE4155XTV2ph+FIgtkVXnZ2BtjdNYzLYVDgtBMIx7lqYSV9wThuh0Gxx8GRwTBzK70cHgjjdhhoYMOhQa6cX0lVkXvse43ETXZ3DVNb7D5uutaaXV3DOGwGc6t8Uy63+fetJQZU+xQvffymCef/n2de4D//MHBcOU/k83/ayWM7evn0m85jdUvlhPMfHRzm/T/azIIaL/9128q0tvHBH21g45FB/ucdq1jeWJbWMkJMp0//fgvff/4I1gnTC10GpqX57FuW0OGP8dy+PrxOO1edV0VDSQG/eOUQvSMx7r6ohZpSD16nneYKLxsPD3KgN8gNi2vxue2YlmZnZ4BCt52mci+dQ2Ge3dfH+Y0ljMRMKnwu6ko8J40tEkvwx23d1Je6WdNSftx7sYTFH7Z2UlPk5sI55SddPpf0BCJ0+iMsqCnE7bABMO++tcSBNyyu5Kt3rclsgGfgY7/ZwqHBEN+4YxU+rzPT4UzJb36zlg+/nHw9eg555ZVXuO3XPcdNE5kzem0N6ZdHTid8e/sjAASi8My+Q1w+tynDEU3e5x7eRjR19tnaNZLZYM7A+B/f2u0B/hvY2u6nNjXtsV25+9nyxWiyB8nyqgC+/tRefvbKUfTYO9GxV1t/s5XSAiehWIKYqUFDgcvGs3v6+e67WzEMg99v6uDhze3s6R5mJJogamoMBQ9u7gRAAev29lHucxOIxOkYDBOMmQA8vacfQ8Ej27so8dgJxzVOu0GRxw4ajg6FAfjjti6UgnDcwmFAgctB3LRQKGxGMuEKxy3shuJ3mzqoLXLTG4wyEk0QS1jYDMWj27u4aWktHqedhGWxvSPA7q4A2tJoFKVeJ49s72LZrBIsrYklLNwOG3/abuGwGSQsi/VtgwC8dKCfz79txdj39Kcd3ezvCeKwKf7y0tl4nMkLmM1H/Ty5K3mCfNOKOmZXTj7pi0QixFKvu4P6tPOOGk32IFnOE50MNh0e4L+fPICl4V3fW8/OT79uwm3c8t8v0huMsa3Dz6ziAu69fsFp5//m03t5eFsXALd980X2fGbixFWI6fSbjUf57vNHTvrecOok/KGfbcXjMAjHk38/s7eXIreDrkAUDWw5GuDGJTXUFnu4bH4FX/jTHkxLs793hH+8cSEvHejnpYMDKAV3rG7kM3/YSe9wlB+/eIgrF1RhKMV7Lm6muMDxmhi++ewBnt/Xj1Lwb29awtzqwrH3vrfuIE/u7kEp+IRnEYvqiqf/CzpLInGTX244Sixh0dY/wptW1LO/J8joJ3poey9fzWiEU/e5P+7kp68kf2O3fvN5/vThKzMb0BSNJntw7FphNNkbnSZJX/a498dPpjWfNOnMsOFoPNMhiHNYNG5xqjTC0sl/Wms0JP9pMLXGSt2kMC0rNc+x5fTYzMnXCUtjpv4dN99x82s0Giu17oSlx9Zhnbhc6m9L69ds19IaMzVxXBjJ9aamxxLWcetMrkcTN62xwBKpDxhLpKZZx6bFzeO/MTM13Uptf1TCPFaPkLDSS9ZOFJ3aYpMSN82xsrDSjHPsO9YQTpgTzh+JHfsu9Fn4TEKcKBxNpDXf+N+npSExboKlk8cxgEjMHNvfR48do/u5Th1DRo8BcVOnjqPHjk8niieOLRszj6+DHP1ba4gmTqyfzC3jv8PR/1t5clAIxo79xuITHxaFmBa9wdjEM5HjNXxFrmTtngE5WbsH8Km3rOQHr3QAUOnN3fy77YGbx2r56lIVGUvri+lLvd+Uuzck88b8ctjTn3zd9sDNtLZ+kg9eM49t7UNsOuJHawu3w47TYWBXijevqEMrg0A4QSAcIxxP0FDu5XVLarHbk7/V1y+ro7bEzc7OYQZHYnQFIgQjcda0lPL4zl7KfC5uXVWPXSkSlsX+vhBP7ewhbprMq/bRFYiytL6YEo+TgVCUglRTKbfdxh+2dRKKJrhmYTUd/hBtfWGqi1y0VHhp94cBTX2Jl/09wxzuD+Hz2FnTXEaZ18Wh/hH6RmIkTIvBUJwbF9fQXOElGE0wu9LL1iNDbG3347AbFLodBCMJrl9UzUAohstuo8zrpK1/hPlVPg4NhHDZbVy/pJqX2wa4flHNcd/rdYtq2HrUT32JB6/r2CF1ZWMpSimcNoN5U2zSWexxj+1D6R6sbzjPx6M7g2PlPJHVLZW8ZVUdz+8b4J9ftzCtbfzw7gt4/0820FhWwEdvWjTh/H933QKe3t3L/v4RPnvr0rS2IcR0eseFzTyzu4dHd/a+5j07YAEfvmYOQzGL5/f14XPauHxBJbMrfPzwpUP4Q3HuurCJedWFFDjtLKorGqvde/OKegAunF2Ox2mj0G1nVmkBf3/9Ap7c00NrUynRhKay0EXZKZr5vfeyFsp9ThpKPa+pwfuLS1oo9jioKXKzsrF0ur+as6rAaeeWlfUcHQyzdFbyc86rLhw7zi2t8WYuuDN0/5uWcmQgTPdwhG++c1Wmw5myu1bCDzcmX39hDfz7n+GvLqriOy8ka/l++ZaqDEYnTvTj993AT/564vmUztE7K62trXr9+vWZDkOcRmtrK1JG2UvKJ/tJGWW/0TIa36x9OkiTqekh+1D2kzLKblI+2U8ptUFr3Xq6eXK3SkkIIYQQQgghxGlJwieEEEIIIYQQeUoSPiGEEEIIIYTIU5LwCSGEEEIIIUSeyuleOoUQQohzWbqdxUgnMEIIce6SGj4hhBBCCCGEyFOS8AkhhBBCCCFEnpKETwghhBBCCCHylCR8QgghhBBCCJGnJOETQgghhBBCiDyVVQmfUuqLSqlnlVJfznQsQgghhBBCCJHrsmZYBqXUKsCrtb5MKfV1pdRqrfUrp1tmfHfUudzl9Ojn8ALb8+BzjC+Lk00TmZNueSRMi65AhMpCFy677TXvD47EGI7G0YCB4lvP7MNhKCxt8dKBfrxOg309YSJRC6WgwAVDETAnEasCXAqUArcdvB4b0YQiFE2ggXddVMeBvjhx02I4EuPIQIS7LmrkK3/eT7nPwb3XzeeF/X30B6M0V3hx22wcGgxT6HZQUeikfTCMXSnKC100Vfi4oLmcR7Z3UOC0UeF1E9cm9cVe/NEYTsOgwGlHAb3DUQLhOF63jYYyL9WFLjoDUYrcdqqL3JR6nWOfocsfZldXAK/LzurmckgtbzfUcfONiiZMeoejVBe5cdiM48osnXIbNdn97kt/2sXPXjnCz+5ZTXNlSVrLbD4ySEWhi/qSgrTmFyJT/H4/r/+fVzjij570/SoPvHFVI6saS7nv15uJxODW1louaKngqd09vHCgH4/NoKWqEI/DxqduWYbDpnhw41G0gpsW11JV7MGyLL797AHKfU4umlPB/u5hCj1OFtUVnfQ4eqJOfxify85I1OT5fX1UFDq5fH7VdH8dGTcQjHGgL8iKWSXY7ccf595bCB/7WO5eL+TLNY9cz2W3qVwXZE3CB1wE/Dn1+s/AhcBpE77x7rpvLT/MwR/h4nGFNpLBOM7U+B9f831raXvgZra2+6k9YZrInBPLqOI08z60pYO2vhAVhS7uurDpuPc6hsL89OXDbDo8hKU1m44MMRI7TSqnIRyZfLwaiOjU8jEYPGEb33i24zXL/Ndj+wHoHI7zT7/ZPjb92f1DE27PYUDcSr5WgKHAblNoDZapcbvsWNoilrAwrWQiWupxUuixY1oau6G4eVkd77iwkdpiD3u6h/m3h3ew+egQPped918xhwtml/Pwlk4MpXhr6yzqSzzHxfDL9UfpHY7SVF7AratmTebrGnOyffF0/rztKF96Ivm9Xfn559LaT3/0QhsPbenEaTf491uX0lAqSZ/IXsv/Y91p3+8Jw3eeOwzPHR6b9rP1nfxsfedx8x3y9wPw9N7HaSjzsrcnCMBPXzrCT997EX/3s1d5bn8/CqgsdBFNWBR7HNywpIaPvu6808bw8sEBntvXR18wyo4OP9s7ArjsNj5+83ncvqZxCp86OwUjCf7x15sZjiRY3VzG39+w4LhrhW8Pw8cyGuHUTfbYm61Odq2QL58tH6U7Fms2NeksAQKp136g9MQZlFL3KKXWK6XW9/b2HvfeszMe3szI5SRP5K++4RiQrMmzLH3cewMjMeIJi5hpMRxJEIlPpt4ueyWsY681YGmImxpLaywgblrEExrTSr6vNYQTJsORBNHU9zESjTMwkvzu+oJRhkJxLEsTS1gcHgjTF0y+Z2nNYGq+UZalx5btT823td0/0x+bp/b0TXqZI4MhAGIJi86h8HSHJERWi8ST+6rWyeOAPxxnYCTK4YHkfmFpCEbimJYmGjfpGoqgtT7tOvuDydpHfzhOX+q1pTXbO2f+GHA2+SMxhiMJIFmjKYQ4O7Kphm8IKEq9Lkr9fRyt9beAbwG0trbq8ZcpuXq3oe2Bm9POzrNZvnyOc0UjEDrN+9cvrmbzUT8LqgsxDHXcewtrCukdLqekwIGhFHOqvKzd3IEFWBZYJ1/lWeezwYiZTM4A7EYyqTNSr2NWsibPYUBZoZOFVYVsPurH1JpSjx2lDKqL3AxHEygUFT4nptb0BqMMBmO4nTZW1BdRV+qlKxChyO3g8gVVLKguBGBVYym3tc5i7dZOqovc3HVxE5U+F4FIHKfNYGFN4XHxGobixiU17OoaZll9MQBL64uZfDp2TKt74nn+7dYV/PSVdkwNLeVpLAC886JmElYbNcUuWptec29OiKyyrL6QLe3Dp53HYSSPB7FxB7C6Yicd/uNvzBjA7atnsbC2iG8+fQCtNe++uIm51YX825uW8JFfbcHjsHHz0iq2dQSpLnLzjgsaUer44+iJLp5TgaVh2axiWptK+eGLh6j0OfnwtfOm+KmzU31JAW85fxY7OgK89fxkK4YzPc5liw9c7uR/nolNPKMQ06jtgZtRn514PjXRXaezJfUM3/u01u9TSv0P8L9a65dPNX9ra6tev3792QtQTFpraytSRtlLyif7SRllv9Eymu4bXpN9XvNsry9dmfoco2Qfyn5SRtlNyif7KaU2aK1bTztPtiR8AKneOVcBm7XWHzzdvBUVFbpuViND4ThFbjtux8QPRIuzq62tjaq6BuKmRZHbwQQ3OMVZEImb2AyFw2bQ1tZGc3NzpkOalJFogphp4bbbcDtsef+bamtro6a+gUAkQYnHgdOeTa3wBSTLqKJ2FpbWFLkdU1qH1hBJmDhsBnYjz3/UZ1kuHufONW1tbVTWzmIkZlLudWKTfSCryD6UfeKmxWAoTqHLjsdpY8OGDVprfdoLhKxK+CajtbVVN9z9ZXqGI3gcdh6991JsNkn6ssmS5atY+jf/g6U1Vyyo5ANXzs10SOe05/f18dLBAZSCt1/QyE1XXZpTd+02HBrkgT/upNMfYV6VjzetqOfNK+szHdaMam1tpeTOzxOOJ6gsdPO7v7kk0yGJEyxcuoLlH/w6ALedP4u3tjZMeh0Pb+lgb3cQp93gLy9tkRuY00hqJ7Lf8pWrKHn750mYFgtqivjee1ZnOiQxjuxD2eetX3+eTn8Yt8PGHz90GS6XfcIavpy+XRxNJDuLiFsWp+skUGSG1skOLwBCUSmgTIuayYdTtE52tpFrgpE46OTvKm5aY/t/PtM6eXwDzonPm4vGd2p02t5qTyOa6h42YWpMKzdvwgoxVZbF2O8+HEtkOBohst/o9UDCtIiZ6Z13sqnTlkn75BsX8+sNR7l6YTUep9wRzTYep427Lmyi0x/mbVO46y2m18VzynHZDYo9DmblYDf6Vyyooi8Y5WB/iOUNxaxpKs90SDNOKfjo687jiV3dvOX8qQ3TIGZWkcfBravqicQtbm+dWhldt7iazUeGmFVagNeV06dlISbN5TD40FVz2XB4kHsum53pcITIeve/cQk/e+UwVy2owud57Zi+J5PTZ5YLWsq5oCX/L/py2euX12U6BJHistu4eM7pRt/Lfm85/9y7cXDjkhpuXFKT6TDEady++szGSStyO7hsXuU0RSNE7rl9TWNejTcoxExa1VTKqkn2kJ3TCZ8QQgghRDpmqidQIYTIdjn9DJ8QQgghhBBCiFOThE8IIYQQQggh8pQkfEIIIYQQQgiRpyThE0IIIYQQQog8JQmfEEIIIYQQQuQpSfiEEEIIIYQQIk9JwieEEEIIIYQQeUoSPiGEEEIIIYTIU5LwCSGEEEIIIUSekoRPCCGEEEIIIfKUJHxCCCGEEEIIkack4RNCCCGEEEKIPCUJnxBCCCGEEELkKUn4hBBCCCGEECJPnbWETyn1YaXUutTrf1BKrVNK/Vgp5UhNe4dS6nml1MNKqaKzFZcQQgghhBBC5KuzkvAppVzA8tTrSuAqrfWlwBbgzamk76+By4EfAu87G3EJIYQQQgghRD47WzV8fwX8X+r1GuCp1Os/AxcC84GtWuvEuGlCCCGEEEIIIc7AjCd8qdq7K7TWT6QmlQCB1Gs/UHqKaSdb1z1KqfVKqfW9vb0zFrMQQgghhBBC5IOzUcN3F/CTcX8PAaPP6BWl/j7ZtNfQWn9La92qtW6trKycgVCFEEIIIYQQIn+cjYRvAfB+pdQjwGKgFbgi9d61wIvAHmCJUso2bpoQQgghhBBCiDNgn+kNaK3/afS1Umqd1vp+pdQ/pXrsPAx8SWsdV0p9G3gWGATePtNxCSGEEEIIIUS+m/GEb7xUz5xorT8LfPaE935IsodOIYQQQgghhBDTQAZeF0IIIYQQQog8JQmfEEIIIYQQQuQpSfiEEEIIIYQQIk9JwieEEEIIIYQQeUoSPiGEEEIIIYTIU5LwCSGEEEIIIUSekoRPCCGEEEIIIfKUJHxCCCGEEEIIkack4RNCCCGEEEKIPCUJnxBCCCGEEELkKUn4hBBCCCGEECJPScInhBBCCCGEEHlKEj4hhBBCCCGEyFOS8AkhhBBCCCFEnppSwqeUulQpdXfqdaVSqmV6wxJCCCGEEEIIcaYmnfAppT4J/BPw0dQkB/Cj6QxKCCGEEEIIIcSZm0oN3y3AG4ERAK11B1A4nUEJIYQQQgghhDhzU0n4YlprDWgApZR3ekMSQgghhBBCCDEdppLw/UIp9U2gRCn1XuDPwLenNywhhBBCCCGEEGdqUgmfUkoBPwd+BfwaWAB8Qmv91RmIbUKJRILHd3QRjCQysXmRBn8oxpHBUKbDECkHeoMMBiNnZVumpQlFEwxH4gAkTAuAcMxkYCTK4EjsrMQxk0xLMzgSI9noIXsMR+KEY+aklonEJncc3dHppycQntQyYuYMBmNsaBvIaAxWan+wrOzaH0RumOwxSMycbe1D7OkKHDetOxCh7yxdP4jTsyyLXZ2BSe0z9slsQGutlVK/01qfDzw22QCn2yWffYreYBSv08bW+2/MdDjiBHFTc+/PNxGKmdyxupFbVtVnOqRz2mf/uIMfv3QEp93g+3evntFt7esJ8qsNR9jTHWRRbRHN5QUcGQxTXeSiPxjj1cNDNJR5uPa8ai6fXzmjscwUrTW/2nCEjqEI59UWceOSmkyHBMC+nmEe3tKJw2Zw55pGyrzO086fSFh87MFttPWNcPOyWt51UfOE2/jKn/fy8/WHcTtsfPtdrcyu9E1T9GIqjg6M8LqvrCMat1jTUsqP/urCjMTx+80dHOwbYXallzetkOO9SI/W8A+/3MyRwRC3rKzn9tWNmQ7pnPbdZw/w5cf3Yij41zcuBmDd3l7++8n92G2Kj960kEW1xRmO8tz2jm+/xLYOP1WFbh778OVpLTOVJp0vKqVm9moxTYOhZA3BSMykyx/McDTiRLGESShVy7Cj05/haMQLBwbQWhONmzyzu29Gt7WvZ5hAOM5INEEgEufVI0MAbGsPEIjECcUSBMJxjg7mbg1RwtJ0DCXvdh7Nolrso4NhtIZYwqJneOK7sd3BCG19IwC8kmYN0Zb2IQAicZOtR2XfzrR1e/uJxk1As6trOGNxjO4Hubxfi7MvZlocHgihNbx0ILO11AJePNCP1hrT0rywvx+AnZ0BLK2JJSx2d8r1dqbt602WQc9whL7h9FpLTSXhuwp4QSm1Xym1RSm1VSm1ZQrrOWMXtpRhNxTzq33UFMsd5mxT4LSzfFYJVUUubmttyHQ457y/unQ2XpeduhIPb18zs3dQl80qoaXCx4LqQpbUFfH6ZbX4XHauXljFysZSls8qpqXSyyVzy2c0jtPpC0YJpJqbToXDZnD5/Epqi91cMYO1lMFIgs1HBoklrLTmX9lYSmNZAQtrCpmTRs1bbZGb85tK8bns3LAovVrK914+m9oSDysaSnnd4uyo2cxllmWx5egQQ6GpNXN+6/mzaCr34nbYZnzfPp0rF1TN+P4g8o/LbrCwphBLa65fXJXpcM55/++6+dQUe5hVVsAHr54DwOuX1TGrzMPcKi/XLpIyyrSbl9bidthY3VJOVbE7rWUm1aQz5XVTWGZGtFQVEjU11UVuLMvCMKY0jryYIRaaEq8Dj9PGsDxnmXGzygq4fXUjTruBYagZ3VZdiYf3Xj77uGkXz6kAIG5adPojBMJx9vUEaSo/+x397uwM8Mi2LuyG4o41jVQWuqa0nvObSjm/qXSaozveP/92K92BCAtqCvnUm5ZMOH+xx8Fbzp+V9voNw+Afb1w4qZguaCnnV3998aSWEaf2lSf28cL+forcdr54+0p87smdmu12g8c+fMUMRZe+JfXFLKmXpl5iciyt6RgKYyjF7u4gN058mBMzaFFd8WuOJyOxBA2lBRhKEYwmKHQ7MhSdAGhtKaO4wEmFz5l2HwKTzpC01oe01oeAMMmhGcaGaDjbOlLNR3qHo2nf/RZnj2lqovFkufT+f/buO06Ouzz8+OeZ7dd7UT11WbIll5Ml904zHRsImE4MIbSQkJiSAAkQyi8hdGJCL6aaYgQGCzfZlot67+Wk63WvbZ/v74/ZO63Op7u9vbZ3et6vl3S7szOzz+zszM4z35ZG1TI1uVq6nX0QjduD1aGnQziWoDvklKy19ESmJYaB943bho4s7jzGGOf8BtDQpdXkZqvTHc5vWXc4Tmcoe7+PSk2GeMLQnbwpfKZDz3PZqKUngjFOR2XtvXqOmm4t3c51QXtflHianWSNuYRPRF4O/BcwB2gBFgIHgNVjXdd4vfnqGh7c28T6xSX4vZkUVqrJ5HVb1NYU09kfY8Pi6au6pxzrF5cSidsUBDwsLM2Ztjjy/R5uWFFOXXs/Vy4qmZYYrlhYTG84TsBrsbQie6uDi8Ab1y9gy/H2rOkURk28t1xdwy+eO82qOYXML56+Y1Op6eB1W7xsTTUHm3p47Tpt/pGNLptfTLA/httlsaIqf7rDueDdclEF2+s6WVaRj8eVXtldJlnSfwAbgE3GmMtE5CbgbzJYz7gV53gIxxIU+LRoOVvVlOZSFIiR59OEfLp5XRYnWvuYXxLAGWFl8iRsw/6GbgoDHhYMk1xevqCYyxeMryqkbRv2N3aT63OT53NzprOfZRV5nGzvJ9fnZlHZ+auK5vnc3L6melzvP1XmFAUoyvFQXZBePX3bNhxo6ibgcWnvmVPkfx46TDAU5WMvWYnbPfZz3Zp5RayZVzTxgSk1Q+T53UQSNsU5eq2Qjfwei4WluXjdknaCoSZPwjYcbellflEg7WUyObJixph2EbFExDLGPCIin89gPeP2rh9tp7M/wsMHW9jyzzfjH2O7BzW54rbhh1tOEo7Z3HJRBTeu0Ia+0+lTv9/H5qOtiMioXfWP11PH2th6shMReMOVC6hIM1kZi2dPdrDlWDsJ2xBLgyg2awAAgtBJREFU2Pg9Lh7a30xvJI7HsnjbtTXMmwWlJf/4y530ReI8friVB9533ajzb6vr5IkjTi+sd1wxj/klM/8zyGYtPRG+vfk44FTJ/K/XXjq9ASk1w4RjCT75+30kbMPe+iB/+kB63cyryVPX3o/LJcxNJhS7zgT57Y4zWCK8ccPCtDoEU5PnH3+xi5aeMH/a08TG949+XQCZJXxdIpIHPA78RERagGnpkaM3EiOeMNgmkXUDHytnoO099UGMgaoCnyZ80yySHPjcJLtWnkwDdcqNIe365WOVGFyvIWHbgIsznf00J+u2NwXDMz7hM8YQDMWIxe20203EE2c/78n67NVZ8YRNLHlstXZrW2WlxsoAA6eq1POXmh77G7r5874mAF51mTOe5pHmHg40OkO+nOkIacI3zQauf2xjSCQSaS2TdsInIguMMXXAK3A6bPkH4I1AIfDvYw12ItywtIxHDreyqjofr1YZzDouS6gu9NMTjrNwhOp1amp87MUreO99O1hUmsdtq6v4yASv/3RHP7GEzeLyPK5ZUkau101Rjoc5aVQ5qGvvJxRLsLAkwIn2fuYUBSgMjFxV+8pFJfjcFrk+NwUBD3Xt/SyvzOXRQ21E4zY94RjGmEmvvjqZRITVcwrYWx9kw+L0qsCuqynG6xb8HteI1VrVxCjJ9TK3LJdowuYd1ywefYFh9EfjPHa4lZWV+SzK8EKqpTtMS0+EFVXpt+lQKhsEPC7uumYRW4618bHbL5rucC54/dE4+xqCuC2hN1IJwNyiAMU5HtyWUFGQWa/WauJ88LZlfO2vR3jZpXPIC6RXY2ssWdJvgcuNMX0i8mtjzGuAH4y2kIisB74EJICtxph/EJEP4ySOp4C3GmNiIvJG4O+BDuANxpju0da9/XQXMdtwuLkPYyfA0qQvm4gIfrcb2+eMs6Om15cfPsrpzhBnusJsOTqxA6/Xtffz6+1nAKcx8Zp5RWl3yHK64+yytjFYIuT6XLzj2sW4Rhg+wuOyqK05+x5ziwKEYwliCdhyrI2dp7vI93vOmWemMcawp76bUMzmmeOdaS3jdllcsXDmbvNMIyJcMq+IhG0oyM2sPfl//eUwe+uD+DwWX3n9ZRTljK3KdW8kzs+fO03cNtR3hXihjo2oZhDbgMdtsWFJGac7w6zRflum1dPH2tlR14UA++qDgNOGL+B14xLBq9dz0+6bjx6juSfCT54+zRvXLUxrmbHstdQrr7HcxjwF3GyMuQ6oEJHrgJuMMdcCu4FXiogHeDdwPfAj4F3prDhmGzyWYAAd5i37GGOoKPBRU5qLMHNLWWaLYHIoBGMM7f0TOxxCKHa2SkEoml71ggGRlOqlfRHnQI7E7JQqm+nze1xcOr+IqsIAIPSPMZZsZAx4LCGm1TOzkjGGuUUBFpTkZFxVujf5vY/FzTnHQ7oSCUMi2awhk+WVml5msCpnJD7zz9kzXTAUw++x8HmswWuFhIEFJTnMLQ4MVmFX06c/6vxmRBMJohNdpZNzx9pL+8rDGNOU8jQOrAEeTT7fBLwB2A/sMcbERWQTcG866/7Ey1bx4y11vODiyjEPVKsmn8dl8cLVVXT1R7l8kgenVqP76Esu4r8fOsycQj8vXTOXT07gupdX5tEbKSMSt8e8r5dW5HHzygrCsQQLS3M40NTD4rLcjO8i1pTlcuOKcvoiCdYtmtnfOxHhg7ct4y97m7nrqgXTHY4aht/j4rplZUQTY//uD3jvjUv5zc56Vs8poDKDDo4Kczy8dM0cmoJhLltQlFEMSk0XS4SXra2moSvM2vmF0x3OBe8DtywlHE/gdVu88+rFfBe4bH6Rc/PRJayo1GEZptvHX7KKn287zU3LKyalSudaEenGKekLJB+TfG6MMQUjLSwia4AyoAuneidAECgGioDuIdOGW8fdwN0ACxYs4CWXzOEll8wZwyaoqbZqzohfCzWFqgoDfOGOtZOybhEZVzXCtfOLBh87pXPjc9k4h3zIJm+9ehFvvXrRdIehRjDeasPzS3N4/y3LxrWOpRV5WT2mpFIjWVyep8PIZIm8gJf/eOUl50xzu6xpGzdXPd9VS8u4amnZmJZJ+xa6McZljCkwxuQbY9zJxwPPR0v2SoCvAe/ASfgG5i9IPh9u2nAx3GuMqTXG1JaXl6cbulJKKaWUUkpdkCa95aWIuIEfAx9OVu98Drgh+fKtwNPAYeBiEXGlTFNKKaWUUkopNQ5T0fDtTmAd8Plk9+gfAR4XkSeAOuB/kr10fhvYDHTitOtTSimllFJKKTUOk57wGWPuA+4bMnkL8Pkh8/0Ip4dOpZRSSimllFITYEYPpnGwsZuvPXyErSc7pjsUpbJesD/K/z52jN/uqJ/uUNKy83QXTx1ry7ire4Cu/iiPHW7lZFvfBEY2tc509vPY4VZae9IbSiMat3kqOQ6hmhmCoRiPH27lWGtvxus40tzD44dbB4d4UEqpTISjcb735Al+8swpbFuHYMhGR5t7+NrDR3jmeHvay8zosQy+8vAR2nujPHOig++8ZZ0OBqnUCL731Em2HHNODvOKx98T5mQ61trLIwdbnCcGrh5jb1QD/ryviYauMLtOd3H39Yvxe1wTGOXU+N3OBqJxmxOtvbz1mtF769x6soNnTjg3wfL9bpZoz3dZ76H9zZzu6GdHXRfvuG4Reb6x/TR39UfZuKcRY6CzP8orLp07SZEqpWa7X++o58G9zohqhX7PNEejhvM/fz1Ca0+Ep4+3s3ZeekOZzOgMybYNde19hGMJNNfLTvWd/eytD2Y0iLaaWAGPi9aeMD2hGDm+cxOfeMImnrAxZnL3U8I257xH/DwDuPpSDmjfeZI02zYcbOrmdEc/HX1R9tYHCcfOHYB0IMHzuCwspw3xmBhjONzcw6n26Ssh7OiN8PSxtrQHkfd5XMl9aZ/zOarJc7y1lwON3RkfP8Y43+WuUBS3Nfbvqdtl4baEaCwxI29qKKWyR57PTWNXP83BEPnJMa6NMeyrD3K0pWeao1PgjF3Z0h0mYRvcVnq/8zO6hG/zkVb6ojatPZFJv1BVYxdL2Hzst3sJxxK86rK5vG6dDhw9nbaf6qCuvR+XS2jqCg9O/8PuBp4+3k5fJMGl84t43br5k3LReKy1l427G8nzuXn9lfPZUdfFsyc6WFyey8vXzkFSErJ5xTncccU8wrHE88YWs21DJG6zpz7Ik0fbsI0hbhu8LovDzT28+vJ5g/O+6OIqjrb0MqcwkFENgB2nu3j4QDMiwp1XzGdBaU7mH0CGHtjdQE84TktvhPfctHTU+es7+rh/xxl8bhe3r9FxSidbKJrgY7/ZS8K2ufv6Jdy6qnLM69h9Jkh9Z4iu/hjhDJK2hG3zpz1NtPZG8HlcvHB11ZhjUGo6xeM2vdE4RTnpDSKtJs+f9jRS1+lcIzxysBmAX207w1cfPool8OlXrubaZRXTGeIF70hzNw3BEP3R9Kvwz+jbv/1Rp3QgbqClJzzK3GqqRRM2oWgCY+BkW/90h3PB23kmSNxANG547HAr4JS4HWnupb03SmtPhI6+aNptxcbqSHMvCdsQDMVoDIY52OTcKTze2kdkmHZ680tyWFaZf04iaNuGX247zbceO8aOU52D06Jxp/RraHs/n9vF6jmFFOdmdhFxpqOfbac62V7XSVvv1J9jbGPoCcdJ2IaOvmhayzywu5HecJyOvih/TlbLUZOnLxpnf0OQg009bD7SktE64gmbXJ8bSySjNqt76oN0haJ4XMKTx9oyikGp6WKM4R9/tYt3/WgbP9xycrrDueDtb+wefLy9rguAXWe6SNg2sYTNrtPBaYpMDajrCBFLGDr6YgRD6V0bzOiEryjgRoCAx6I4xzfd4aghcrxurltWxrKKPO6o1TYl021VVQFuC3weiysXFgHgsoTammIWl+WyrDKPRWW5VBX6J+X918wrpMDvZn5JDvOLc1hXU0yez82lC4rSLtHoi8ap7wxhG4PXI1y1pJQXX1LNW65axLqaEl58cfWExpzn9zC3KId5xTnnrVo6mSwR1swrpDDgYcOikrSWuXR+ET6Pi1yfm8uS+1lNHkuEgMfC77bI92XW3uW9tyzlphUVvOemJVQUjP34u3JBCSuqCsjxunn15XquVTNLNGFoCoaxbZudyQRDTZ9/vf0icr0WeT43//GKVQC8+aqFzC/JYUlFHndeMW+UNajJVlXgw+MSCnPc+DzpVdac0VU683xuOkNxvC7B753RmzI7Gejsj9EdjtMXSa/9kZo8yyvz2Hy0DUtg1dyiwenXLSvnumXlk/7+kbhNfzSBy4qRsA1r5hWxZl7RqMs1d4fZUddJZYGfbac62XK8HUuE162bz4bFpQBsOdbO1lMdbKvr4MqaUq5aUjriOsOxBE8da8PvcbFhUSnWedpNXTa/iKZgGL/HmrbOTzr7YvSE4wRD6VXduGVlBU8eayfP52ZNyn4eyYHGbk6193HFwhLK8/Xm2Vi4LaGtL4ptIMeb2T3UeMJQEPAgjL39HoDX6+K7b12X0bJKTTevSzje1kuwP67nnyywtDKfSxcU47EsqgpzAcj3e7h+eTkey8Kr7YSnnddjEYsbMJDnTy//mdElfPVBp4pVdzhBQ2fm3VmrydEfjbO3Pkhzd5g/7G6c7nAueH850IwlzsXlT56pm/L3P9TUQ9w2dPbHaAiG0l7uL/uaONDYw/3bz9DcHcYSoarAT1+y+/n+aJynj7ezs66LHXVdPH28fdR67c+d7GDX6SDPHO8YsSv84lwvb1i/gFdfPm9aOsOwjaExGMbjEo60pHeO23SwZbDq7JY0umzui8T5c/Iz3nSgebwhX3Dak8kewK+3N2S0jqeOtdPcHea5kx0E+2MTGJ1S2a8vkiAcTeBzC3vqu0dfQE2q7z95kpNtfRxp6eFnW51rheOtfURiNr2ROHUd2kRnuh1r7QNxClVOpTmcz4xO+IoCTvUZv8eiqnDqO1NQI/O5XRQG3LgsuHhOwXSHc8Fbv6gEjMFtwYtWj71jiVQJ25zTw2YsYWOP0hPrJfMKyfe7mVscGNOwEIXJRvxzi3OoKc1hQXGAykI/ly0oBsDvdlFZ4Kcwx0NhwENlgR+/e+TkrCjgrNMSIT+Lu522RFhelYfbZXH5wuK0lllcnkc0HgeTYFFp7qjze1wWOV7n8xo4p6r0FQY8uCzBErh2eWbDhywscX6/yvK85Poyu7EQj8dpCWpbdjXz5PhclOX7sCxxfqfUtFo7rwgxBpfAxXOcLv9XVuVTnOOhosDHYh3qZ9qV5fmStUpcVKfZDGdG14NcVpHHzjNB5hQGsNLsllRNIYG4bQhFbe0qPAvctqqKvQ3dFAU81JSPngicT2dflF9sPU3cNrzysrn0R+L8cU8T+X6n982c81SvnlsU4J3XLR7z+73k4ipOd4aoyPfhc1nsOhNk1+kuqgr8XDy3EMtyqnfevsZpv5fnc5+3iuaAS+YVUpzrwe9xUZaX3VWIfvmuq+kJxykIpHe6buwKcaSlD4/LIhgZvbTI67Z4w/qFtPZEWFCiN87GqiTXy7fftYFwzOaqJZklfFcvLWP13EJyvS7crrH/lrX1hnnF156kJxznxRdX8fk71mYUh1LTwRLhT++/jta+KPOL9Rw03RLGpiscd4YySvaAX5rnS2scWDU1Fpbk0N4bpSzPhzfNJm0zOkvafSZIJG5zqqOPrj69s5ltwrEEx1r6aAyGePSQVhWbbn/e20hdez8Hm3rG1cvW6c5++qOJ5GDgfRxt6cU2ThXC5u4IkXiCI8099EaeX60yGrf5455G7t9+hu7w+ZORWMLmT8n5+qIJFpXlkutz094f5VhrL6FonK2nnMHFW7rDnO7opzDgGSxtGUksYXO0pYeCgGfUZK87HOP+7Wf4457GjHpPnAiPHmrhvx46xPZkr6Sj+cPuBkLRBN2hGBt3pVfFMM/nZlFZ7qifnXo+2xh+/twZ7nu2jsau9KsqpwpH42w92UFjV2a/Y08fb6e9N0o4ltBeOtWM9Pc/2cYd33yKnz879c0N1Lke3NuE4PSe+uB+59qtNxLn25uP8aOnTxKJa58M021PfZBo3OZMVyjtmh0zuoQvlLwAi9uAmZ6LMXV+tm1oCIZI2IajzdrGcrrtPNNFTziOCJxozXwg8aUVeRxs7CGasFk9p4BwPEFLT4SiHA/zigP8z6bDbDnWTjxheP2V889p/3a0pZdDyeEYdtZ1cf3y53cWE0vYPLCrge2nOinN87GjrpMbVzhj/pTkegl4XHT2xagu9NPcHea+Z+swBm5YUc7lC0av9vjnfU0cbuqhKxTjztp5XDJCxyY767o41e60V6gpzWXVFFdNNgb+/Q/76Q3H2Xaqk43vv27UZXrDMRLJ2rUJHZ900rX2RPjV9tMYA8bYfPUNtWNex5c2HWbn6SB+j4uvvP7SwWrM6VpenofLEhJxQ0X+5PSyq9RkCYZi/PWQc6Pi47/dy+uu1DF7p9Pd1y3hUFMPliW87eoaHgF+/PQJvvfkKSwRCvweXnGp9gY8nULRBAaIJQxuV3o3amd0wpdqhMICNU2iCRsvTi923dpL57SL2wYREKAnjap+55PjdfPadfPPmfaWq2sGHx9t7qUnHCcUTVDfGaKhKzRY57+iwIfXbRFPGOaepx3fM8c7ONjUw/G2Prxu65z2fvGEYV1NKbYxVBX66Y3EB2qc0BtOrxfL3nCc+q4Q9Z0h/rSniXyfh5qy4au4zi0OsKOuC7dLqCyYnqqfPeE40XiCrv70xtqJJs4meZ3aAcikC0UTuJKnt4NNmd1IGdhPkXiC/miCsTZJL8zxccXCYiJxmyu1DZSaYWIp7cHjo7QFV5Nv/ZJSnrjnlnOmHW7qoz+SAIGjLT3TFJkaYFlCImGwBCKx9K59ZnTCV53vpbEnSr7PRZne1cw6BX4PVWW59IbjOm5LFnjzhkV889Ej5PpcvPryefxkkt7nFZfN5dfbzyBAdVGAOUVnE7ayPB9vv2YRcds+b2cpIuB3W1w6v4hXXTaXpRX5g68FvC5uX1PFybZ+LltQREmul+uWldEbiad9oXvbqko6+qIEPC48LgtLhHjCHrbt1JLyPN5+bQ1uyyLgnfp2qCJw80UVHGnqYf3i9LbvzRsW8qVNh/G6LN52Vc3kBqgoDHjwey1sA7ddlFlnSH93wxJ+ufUMa+cXUV2UfodGAyoL/XzoBcs51d4/WBqu1ExRGPAQcQvhuGH1nPzRF1CTLp6sQed2O7+LL1xdyf7GbtyWcNPK8XX6psbv1pUVPHmsjUVluVSkeYdwRid8q+cV0HqwjQWlgcEvpcoeXrfFy9dWc7i5j1dp8f+0K8/1krDBwsq4J8B0vOSSal5ySTXGGLbXdbL9VCfrFpXgSSZUTuLkoq03wp4zQRaX57IwpTfJygI/7b1RCnM8zCkKEI4lON7ax9yiAD2RGNG4zSVzCyjO8XC0pZfqogBzh7lI7uqP0tAVZnF57jmdBpXkenn15XOp7wpR4Hezva6TX2/vY/2iEq5e+vxON6a7F8/LFhRhDKxNo7oqOL2x5vo95HvdLK1Mrze1zr4ojcEwSypy8Y3Sw6k6V37Ag9fnJhq3ecXl1Rmto6M/htdjEQw5Y1Rm0pby0vnFXDo/ve+IUtnE47J4+w1LeOp4Ox+/feV0hzMuO+o6ae+NcPPKihnbmeDuM1189P49eFwW/++1TgdQL1hdhcGp4ZNO0wk1ufweIRq3sUTS/r2Y0QnfpgNOne99Db109PRTkq+9O2WT7lCMex8/gW0MPaEY33v7ldMd0gXt03/aT1tflLa+KN/dfGxS3mPbqU7OdPazYXEpwVCMxw87x6iInDMYujGGB3Y10NUfY299kHffuGQwIdzXEKSjP0pfNE59Z4jHj7Ty3MkOCvweSvO87K3vZk6Rn5qyXCIxG2MMr7li3jlJYyxh87PnThOKJqjI9/G6dfMHS/D+eqCFPfVBinI83HnFPP6wuwmAg009gwmfMYZIPDt6l/3DrgaOt/UTisV58cWjJxR3/3gbzd0Rmonwkft38827Rh6QOxJP8LPnThOOJVjSmsfL186ZqNAvCKfa+ijscarb3vXtZ3ju4y8Y8zoONfUQisQ53dFHTzhG0Rjb8Ck1k0XiNt998iTxhM1Hf7OPB943elvlbLT7TBeff/AgxkB9V/icpg4zydcePsL+xm4E+P4TxwHYW9/N7jNBRGBBSc55m0GoqfGbnc51y47TQRra06tiO6MTvlRd4TglWhMgq8QSNtFku6rjrVrne7q1dEcGH+85Ty+dxhgePdTKma4QNywrZ0Fp+jdRnASvFXAGQ6/M99MdilEQ8JDjddEfjRPwuLAN/HDLSR7a30RVQYA5xQF6QnG213Vysr2PUDROfzRB3DbYxrDrTBdd/TEaukJ4XRbH2no50eamJxzF73ZzuKWHuo5+3nfzMpZUOCVatjFE4zYn2/vYUx+krS/Ca6+YT3VRgOYep0errv4Ycdswr9jPjrourllaSk84htsS/rinibqOfi5fWMwNyY5leiNxGrtCzCsOcLi5F7clLKnIm9Sk0Daw5VgHNvDQvqa0lulKabeXTq+Pts1gD6ThqLa1HatISu+t7b2ZtZk82tLNr7fXM6cowHtvXjbm5Y0x/Hr7GY619nHnFfN0nCw1o8QTNqHktcKxlpnbwVtvOKVN+TC9VM8UB+q7GGhKub+hG4A9Zzq575k6XBbU1hRrwpdFeiLpte+fNQlfsLcPymfm4N6HGoP8ZX8zb75qEYU5s2fg41AswUDLygYdEHjapfTlwYm24X9UO/qi7DzdBcDTJ9rHlPAFPC7y/W56wnHae6McbekjlrC5dmkpTx5t46O/2c2aeUV86Lbl/HlfE/2ROPV2PyW5Hv7517sIx2xWVRfgsoRL5xfhcQnHWntJJAx5fhdLKnLpCcXY19BNLB7jSFMvlYUBWrojeCyLh/Y3DyZ8PreLV1w6h+9sPo4YeOpoO9GYzfKqfA40dtPQGeKKhcXc92wdv9pWT67XRUdflL5oAo8lLCzNpSDgVBm9YXk5tm345qNHOdMRoizPh89jsac+yOKyPN6wfgErqibnbpMxNgPpRLr9HoVjZxOQjt7ICHM6Al4XV9YU8+SxNm5ckdk4chcyO6Un1Ez7iv7V9nq6Q3F6wj3srOtk/RjH8zvZ3s+vt9VjG0N/NMGnXr46w0iUmnr90TgDldL6YzO3x/Wrl5bR0BWivS/KG9bP3J5GG3vOJhCHmrvxAz/ccorOkHND66dbTnLTCm3Hly260+ycbdYkfK3BmXmSCPbHeM23thCN2/xi6xk2/8vN0x1Sxt7+vWc53tbL1//mClbPK6Q/ejbhm8Hn8FkptSfHVM7YdF7aeqMsGuMdPK/b4q4NC+nsj/LArgb21gexxGkDd9+zdYNt6gq8Fgcag4iBxRV59Edt9pzpImFDgd/FW69ZTE1ZDtG4zUfv30M4lmBOUYCPvvgiHtjVyOYj7RhjsIG47dRh93tcz0tOF5bm8tZrFvGjp0/hSnZbvPVkJ/WdIWwDTd0RvG4hEovjEjjR1kdRjpcokOd3U5TjGewIJhJPsKOui4RtaOkJs7wyn2jcxmA40dY3bMLXG4mzaX8zXrfFrRdV4s2gnXEmQ/+l7tn+NO4yB0NRPvH7ffRF4xxt6eNbb7pi7G96AXn1N56gN5zgJ+/cQHmBj4noUzAeT67FQCQ+9pKBPJ+b/Q1BeqNxSvJmz01DdWGIJQyzpRLzHbXzR58py6X+7vRFDX7geErJ6/4mrbGVTdp7Rr+xC7Mo4Vu3dGZ2Rd0U7CccS2Db0N6XXrFsNvqvvxzgkcOtGAN3fedpdnzihbhEB3HOViur89k/zHSPy+IN6xcSjiXI9Y399OD3uKgudDpRmVccwOd2EfC6KAy46QrFCEXj/PjZM/RHE7gspyphR3+E9r4YlsCZrjBbT3UQDMVYv6gElyWICAUBD3l+Dy+/dA5PHG2loStEdUGA/BwPZXl+bl5ZwW2rnn/HcXF5Hv/yopU8fbydHK+LZ463c7ilh6KAh4uq88n1uglFE4CwflEJDx1oJsfr4u3XLqI4pR2Vz+1izdxC6rtCrJ1XSG1NCc+d7CDgdXP5giIi8QRel4WkfOd3ne7iRJvTTf/84hwumVc45s/Tk0HnHXOLfZzpdH4Abls9+l3Y9l6nvSRAk5bEj+idP3iO7XVOdeg7/vdJHvvwzaQ5BNKIrl5ayhNH2ijO8bB6ztg7RPj9rjMEk1XiNu1rHn9ASk2hwoCHUPKxZ2b2czJrDTRYiKfc2eqcwdeqs1HtkvRKW2dNwueeob0hVRUGKAh46QnHWFAy9u64s4XPffarZCWvgFK7sfd7NPmbbv5kt9cAfs/5SwFclmSU7AF0h2O0dodZM68QAxTneFlQksO/vWw133j0GE3BEEdbejE44wG6XRaXzS/i8UMtCE5bjkjMZmddJ5cvKOLu6xdztKWX21ZVJeN2sWFxKW29USryfbT1RvGXWdy6qnKw05cBvZE4zd1hAm6LE219eFxCJG5Tu7CYfL+HV6ydw+92NVJTlstLLqmmIt9PWb4Pv8dFUeDcz8eyhPfcvJT6zhCLypxePy9N9lS29WQHm59po6rQz2tr5w/2mDWnKJDsQYuMx/ATEdzi/NjmpHkl5E/pZTOaRgnh4vI87rxiHttPd/G2GdrJwFTxpZTSusV5nOc/e6xcPCeztnMJ20bEabOZQY5PPKUKhQ5jpmYalyUITu2EsrzpGe9UneWxztbKyvE7vye5PoueiDOxskiHQZtuA8cLQCyeXnuPWZPwRWIJZmKfLT63xeULiugJx1lZPTPbIAJcu7SU//cX5/HCEudkkLDPXoSEY3oVMt1yXDAwNvmqOXlsn+D1h2MJfvpMHfvqgxjghuXlrJ2Xw7HWXtbVlPCPt7n44l8O0dUfw7YNBX43pbletp3qxEZI2IZcj3CouZt4At5/33baeqMYY9i4u4G/vW4JC0pzWFGZx43L/Rxs6uHy+UXsONPF/dvP8JJLqqks8LP1RAd5AYsnjnTQE44TiSXweVx0haL0hGKEYzb9sThPH2vHElhZXcD+hm6Ounp5+kQ7LksoDLjPGf8PnHElC6qfnygfSVZ1aQqGz+lhcVFZLm+/tgaXJeR4MzvVGszgndV027Z0pwzQHkmzE5YP3Lo8a3olzWZff+MVhL73LF2hKN9/s9P7aTRuGKhMXNfen9F6d9R1EorZhGMRjrX2UjvGi97XrJvP9546SVcoxqsu0zFP1cwSiduDF69tabQ7VpOrwCu0h509Mr/QTzOwvCzAtnqnxkqtDv8y7VKvqN2u9K6vZ03Cd6ajm7I0Bx/MJn6vm8++6hIONHazLs2Bo7PRD7ecGnx8qMm56OkJx5m5ZZazT0fk7EnhD7saBh8nbENbb4SSXO/zSslSpc4XiiVIJAzFuU5y09YbIZ6w6Q3HaemJkO93s6+hm1Pt/bgs4aaVFZxs6yPX62ZZRR4rqgqS1R0Nu88ESSQMBkNDd4SllQX0x2yePdFOKJbAtg3H2/o42LSLpeU5uFwu3JYQtw0GWFdTjNuy2NcQ5CfPnOJHW04hwDVLSllckU+uz01nf5THDrWSsA1el1Cc50MM+DwWfdE44ViC3+yo51BTN5UFfl68uirtz3VdTQmPH25lQUkOhUNKBvN8bp461k5nf5TrlpaPuVOmWHzsN0pa+s62AfvrwZZR50/Yhl9tO01DV5gNi0vPGT5juv346ZOsW1jMiuqxV4edLN9927nDy4RiCYqSj7sjmTVWTr0hlk67y6HK8vy847pFbD7SxvtvHXsvn7NNS3eYgoAn4xsYNfdsTGu+k5+7PaP1qyFSOj7SW8PTbyDZA9jX3EcZcLgtNDht+6mOaYhKnc9j++vTmm/WJHxNXTO3C1yPZeGyBDczt9rjuoWF3L/DSSI8ySqdeuLOXpKS2P1hdwPHW/uoKvRz+5pqbNsMOw7Yxj2NHGvpxe+xiCcMCWN46Zo59IRjPHqoFbcldEdi+NwWJble1s4r5GSyxCMcS9DWG6G5O4yI0BeJk+Nz0RQMsaAkh/5ognAsgduyqC4M0BAME/BYCE7VTLclxBI28YTB5XJ6Ey0IeMBAjteFMbC0PJ97HztGOBYnbsPx9n7WLijm5WvnsLc+yJ4zXTQEw+T6POT73JTl+VhRlc/r1s3ndzsb8Lkt8nweFpfn4Roh8R1qaUUeSyuGr8p3pjPEsyecH0e3ZfGii9NPJMGpSjo+ox+FfdE4DcnhG4619mZNwveKrz3BgcZu3C6L37/3apZWzNwaEKOZWxSgrqMfj9tibsnYb1w+c7yVz/3pEAkDr/vWFh6fwZ1/jdcTR9p47mQH+X43d21YON3hqDR43S4SOL3cFusYlFmpJ+Vm1uk0hvtRU6c6zTHpZk3Cd+OqiukOISPRaILrv/hXeiI2y8oDPPSPM/OHem/j2R6cwjGnGlmutr7OKgU+oTtZynfLinJ+lZz+8IEWdtR14nEJTx5pxe91saIyn1dcOpd8v4c/7W2kJxyjrqOfXK+bM50hSnK9WCK0dIep6+hnz5kgLpcz1l8olmB5ZT6vvGwujx1upS8S54qFxZzucJI/j0sozfPS3RajLNfHhsWlfPDW5fx6+2kqCwKUF/h501UL6Q7H6Isk2LCohEcOtbB+UQm5Pjc7zwRZUpFLPGF40eoqblpZwY+2nOLTG/cRTdgYA25LuGROPovLcynO9XLdsjL+sr+ZnkicykI/b7hyITcsL0fEaSe3flEJ3eEYZXk+Ll9YzNKKPKJx+5yeNfc1BDnS3MvlC4rTHq6iIODB67aIxm3K88d+IePOoEeQ5eUBDrc6d2P/7oZFo8fo99ATjrOnvos3TdIFciwW44VffpK23ghvv3YhH7x15ajLNHY52xBP2Ow8HczahC8vpa1yRW5mJUoelyFmG0wswZKKsTdO2HWqa3DYlYau0MgzZ7FIPMGf9jQRiiV44eoqSnLHfsw0dTsXoz1hZzxPlf0sOTukSSQ2c2/ez0bDHYHaD3B2Wb8kvfxn1iR83ZE4ft/MuzP01PHWwTsnR1pn7g91LOUknUj2GtAfs9Hhf7NHJKV64NHmswn67jNdtPVFsY3B57awDfSEY4RjCWIJQ11HP629EWJxm75IgldfPofiXB/hmM2lC4poDIbwuAXbNjR3h4knDIebe2gMhtl1OphsyynsqOukqtDPgpIcbl8zhz/vbcI2Nh39US6eU8jvd9XzzPEO5hf7+eOeJgJui8WluVw6v4h3XreIPQ3duARCybZsZXleLppTwJNHWvnZc86wD9GEwRLI9bmpKsxBgK89fBSv26KrP0qwP45Lwjx7op2bVjonSds2zCkK8J4blwIQjSe4f/sZTrT2cc2yMq5bVk4sYfPQ/ma6+qP8bmcDb7lqITeuHP0km+N1UZrrpbk7TFVhehWcjTGDvX0mMuiB40zX2TYwfznQyvtvG3n+5u4wf9nXSH80wU+frePll84d83uO5qfPneFkWx828KOnTqaV8L3/1mV89eGjzC/O4Y4rsrer80RKdbQEmd3kOtjs3AyJG/jPB/bxkZeNbRy95dVnk8SAO7NS4WOtvXz94aPk+d388wtXntMZzVQ53trH08fbCMdsqgv93Lhi7Ddyr1tWxlPH2phTGMgoYVRTryelGnNvOj1NqSkzXH+cYb2PklVOtA4/rvJQsybh643EmIllfGJmR8XHvWc6Bx8P9C1xnqHe1DRJHbh7X2Nw8LHXbZFI2MRsw9GWXlyWxamOfp453oHf4yIUTWAwRJOD83zj0WPcelEVVYV+znSGKAh46A7F8bic8fBCJkGez01DV4iHDzYTTdic6ewnz+fmVJsz0HpjsJ+O/jDbTnbhcllsPtLGqfY+XCJ8euNBvC4BEUpyPGzc24jXEioKA/hcQjhm0xAMUeh388c9jeyr7xwsubRw7hSHYlHqu/p57lQ7J9v68ViCz2PRE44SSyQ40dbL/20+jiVOldF4wnD5wmJyvS4e2t/Mn/c343UJZ7pCPHW0jRyv025w85E2jDF8/sGDJOwE1yyroCkYpj+aoLrQP9imcUBDV4jG5FAHu053Mbdo5KTvz/uaONDYzeULirl+eTmJxNgvflIHXg/2jd4BQl8kRk8kTixu09KdXocJ+xuCfPiXu6gpy+Xrb0xj3L5EbPAOfmd/elcLd22o4a4NNWnNO51CMXuwDV97X3oD4I6komDsP8uHUsbF6smwg6wHdjVQnywd3Hy0lRdfXD3mdTx7op3HD7dx44pyamvG3ib9SHMPv9vZgG0gz+fKKOGrLPBrxzUzjMjZyuczuZfZpmA/r/7GFvqjCT7+0ouy+kbVeGi+l13C4QtsHL4/P3OIv3vplaPPmGV+9dzh6Q5hQuxv0TrdM0ljd5wyINgfY31NCXXt/djGEDWAbSNALGETiTsDm8dsm1jcJKfHePZEOzesKKetN0JHbxSDwWNZVBf66QnHWFqZx4+ePsmBxm4sgZIcL9Fcg8cNP336ND9++jSCU5VHOHeMH4C+5N/uUGyw++FDLX3nzCNArteiL3p24YGkwgC/3l6PMWcvJPJ9FnHb6WTj2ZNd7KnvwRhDQcDDxXMKqes4w466IL2RGPGEweO2ONjQTWtPBGMMC0pz8LktWrojRBNRvv7IcZ481kFzd5iEDWvmF/L2axad01FERb6fohwnIT5fO7/B2G3DgcZujIH9jd1cv7yc3tDYj6vUFPF0cPQEpCzXh7EN4ZhNrje9Eqq7/u9pOvrj7GvsoebBA3z4RReNOP/Dh8428tf79yPbcbITbhjbMr/bfmbc73tRVT5/2tuEz22xqiqz6rNf+etRukMxdp3p4kfvWD/m5fc1dOOyBBdwqn3m1nhRY9PVFyN7umXK3Of+eICG5A2+L/zp4KxN+FR2+dmTe9Kab9Y0sppbWj7dIWRk+5n0imKVmhw2f9zbRCh+7mW4Sf5XluulIt+LGOdkYQCvSwjFEjQFw7T3RPnps3UcbOxmf0MXzd1h+iIJHtrXzEN7m5JVLAWvS7h+eRltPdFzkrKEeX6ylyphzv4byuBU3UwdG+18LKAo4CXX5yLH68ZtQTyRQAQq8n34vRb9MXuwYxi/18XKqnyWVObhtpySS6/L4obl5cwtDrCwNAe3S2jpiRBNGPqjcaJxZ/lUAa+Lt15dw3tuWsLyypHbZlmWcNmCYvweF5cnx/ibijuph5qDdEcS2MCBlJKikXSFzlbB2lcfHGFOx+fuWDvYJdXKytwMorxw7GvsHH2mIXInoPplc3eESMzpPKmlJ7Ou8YOhKO19EYKhzEo633Z1DYvK8qgs8POeG5dktA418+SkeaMp2wXDZ8+L4bje2lJToy2a3vEzo0v4PMDAz8rLrxq9c4Js9LZrFvHpP52Y7jDUBerJo+109Q9XSx/8Hot5JTmsW1TCjlMd7DodJGY7beRqSnOpLPDz+931dPRF6QrFCQL+UAKPS+hOqT/aH7M53t7Hd584TiQ69gvB8+WDAbfFSy6u4sF9zbT0RM5JCt0WXFSZS11XmFDUpizfi1jO2JBFuT5qF5ZwvK2XaMImbts0BcMsLM1h9+kuRIQra0qYVxzA63YR8FiDnT8sLM3ldbUL+PnWOnK8LjYsLuFYaz+WCFWFfkLRBPn+s03az3T243FZVBakN1DtDcvLuWH52ZtXBX4fk31LqK7jbMlputcopTluWpPDP9QuHr3qXnVRgD2ffCENXf0sz7D06ELhy6AN3uq5RTxX1z2u9z3Q1E0s4dy0ONbWm9EwQTcur+Bkex+LyjJrvV1VFOD+91xNwjY6JuQFJBK3Z0VHIH45+yMUS2jFRzU1VlWkVz6eVQmfiHwJqAW2G2M+MNr8+z/zYhq7wlQUjG2Q2myiyZ6aTpYl5yRUHstpQyE4pWodfVEw8LZrF/OFBw8RjiXI9bpYv7iEbaecnj1bkr3iGSAUtwkN08naZFTPCsVtfvxsHfHE85PCuA1HWvoYaP/f2RvF53HTH7UJeBLsbQxyqq2fvmgCS2BucYDOvijXLy9nf0OQpmCY46195PpcxG1DZYEfEagu8rPlRDvd4Th76rvpDMW4dkkZO093seV4OwtLc7jjinnMK85hf0M3P332FI3BMHdeMY8XXVxNTzjGQ/ubcbssXrCqctSL2tOtfUz2ELeXzS0afJxusvHuG5fyf0+coNDv5VWXjV5tyRjDMyfaaeoOk+NzM6945o2ZOlU81th/lgvGOL7jcF516VyOt/aS43Vz4/LMasy8dt18DjR2c1F15km9x2Whud6FJRRNzIoO3v50sH3wcSjDtrRKjdVfDzelNV/WJHwicjmQa4y5TkS+KSLrjDHPjbTM9588we93NXDlohI+fvuqwZ7tZpKlfjiqzd/UNIkmbBaW5NAYDBPwuijL8xK3naE1CgMeqgr9+DwurlhYwr+//GL+vK+Jq5aUcOWiUoQTbK/rxO0SItPUQ09shJuoqT2JReIGy0pgDMRsQ1tP1BnCASfB7eqP8cLV1ZTne2nrjZLjsTjZ3k804bRh7EyWgj53wqlud7y1F7fLIhKzefhgC6FYgoauEBX5PnqS1Xq6wzFOtTnr2HykjVsvqmTPmSCnkmMTHi7NYc28ohG3bypODf1x8Lst4rah0J9e4vD6KxeyqDyP4hzvqB3RALT0RNh9xqn6+eyJDk34RrBu8djHQVxfU4LFMWygIoPhPwCuXFzKj9+5AUsEV4bjP84vyWF+BuMIzlQ6QPvEGKlav1JqZAdb0psvaxI+4CpgU/LxJmADMGLC9z+bjhCKJTjU1MOHb1s2I4dl+PE/3MCG/3xsusNQF6iL5xTyokuq6eqPMrc4wOo5hSwsyeFEex/zinLwui1K8rwU+D1cubiEK1Oq773minnYxtATiXGgoZtI3KQxzPfE8bmEuG2wkx2zDHQCM5B7el3JHt8MFPhdLCjLozcSJ9frpjDg5mBjD8FQjDyfi9dfuYB/euFKRISrl5Sx6UAzyyrzKczxsLgsl12nu4jEDV63EInbGAzzinKYWxygIOBhf0OQgMfFNctKWZFsq3f5gmIuXVBIfWeYS+YW4nZZzC0O4DrlXFBXFY5ezXNpqY++Ueca+rmc7ZF1fc3oVT2WVeazvDKfxmCIW1dVpvUeuT43N69Mb16AohwPRTkeuvpj1JRpG76h5hR6aQhGEeA9t4zcAc5w1s4v5uqlpbR0R3htbeYdRXhcs6MtlZpZCnzZdCmq1Mzy4zsqWf/50efLpqOsCDiWfBwEnjcQkYjcDdwNsGDBAnJx2hMZIJIQ0mslk106+uK4LMEkx0CbqT7/6ov5l/v3AjAnecPfqxcPWeXEf76ERR/5I3Pz4MmP305t7SdYXJ7Hh1+44nml44vKR69gM78kh/ffsozXrpvP8dY+jG2DGKJxQ1tPmEcOtnC6s49ILEHA66a60E9rT5jecASv24OPGHtbEiwocuHyCKdb45TkuXnrdUu4f1sdB1ucaqBFfou/u2kZHpdFTyjK3oZuqosCxBOG3nCcE+19xOI2ZQU+EglDTzhGwja86rJ53LyinDPBEJUFfuYUBuiPJnhwbxNdoRhr5xdz+8VVRG2bNfPOVpy8eG4hSyvy8LiswZKOSxcUE4o6pZ4Hm3qoLvSf0y7v6iWl5Prc51wwe90WH7ptBV39MQoCTsnZwtJc3nHtIiwRAt406q1ZY6+qd/DTL+HuHz7LvKIAn3jFmlHn93lc/Orvrqa9N0J1GqV1mfC5Xbxpw0LCcZu8WXZxNxH1Sh7/8C1sPtrCRdWFlOWP/Zcsz+/h3jfV0hmKpVXiqlQ2WViWS16xl5OdUX7wumXTHU7GTn7udtZ9ciOtYTj+2RdPdzgZG+gZG2C4/nQCWuV62r1mpZdfH4yyxA+XXXZ5Wstk0y9vFzBQ8b8g+fwcxph7gXsBamtrzXtesIIfP13HzSsrKJyANgzTYdWcIl58cRVbT3bwtmsWTnc4GXvdlQvZWtfFqbY+/u1lqwBYUZVPZ7LE5V3X1kxvgAoRGbZq0XiqQrtdFvOKc4atove6K2t49mQHkbjNhsUl+Nzn/kq0dIfZebqLJRV5LBmSYL71qgW8/Btb6Isk+N83Xc6qOUWDrx1p7uFkez8XVedzpLmH0x0hLIHiXB8rKvPYuLeRooCXN21YgGVZLE3pJKQwB1526Rx21HWxsDTnvD1nDm1bV+D3UJCs7njp/KLnzV+UM3ztAhF53th8uWNIeAJeF76Am45QnCVl6VWVExG+/ZaxdYnvdVuTluwNcLss8mbhTaDllXn04FwgffP1l2S0Drfb4qaVVeOKI8fnJmeWJdMXmgu5iuij/3LbdIcwIZ775MzfN99786W89Yc7EeDX776Wt/0Fvv3Gtdz9k10I8NCHrpvuEC94//zq63jh6a7Bm9PpEDOFA3+LyHrgSzi9jW81xvxDymsvAL4HnALagE8bY54937pqa2vN1q1bJzliNR61tbXoPspeun+yn+6j7Kf7KLul7p90E6rZYLqSwkySVj2Gspvun+wnItuMMbUjzjPFCV8V0GWMCYvIT4DPGWP2JF/7ClAFzAVqjDFzR1pXWVmZ6fGcrYZ1ydyZO2znnpRxrGbTdpw8eZLZso9mg+H2T01NDfVdIWdoBgNiyWAVtYDH5QyCnBwdPRJLEInbBNwu3G4Lj+W07+uPxunqj+G2hJ5wjOgUdeBiAYhTHXpw2jlt+CxsY5Owz7ZNits2LhFyfW76owls41QLd1sWBX43wXAcwSlZ83tceFxOm72Endrdtk0sYXBZgs9tOeMMui0scd5noHTQGAiGYohAYeBsDYRQ1Bn/b7geOuO2IRJP4Hc7n30mx1Am55MDjd3EbUNxwMO8NDrd6A5FOdURwiXCqjnp9ch4qKmbWMIwtyjwvFLP4YRjCRqDYXK8rrSHtZgKJ1r7iBvD4rLcjPfRUBPxGzCwDhewKoN12MbQ0BXG45KMP++ecJzO/iglOV7yMhwbsKk7TDxhmFPkx8qg9oExEIrFcbssvC6LkydPklNaRXP32MYWdFuCMZBIOb/4XILX7aI413vOMd3QFcIAlfk+2vuieF0Wxble6rtC9EXiVBb4SNjgc1vnLeFP2IbW3gg+l4Uvee4drplHwjY0BkN43RYVGVT/zUYDv0UqO+m1XPYZ+puxbds2Y4wZsahvSut/GGNS+w6Nc+64wmuAm4wxRkQeEJF8Y8x5RwGuqamh7dZPDT6fY8HvPzvzitJr7tlIdcrzrTO0qsZw2+GrXkb1W/7nnGlq+qTeeW0DajZ9gu/c/xCv+MZTnK/7DQtwWYJtDF4DAxUgPS6hKOBhZXUBJ9p6McEwMRvG3r/g9BlagVFwGhIDeATyAx6Kc730hGL0hqOI5SKWsPEkDHZyfo8Lcn0e8n1u5hbncNmCIl5zxTyWlOfx02fq+N3OegDuvn4xt1xUyfa6Th471ArAS9dUsyylSqkxhnsfP05/NEFpnpc3X1WT0TE0dD+Pdqf/C3/cwzcerxvzewwc7+VVufzpgzeOOP8//Gw7bTsbgWT1jjTe47rPP0y0O4yI8KU3X8F1yytGXWay3f2D52g74HSJVlyaw8MfvmlCznNj3WfDLT/e35G7f7iVLcfaAPi7W5fzjusWj3kd6z79EN5YAr/HxXMfH3sVvW89dpSvP3wUgCuXlvGtN414w3pYf9jdwJHmXlyW8LZrarjp2qtou/VT53w+41VR5Oe/X3sp6xeX8tmN+7nvWef4Kcz1kmsbLBFuW1XO956swwfYArULi3BZLj77qouHbSP9/vu2s+1UJ5G4zXXLyllQksMb1i94XvL95u88Q7zO6S34X162ijtrF0zglk0PLUHKbrW1tedcb2dyjlITa+hvBtteun20ZaalQYWIrAHKjDH7Uya7zNlb9UF4/vBTInK3iGwVka2tra1TEapSSo3JFFaayAq2fYFtsFJpMPYErGOUQ8uegPdQSl0YpryFt4iUAF8DXjvkpdTSvrQ6bWlLeW0mlu6Bc5dkNrQrGG47LplbSNt55lfT6+TnnF461y4o5q718/ndjnrito3f48brdpFIJFg7r4hcv5tCvwfLZXGspZsTbf1cVFVAeaGPynw/d21YyI4znfxmWwPleT7+eqCJM8HIOb18TZZcF1guF6HkYHyWgRwPdMWc1xeVBugJR+jqt6kq9CEIbb1RCv0WVy4uY3d9kN5QjIDPRUmujxeurOQvB1sQgUvmFbGiKp85BX6OtfYSDMVxWU51qoZgiJbuCPk+N0ur8gl4LBYU5+L3uphbHGBphXP3/rVXzMPnFnJ8bm65yClDvXReER7Lwu2Sc0r3wOls5TVXzONEWx/LK5zXxnsMpXMX9p9fcgm/3NZIa1+M11yaXsch/3fXWu7+8S7yfS7+/KGbRp3/S6+/nK0nN9HYHeGzr3xeB8zDv8ebr+A/Nh6ktqYwK0r3AO59yzrecO/TBMNR7nu70zHORJ/nMrlznnr+zXQUvP9+7aV86g/7qMz3ZVS6B/DFO9byy22ned26zIaGePcNS+kOxWjvjfKvL03vezLUrRdVUl3opyLfT36yo6UP37aMLz50ZNRlLWAgh6rIdRFLQDCcwMapKruwxMeCsnzuqJ3P+uR4if/8wuWD1bzff9MSfvhsHQuKc7jjivlEEoZnT3TwgVuX0hO2WVyWe94ekD/1sou594njLCjJYUVV/uA4qEN95fVr+Y+NB5lfnMPrrpz5pXtq5tHSvexy8nO3I2kMyzDVbfjcwO+BTxljnhny2leA+4DdwEZjzI0jrUs7bcl+Wk0ju+n+yX66j7Kf7qPspvsn++k+mlpj7VhH90/2S6fTlqmu0nknsA74vIg8KiJXichXk699AfgMzqDrn53iuJRSSimllFJq1pnqTlvuwynFS7Ul+doZ4OapjEcppZRSSimlZrPZNwquUkoppZRSSilAEz6llFJKKaWUmrU04VNKKaWUUkqpWUoTPqWUUkoppZSapTThU0oppZRSSqlZShM+pZRSSimllJqlNOFTSimllFJKqVlKEz6llFJKKaWUmqU04VNKKaWUUkqpWUoTPqWUUkoppZSapTThU0oppZRSSqlZShM+pZRSSimllJqlNOFTSimllFJKqVlKEz6llFJKKaWUmqU04VNKKaWUUkqpWUoTPqWUUkoppZSapaY04ROROSKyXUTCIuIe8tonRWSXiDwqIh+ayriUUkoppZRSajZyjz7LhOoAbgF+c57X/9EYs2kK41FKKaWUUkqpWWtKS/iMMWFjTOcIs3xeRDaJyKVTFZNSSimllFJKzVbZ1IbvK8aYK4C/A7463AwicreIbBWRra2trVMbnVJKKaWUUkrNMFmT8BljOpJ/j4wwz73GmFpjTG15efnUBaeUUkoppZRSM9CY2/CJyKtHet0Yc38mgYhIgTGmW0TKMolLKaWUUkoppdS5MkmsXpb8WwFcDTycfH4T8Chw3oRPRDzAn4C1wJ9F5KPAXcaY9wFfFJGLcUod78kgLqWUUkoppZRSKcac8Blj3gYgIn8AVhljGpPPq4Gvj7JsDLh1yORnkq+9a6yxKKWUUkoppZQ6v/G04asZSPaSmoHl44xHKaWUUkoppdQEGU9buUdF5M/AfYABXg88MiFRKaWUUkoppZQat4wTPmPMe0XkVcD1yUn3GmPON6C6UkoppZRSSqkpNt7eMLcDPcaYTSKSIyL5xpieiQhMKaWUUkoppdT4ZNyGT0T+FvgV8L/JSXOB305ATEoppZRSSimlJsB4Om35e+AaoBsGB0yvmIiglFJKKaWUUkqN33gSvogxJjrwRETcOJ23KKWUUkoppZTKAuNJ+B5LDpweEJHbgF8CD0xMWEoppZRSSimlxms8Cd89QCuwB3gX8EdjzMcmJCqllFJKKaWUUuM2nl46P2mM+Tfg2wAi4hKRnxhj3jgxoSmllFJKKaWUGo/xlPAtEJGPAIiIF7gfODIhUSmllFJKKaWUGrfxJHxvAy5JJn1/AB41xnxyQqJSSimllFJKKTVuY67SKSKXpzz9Ms44fE/idOJyuTFm+0QFp5RSSimllFIqc5m04fuvIc87gVXJ6Qa4ebxBKaWUUkoppZQavzEnfMaYm0TEAu40xvx8EmJSSimllFJKKTUBMmrDZ4yxgb+f4FiUUkoppZRSSk2g8XTa8pCI/JOIzBeRkoF/Iy0gInNEZLuIhEXEPcxrD4vIUyJy6zjiUkoppZRSSinF+Mbhe3vyb2pJnwEWj7BMB3AL8JthXrsH+DiwG6fXz03jiE0ppZRSSimlLngZJ3zGmEUZLBMGwiIy3MtrgA8YY4yI9IhIvjGmZ6T1/eSZU/xlXzOXLyziA7csH2s4WeM9P9nOgcZuXrammg+9YMV0h5ORF/znRg4Hzz4/+bnbqWvrpeaejedMU9MndV+8YHnpOa/Vtffz4L5GinK8eF0WTd1hblpRwZwiP/fvqGfPmS6qCwO8dM0cVlTlP2/dvZE49287zR/3NGFZwotXV9HWF+Gxw22U53q5aG4+zxzv5EhTBx0hZ5l8n8V7blrO8so8VhXZXPXlsx381pTmEIklKAx48LgtFpbk8orL5nLbqspz3jccS3D/9jM8c7yD9r4oq+YU8J4bl1CU4+Uv+5o41trH1UtKWTu/aMyf176GIJuPtLGwJIcXXVzFec5bHGjs5rHDrcwvzuEll5w737ZTHTx7opOV1fnctKJizDEAYz6G3va9Z3jkUBsAn3nlKt64YeRTdSKR4N0/2c6xlj5ec8Vc/v6mZRnFOZLecJxXf/NJ2nojvHlDDR+8beaer4faUx8c93nu7d97lsePtFEY8LDlX27A6/WOafmEbfjD7gaagmFuXlnBssrnH6MzQX80zm93NBCKJXjpmmoqC/wTtu4X/9efONBqP2+6JfDC1VV8864rRly+Nxzn3/+wj/beKH973WI2LCkdcX6lZpPUc1yeC/Z+Rq/nplMmvznjqdKJiFwsIq8VkTcP/BvH6lzGGJN8HASKR1vg4QMthGMJnjraTjgaH8dbT5/6zhC7TncSjSd46EDzdIeTsdRkb0Awkpj6QFRa/nK4/Zznu+u76IskONLcw67TXYSiCXad7uJ4ax/1HSFOd4So7wyx83TnsOs70drHsdY+Tnf209EX4YE9jRxq6qG1J0xzb4Q/7m6ioy86mOwB9ERswrEEu88Eecd9h85ZX2tPhGAoxqmOfpq6wxxu7mFvfZBQ9Nzv1Kn2fs50hjjc0kNLT5jDTT0caemlPxpnX0M34ViCHXXDxzyancnP4WBTD93h859fBuY73NxDMBQ757VtpzoJxxLsrOsinnj+xeZo6jt7x7zM5iNtg4+/8tejo85/tLWP/Q3dROIJHtzbNOb3S8emA000doWIxW3+sLthUt5jJttyvAPbGDr7o2zcM/bfgfa+CMdb++iPJth5umviA5wiJ9v6ae4O0x2Ksb+xe0LXPVyyB2AbeOpo27Cvpdp9potT7f30RuIz+rdaqfHq1Uu7rPLp3+1Ka76MEz4R+QTw1eS/m4AvAC/PdH1A6leoAOga5j3vFpGtIrK1tbWVKxeVIAJr5hXi946ndur0qSrwsqQiDxHhqsUz945h8TCFHzmecd1PUJPo4sqcc56vrCrAbQnzS3JYVpmHJcKqOQXUlOZSlu+lNNdLaZ6XVdWFw65vQWkO1UV+SvN85Hjd3LS8nAWlOeT73OT73Vy/vJxcn5tAylfCZ4EILKvM49N3nFuyXRDwEPC4KM/3URTwMq8kwJKKPPxDvlPzigOU5fmYVxygwO9hfkmARWW5BDyu5HEFq+cOH/NoVlUXYImwMLkdo823oCSHAr/nnNdWzylEBFZW5eN2jf14mFucN+Zl1qRs75vWLxh1/kUluSwszUVEuGZJ2ZjfLx3XLiujKMeLiHDdssl5j5ls9ZwCRIRcr4vbVlaOvsAQJTle5hYFsES4qLpgEiKcGvNLAhQGPHjdFssqxv7dH0lVzvDTRUirBsDqOQVUFPhwW/odVhc2vbLLLh9/xdq05pOzhWpjIyJ7gLXADmPMWhGpBP7PGPOyNJZ9FLjVGBNPmfYV4D6cNnwbjTE3jrSO2tpas3XrVmzbxrJm/tcvkUjgcrmmO4xxe/DBB3nRi14EQG1tLVu3buXAgQNcdNFF0xyZAjh69ChLly4Fzu6fAcaYweqIqY8HngPnrdaYOp8xBsuyBtcxcIzato2IICI0NzdTWVn5vPdpbGykqqpq2FhGeu+B9x0639D1j1W6y48033hiGNhHTz75JNdcc03ay/X19ZGbmzum95qKc1A8Hsftnpk3585nYB899NBD3HbbbRmvJxqNjrkq51Dj/b5ni4ncjtTzXCKRYNu2baxevZrc3FxiMadE3uPxjLSKc8yWa45sMvS3SE2u1OqAIxmoKjiwf3bt2sXateklF2ry1dXVsWCBc1NXRLYZY2pHmn88Z61QcniGuIgUAC2M3GELIuIRkU04ieKfRWS9iHw1+fIXgM/gdNby2XSDmC0n3tmQ7AGDyV4qTfayx0CyN5zUC6yhF1ujJVyp8w0ckwPzDzy3LGtwWmVl5bDvU11dPbiOgfdMXW609x0u7vFId/mR5puIC9exJHvAmJM9mJpz0GxL9lKNJ9kDxp3swcR817LBZG2Hy+XiyiuvHDw+PB7PmJI9mD3XHEqNlSZ72WUg2UvXeH59t4pIEfBtYBvQCzw70gLGmBgwdMiFZ5KvnQFuHkc8SimllFJKKaVSjKeXzvckH35LRB4ECowxuycmLKWUUkoppZRS4zWu+jUi8mrgWpzx957AaX+nlFJKKaWUUioLjKeXzm8A7wb2AHuBd4nI1ycqMKWUUkoppZRS4zOeEr4bgIsHxs4TkR/gJH9KKaWUUkoppbLAeLqbOgSkdhEzH63SqZRSSimllFJZY8wlfCLyAE6bvULggIg8m3y+HnhqYsNTSimllFJKKZWpTKp0/r8Jj0IppZRSSiml1IQbc8JnjHks9Xly0PXZO5quUkoppZRSSs1QGSdqInI38B9ACLABwanauXhiQlNKKaWUUkopNR7jKZn7MLDaGNM2UcEopZRSSimllJo44+ml8xjQP1GBKKWUUkoppZSaWOMp4fsI8JSIPANEBiYaY94/7qiUUkoppZRSSo3beBK+/wUexhls3Z6YcJRSSimllFJKTZTxJHxxY8yHJiwSpZRSSimllFITajxt+B4RkbtFpFpESgb+TVhkSimllFJKKaXGZTwlfG9I/v1IyjQdlkEppZRSSimlskTGCZ8xZtFEBqKUUkoppZRSamKNuUqniPxzyuM7h7z22TSW/5KIbBaRLw+Z/kkR2SUij4qItg1USimllFJKqXHKpA3f61Mef2TIay8aaUERuRzINcZcB3hFZN2QWf7RGHOjMea/M4hLKaWUUkoppVSKTBI+Oc/j4Z4PdRWwKfl4E7BhyOufF5FNInJpBnEppZRSSimllEqRScJnzvN4uOdDFQHdycdBoDjlta8YY64A/g746nALJ3sF3SoiW1tbW9OPWCmllFJKKaUuQJl02rJWRLpxSvMCycckn/tHWbYLKEg+Lkg+B8AY05H8e0Rk+IJCY8y9wL0AtbW1oyWXSimllFJqBqi5Z2Na85383O2THIlSs8+YS/iMMS5jTIExJt8Y404+HnjuGWXxLcAtyce3Ak8PvCAiBcm/ZYxvuAillFJKKaWUUoxv4PUxM8ZsB8IishmwjTHPishA9c0visiTwAPAPVMZl1JKKaWUUkrNRlNekmaM+cCQ5+9L/n3XVMeilFJKKaWUUrPZlJbwKaWUUkoppZSaOprwKaWUUkoppdQspQmfUkoppZRSSs1SmvAppZRSSiml1CylCZ9SSimllFJKzVKa8CmllFJKKaXULKUJn1JKKaWUUkrNUprwKaWUUkoppdQspQmfUkoppZRSSs1SmvAppZRSSiml1Czlnu4AlFJKKaWUmg1q7tmY1nwnP3f7JEei1FlawqeUUkoppZRSs5QmfEoppZRSSik1S2nCp5RSSimllFKzlCZ8SimllFJKKTVLacKnlFJKKaWUUrPUlCd8IvIlEdksIl8eMn2OiDwsIk+JyK1THZdSSimllFJKzTZTOiyDiFwO5BpjrhORb4rIOmPMc8mX7wE+DuwG/gBsGm19qV3fzuTubV/3zSfYWhfkg7cs5n23XjTd4WRsYH+84uIAX77r5nOmXV4I939k5u6j2eCzv/8T9z5lA2ePl4Rt+MkzJ9lV18Xf3bwEl7ho7QljieCyhMVleRTmeADoi8Q50dbH/OIcOvujNHeHsQQ6emN8euM+3G6Ll62Zw/wiLw/ua0HE4tL5hVQVBahr62fNvAJ+s6OBYDhGR1+EurZeemOQ54UrFpZRVeinyO+mvS/MgaY+KvL93LC8nPbeCG63cPPKSh7Y2cCTx1rp7I1w9w2LyfX76I3EePxQK3sbgrzz2houXVDKnKIA0YTN9lOdBDwWfZEE6xaVIMDDB5o50BjE7XKxvCqfi+cW4ne7WFKeyxNH2ykMePB7LDYfaeOVa6v566FWCgMeLl9YzI66Lq5fVk5Jnnfwc43EExxp7qWywE95vu95n/uOuk7qO0O8YHUVXvfY77GN9Tz33HPPceevW9KeH+DHT53gj/ua+fjtK1k1p2jMMV7oBvbR3+bDxz429vPcPb/Yzs+2NxJww4FPZ3aefNXXNnOwqZdv37WWa1fOGfPyxhgON/eS43UxvyQnoxiyWVd/lEv//aHB5wL4gDAwt9BNvs/H9SvKcVnC/oZu+mJxrl1cTktPiL/sa8JGuHJhMQvK8nj3DUtp743wr7/fQ0t3hH9+4QpedMkc4nGbrz5yBJ/bxfpFxTQGI2kf97ZtONTcQ0HAQ2Weh//561Hy/G7efcNS4nGbv+xvprLQz0XV+Ty0v5llFXmsmlM4aZ/XZHrmRDvPHu/gDesXUJr3/HOmyj4bN27k7zc7j2fy9fZskUn+M9Xj8F3F2URuE7ABGEj41gAfMMYYEekRkXxjTE+6K/7lLzdy550z70v44O7TPHMqCMB/bTo+YxO+1C/f7/aG+DKwpz5IdXLa9uC0hKVSDCR74OyvMuC3O87wn388SMI2PHW8g1deOpedp7uIJ2zKC3xcNr+Yd163CBHh97saaAqGicZtBMPOM0GiCZsdpzpJGGe9X3vkKAKDzx891ILXbWFZQjRuE0sYzJC4eqPw2JE2wLkIG3h9X2MPm4+0IpbgdVl878lT9IRjg+v+5B8OUZHvpbs/SjjhTPvPB49wzZIOLp5byK7TXTR1h+nqj5Ln93DxoQK6QjF2ne4iHHM+C78LFpTlcfHcQkA409lPIpGgIRhBBH7yzClclmCMoSjHS77fw5NH2/jinWsH439ofzNHmnvxui3ece0i/B7X4GuHm3v4woOHsI2hrrOf99y4dEz7rKUlOOZqGAPJHjj7ebQfgyNNPXzyDwewjeFvvv0Muz7xwjG+44Ut9Tz37R74WAbr+Nn2RgBCcXjHd7fwnbdfNabl/3PjXnac6Qbgru/v4OTnxp7wbTvVyebkcXhn7TzmFc+upC812QPnPBNOPq4PxoE4B1v6sICBM+XWk+f+cD14oBW/q40dp7to7AxxJuis4b337eTBiny+9+RJfr+rHtvAT572MK8kJ+3jfsvxdp490YEInGzr468HmgGwbfC4hE0HWhCBecUBTneEcFvCf7/uUioL/Jl/KNOgKRjiX361m1jCZuupTn7w9iunOySVhoFkD9L7XVFT50PfTW/cx6mu0lkEdCcfB4HilNdcxhhzntcAEJG7RWSriGxtbW0957XfNUx8sFOhvTc63SGoC1hvOI6dPOriCZuEncAYQ8y2SdiGaMJm4KiMxJysKhxLYOPckY4nzODyAMYwOD84F1WJ5DTbHprqPd/QOWyckgfbQNy2z1k3A+seuowx2MaJ3U4ui4FYwiYac6alvl88YUjYhlA0Pvie8WRWGU2cTZIHtz+eOOf9onFnnnjCEB+yjaFofPD9wtFzl0tH/+gf2bh1h2ODn+HAdqvp09IdGfsyPeP/HYnEz37XY7Pse2CGnjhGmneU120gFEuc83kZDKFYgt5obPD9YrbzerrH/cB5xBjoDsUHp3eHYvQl12EM9EecxwljiMbs568oy4VjicHz5MA5VSmVuW0n0ptvqkv4uoCC5OOC5PMBqUf+0NcAMMbcC9wLUFtba9pSXvvxB2bm3YY3Xr2E/3roKB2hONcufl6OO2Oc/Nztg6V8geS0S+YW0nb+RdQUK+LsQXXyc7dTW/sJXnflAo609HKgqZsP3LyUvICXVXOLcIngc1tcPLcQyxIAXrp2Dgcau1lclktLT4SVVfm4LItjLT1869GjuCyLm1ZWUp7v5fEjLWCEldUFLCjJoaErxNLyPB7c10RXKEZ/OE5n+OxFzbwiP1WFfvK8Ljr6o5zuDJHnd3Pjsgra+iJ4XS5euKqSX2w7zc66IOF4jBeuqmZ+aQ6dfTEeO9xMS3eEWy6q4MWXzGFRWR63XlTJk8facIsQidvcsKIcY5xSzQNNPbgsYXllPlctLiXP72ZlVT5/3tdMvs+N32vx+OE2Xn5pNZv2t1Dg93D98nKePdnBC1dXnfO53rqqkt2ng8wtDpDnO/eUunZ+MW+5qob6rn5eWzt/zPuspnLsx9ANi3N47Hj/4H4ezRU1Jby+dh5PHm3nwy9aPuYYL3RlKbdNJcN1LCj0Uhd0krYHPnjjmJf/0usv59FDf6YrFOe9Ny7OKIYrF5XgsoQcr4tFZbkZrSNbiQhP/MtNXPv5R4Z93WNBjs/N6up8cn0eDjV1E03YXDKnkPZQhF2nujHAwtIASyvz+afbVtAYDPOR3+ymNxznrdfUcMm8Iv795audEjm3xbVLymjvi6R93F+1pBSfx6Iw4OFtVy/go7/ZR8Dr4p9esIxQ3JDrc1FZ4GfDolLu317P8qp85pfOvFLYmrI8PnDLMrad7ORt19ZMdzgqTZdVwg6n0JmvXze9sahzPfYftyOfHn0+Gcudr7EQkTk4bfFWAXnGmHiyDd+7gE8BzwIdwIeMMZtE5JvAJYALKDLGjFi3sba21mzdunVSYlcTo7a2Ft1H2Uv3T/bTfZT9dB9lN90/2W9gH6U2DRlJtlcnzPbtGGt8egxlPxHZZoypHXGeSUz4/DiFPb8BbjXGxJPTvwzcAWwB3gocNMbME5FPJad3A7lArTHmvPVUysrKTI/nbInYJXNnZuNlcNqADJhN23Hy5Elmyz6aDYbbPwsX1nCkuYdIwsbrEkRksFqf2yX4PS78HhciEIvbWCIUBNyAEI0nsI1TbTEYioEIlgh2siqTnaxuaQn43C7iCZvYMNU6U9vtjWTofH63hYgQS9jOdAM+txCKOXGW5nnpi8SJ24bCgIeCgIeW7jD90QSWCAZwW4LfY2EbCHhchOMJLASfx8I2hjyfh67+KC5LyPe76YskyPe76Y3EicRtvC6LklwvLivTsp1zxRI2kbiN3+PCbUlGx1Am55NDTT1EEzYluR7mFo1eahCN2zQEQ/jcFtWFgVHnB2jvixKOJijP92XUeU222rH/CO7CisHnmZznJuI3YGAdHgtWVo99HQnb0NwdxmXJtLYLC0UT2MaQ43NnXGKa6uTJkyxYuJD+SIIT7X3nnU+AHK9rsPqkAB6Xcx5IraptCZTm+ijO9dLYFSISt3FZgtdtUZbrpb4rhNtlUZzjoavfqS5dkuslGrfxuS0KAp5h3z+WsGnpieBzW5TkeGnuCSMiVBX4MQb6onFclnOObQyG8LldVBfOrPZ7Azr7o/RFEpTlefF7XLPqWmHgOCzL9VJdlN65Mdukno8EyIt1UlNTM23xqNFt27bNGGNG/GGdtCqdxpgwEBaRodM/ICJrgTuTHbTsEJF8nJLAm4wxLSLyVWAFsOd866+pqaHt1k8NPu8DDmT5XZ/hLL5n42CDf4CtM3AbwLljNHQ7fNXLqH7L/5wzTU2f1Lt6bUDNpk/w8k98n+8+eeq8y1ji/HCBEIrFCXjdzCsOsK6mhF2nuzjTGaK+K0Q29bM2cKkgyccG8LqEWy6u5IFdTQytrGaJc2EnQJ4lGGPj87ipyPcRidsUJ2xs2ybg9VCe76M3EscYw+mOfqoKA9y+ppoP3jr+qpDGGL752DEiMZuiHA9vu2ZRRsfQ0P082l3k728+zic3HhjTe9zxzacwTU5z7I++fDWvvmLkams76zp57307MMawoqqA77513ajvMVNMxHlurPtsqIs+upHqlOZcmcTwkV/v5vEjTtv4D966nDszqII8Xsdae/n9TqdB/vrFJVy9pGzc66ytreVj3/4t//DzXef8Rp1PweizYAGL5hXQV9892IY51+vCGENpsi1ejtfCRG0sy7lxdllVPpYlfPoVF7O0Mv9563zPT7az63QnAKvnFrIvedH9jmsXs6gsl52nuwDYtL8JOvpBhE++6mJeunZuGhFnj5Ntvdz1nWcxxjC/OIefveuqWXOtMNx10Ew0tASwbNMntIQvy4nI9tHmma7brMN10FLE+Tt0AUbutOUVkxfrpCqf7gDUBa3oPHebBwhgWSACliXJ0joLV/K5y+VMy0ap95pEIMc7/P0tSb7usgQR57nbcko7U3vcHHjs97gG2zW6LKdkcKJ4Xc4peaAEbOLWfH6VhWNP13O8TmQiQt4o3yGAgNc1+D3xT+DnNRtl0iorZwIKelLbnw5tizpVBr7/4JxnJorPPbHfOUucBG/oqS81fk/yHJJ6PrFEzhtLwOMsKyIUBc5+/vl+9+BnIQL+gWMPKPSPfuxlm4HaCwC+WXYumF1bo2ab6TmrD99BS1fycZgMOm353Ay9k/JMSmcnM1lqpy0DHYKndtqyNEuTggvVQKctf3/zcg42dfPU0XbWLy7BGKG+q59YAhaX57CysoDlVfm4LKGlJ4LHLVyzuJyEMdywIkw0nuBocy+/2VmPCyHX56KjL4yxE7SHbKJxm/yAh+sWF3Oys5+9Db10h88e/vluyAv46OiPEEtAjtsZKqEv4WRePpdFNJZAgOJ8P/3ROF39Tmcvb1k/h7aQTXN3mEgkQTSe4LolRTxwoJ3yXC9vu3YxO053cLojxKsvm8elC4qpKgzw2MFmyvM89EYNVUV+lpXl0R9LsHZeAYea+wj4XCwozqEnHOfapaVs3NN0zjh81y4pZfvpLhq6wlTk+3jBqsoJ2Sciwp218znd0T/YacaqDDo++mAV/E+T8zidkqIXr5nLC7af4YnjnXzmFavSeo9vvPEK/usvB1lZVcALVlWNOv+KqgL+4xUXs68xyBvXL0zrPWaKieicqsoPTckxAvZn8Fu27d/Onn//Zm1m38d7XryC6qIA5XleXnxJOmVhE29+SQ6vuXweoViC5ZV5E7beWy+q5A/vu5aXfvWJ884zt8jH6y6bwzceO0HYhjkFXlZWFhBKxHjueBAbKPILuX4f//Lii7hqSSnfeuwoR5p7qSrwU17g5x1XL+Jf7t/F3OIAt19SzYN7mwjHbe66agGHmnpZXJZ33o5WPv3KS/jBUydYVpnHjcvL+ckzdeT43NxxxXxs21Cc66Ug4OHtV9fw35uOsKo6n+tWVAy7rmxWVRjg83esYevJDl6/bgFw7jE081LYs46lXAf9+G8vmeZoMpd6PTdwraBmvklrwzf4BiKPktKGLzntK8B9OIOsbzTG3CgiHwIagV8AjwC3GWPO2z+1dtqS/bShb3bT/ZP9dB9lP91H2U33T/bTfZTdUvdPtndIc6FKp9OWSavSKSIeEdkErAX+LCLrk23zAL4AfAZn8PXPJqf9H/BGYDPw3ZGSPaWUUkoppZRSo5vMTltiwK1DJj+TfO0McPOQ+buBl05WPEoppZRSSil1oZk9fWMrpZRSSimllDqHJnxKKaWUUkopNUtpwqeUUkoppZRSs5QmfEoppZRSSik1S2nCp5RSSimllFKzlCZ8SimllFJKKTVLacKnlFJKKaWUUrOUJnxKKaWUUkopNUtpwqeUUkoppZRSs5QmfEoppZRSSik1S2nCp5RSSimllFKzlCZ8SimllFJKKTVLacKnlFJKKaWUUrOUJnxKKaWUUkopNUtpwqeUUkoppZRSs5QmfEoppZRSSik1S01qwiciXxKRzSLy5SHT/0dEHk3+60xOe6uIHEpO+8JkxqWUUkoppZRSFwL3ZK1YRC4Hco0x14nIN0VknTHmOQBjzAeT81wG/GPKYl80xvzfZMWklFJKKaWUUheSySzhuwrYlHy8CdgwzDyvAu5Pef5BEXlcRG6ZxLiUUkoppZRS6oIwmQlfEdCdfBwEioeZ50XAg8nHvwXWAK8B/p+IuIbOLCJ3i8hWEdna2to64QErpZRSSiml1GwymQlfF1CQfFyQfD5IRJYB9caYfgBjTJcxxjbGtAKHgcqhKzTG3GuMqTXG1JaXl09i6EoppZRSSik1801mwrcFGKiaeSvw9JDXXwX8ZuCJiBQk/waAZYAW4SmllFJKKaXUOExawmeM2Q6ERWQzYBtjnhWRr6bM8lLggZTn/yAiW4BHgc8ZY2KTFZtSSimllFJKXQgmrZdOAGPMB4Y8f1/K4+uHvPYp4FOTGY9SSimllFJKXUh04HWllFJKKaWUmqVGLOETkR7AnO91Y0zB+V5TSimllFJKKTW9Rkz4jDH5ACLy70AT8CNAgDcC+ZMenVJKKaWUUkqpjKVbpfOFxphvGGN6jDHdxphv4oyXp5RSSimllFIqS6Wb8CVE5I0i4hIRS0TeCCQmMzCllFJKKaWUUuOTbsL3BuC1QHPy353JaUoppZRSSimlslRawzIYY04Cr5jcUJRSSimllFJKTaS0SvhEZLmI/FVE9iafrxGRj09uaEoppZRSSimlxiPdKp3fBj4CxACMMbuB109WUEoppZRSSimlxi/dhC/HGPPskGnxiQ5GKaWUUkoppdTESasNH9AmIktIDsIuIncAjZMWlVJKKaWUUkpdYGru2ZjWfCc/d3va60w34ft74F5gpYjUAydwBl9XSimllFJKKZWl0k34ThljbhWRXMAyxvRMZlBKKaWUUkoppcYv3TZ8J0TkXmAD0DuJ8SillFJKKaWUmiDpJnwrgE04VTtPiMjXROTayQtLKaWUUkoppdR4pZXwGWNCxphfGGNeDVwGFACPTWpkSimllFJKKaXGJd0SPkTkBhH5BrAd8AOvTWOZL4nIZhH58pDpnxSRXSLyqIh8KDktX0QeEJEnReTNY9wOpZRSSimllFJDpNVpi4icAHYCvwA+bIzpS2OZy4FcY8x1IvJNEVlnjHkuZZZ/NMZsSnn+t8B9wM+BR0TkZ8aYaLobopRSSimllFLqXOmW8K01xrzKGHNfOsle0lU47f5I/t0w5PXPi8gmEbk0dX5jTALYhdNuUCmllFJKKaVUhkYs4RORfzbGfAH4jIiYoa8bY94/wuJFwLHk4yCwOuW1rxhjPikiy4DvAtcl5+9Omb94mHjuBu4GWLBgwUihK6WUUkoppdQFb7QqnQeSf7dmsO4unM5dSP7tGnjBGNOR/HtERIbOHx46f8py9+IMAE9tbe3zElCllFJKKaWUUmeNmPAZYx5IPtxtjNkxxnVvAd6F0+7vVuD7Ay+ISIExpltEylJi2ALcIiK/AC4FDo3x/ZRSSimllFJKpUi3Dd9/i8hBEfkPEVk9+uxgjNkOhEVkM2AbY54Vka8mX/6iiDwJPADck5z2f8Abgc3Ad40xkfQ3QymllFJKKaXUUGn10mmMuUlEqnCGYrhXRAqAnxtjPj3Kch8Y8vx9yb/vGmbebuCl6QaulFJKKaWUUmpkaY/DZ4xpMsZ8BXg3zhAN/zZZQSmllFJKKaWUGr+0Ej4RuSg5WPpe4GvAU8C8SY1MKaWUUkoppdS4pFWlE/gezqDoLzDGNExiPEoppZRSSimlJsioCZ+IuIBjxpgvT0E8SimllFJKKaUmyKhVOo0xCaBURLxTEI9SSimllFJKqQmSbpXOU8CTIvJ7oG9gojHmvyclKqWUUkoppZRS45ZuwteQ/GcB+ZMXjlJKKaWUUkqpiZLuOHyfmuxAlFJKKaWUUkpNrLQSPhF5BDBDpxtjbp7wiJRSSimllFJKTYh0q3T+U8pjP/AaID7x4SillFJKKaWUmijpVuncNmTSkyLy2CTEo5RSSimllFJqgqRbpbMk5akF1AJVkxKRUkoppZRSSqkJkW6Vzm2cbcMXB04C75iMgJRSSimllFJKTYwREz4RWQecNsYsSj5/C077vZPA/kmPTimllFJKKaVUxqxRXv9fIAogItcD/wn8AAgC905uaEoppZRSSimlxmO0Kp0uY0xH8vHrgHuNMb8Gfi0iOyc1MqWUUkoppZRS4zJaCZ9LRAaSwluAh1NeG7X9n4h8SUQ2i8iXh0z/hIhsSf67JTntrSJySEQeFZEvjGUjlFJKKaWUUko932gJ333AYyLyOyAEbAYQkaU41TrPS0QuB3KNMdcB3mR7wAE/NMZcBbwY+ETK9C8aY240xvzzGLdDKaWUUkoppdQQIyZ8xpjPAP8IfB+41hgz0FOnBbxvlHVfBWxKPt4EbEhZ74nkwwhne/8E+KCIPD5Q6qeUUkoppZRSKnOjVss0xjw9zLTDaay7CDiWfBwEVg8zzydxOoYB+C3wQ6AU+IuI1BpjEmm8j1JKKaWUUkqpYYxWpXM8uoCC5OOC5PNBIvIqoNQY81MAY0yXMcY2xrQCh4HKoSsUkbtFZKuIbG1tbZ3E0JVSSimllFJq5pvMhG8LTkcvALcCgyWFIrIG+Pvkv4FpBcm/AWAZ8LyMzhhzrzGm1hhTW15ePomhK6WUUkoppdTMN2kJnzFmOxAWkc2AbYx5VkS+mnz5izgleH9OdggD8A8isgV4FPicMSY2WbEppZRSSiml1IVg1DZ842GM+cCQ5+9L/n3hMPN+CvjUZMajlFJKKaWUUheSyazSqZRSSimllFJqGmnCp5RSSimllFKzlCZ8SimllFJKKTVLacKnlFJKKaWUUrOUJnxKKaWUUkopNUtpwqeUUkoppZRSs5QmfEoppZRSSik1S2nCp5RSSimllFKzlCZ8SimllFJKKTVLacKnlFJKKaWUUrOUJnxKKaWUUkopNUtpwqeUUkoppZRSs5QmfEoppZRSSik1S2nCp5RSSimllFKzlCZ8SimllFJKKTVLacKnlFJKKaWUUrOUJnxKKaWUUkopNUtNasInIl8Skc0i8uUh0+eIyMMi8pSI3Jqcli8iD4jIkyLy5smMSymllFJKKaUuBO7JWrGIXA7kGmOuE5Fvisg6Y8xzyZfvAT4O7Ab+AGwC/ha4D/g58IiI/MwYEx3pPWru2Tj4+OTnbp+ErZgas3k7Zsu2zQbX/dtGTiePqNR9cbq9n58/e4rf725kXkmA65eV09oT4a3X1NDaG2V3XRdt/VGau0LsOtPFsoo8CnI8VOb7ec8NS2npDfHuH+/A57boDcc43taH20DITEzcArgAlwCWIEA0YSjJFQJeL119cUKxBHEDhV4IRp07WRdV59HZF6UvHCXP78MIxBI26xeVMKcowI66Ljr6ouT63Fwyt5A7Lp/P1x87iktgeWUB/dEYBTke4nFADC4R4rahpTtCrs/FhsVlzC8NsGlfM/VdIRaX53LDigpWVhUAEI/b/GxrHdtOdbKquoA3bliI3+1iX0M3bpdwUXVBWts/1mPolZ/ZyM6e9OcHuOX/PcKJtn7+9toaPvLS1aPOv/VoE2/+wQ5Kc71svueWtN7jqs88REtvlH9/+UW88arFo85/qDHIf2w8SG1NIR+8dWVa7zEV3nDv03SGovz8HespyPMB4z/PXfLRjfTYmS+fGsPqSj8b/yG9fZIqGrf57c56igIeXrC6KqMYHj3Ywi+3neZ16+Zz/fKKjNbxhQcP0N4b5V9fupo8/9gvUcKxBHvrg1QW+JlfkgPAvY8e5bMPHhp1WQ+QAIpzPayZU8Cp9j7OBMN4XFBdEODNVy3i4vlFXDK3kJbuML/YeoZoLMFrauextCIfgINN3UTjNhfPKeSPexrZ0xDkTesXMK8kd8T3jiVs9tQHyfW66InEae2JUFOay0XVBYOvFQY8zC3087tdDZTn+7l5ZWafcTaaLdcKA9vxxisq+Myd66Y5mszMln2hzjVpCR9wFU4iR/LvBmAg4VsDfMAYY0SkR0Tyk/P/vTEmISK7gBXAnnTf7BO/3Min7px5X8xFKQfWTFYzzHbsqQ9SPQ2xqOGdTrl9UnPPRsqSjz/34AEe3NdEwoa6zhBbT3ZSGPDwzMkOFpXm8tzJTnoiMfoiCQAOt/ThtqA4x0s4ZrNxTwOnO8PnvFdsAuM2QByIGyBxNots6zPQFzln3mByG21gX2NvyvSz8/1hTzNWcp4B+xt7+N3OBiLxBHEbHjnUgohgJf/ZdgLLchFL2BjAEmHbqU5sA3Ud/URiCXweF43BMHdtqGFpRR6/2HaGbzxyjLbeCJsPt9IXTXD98nIeO9QKgNsSllXmj7jtmRxDA8keOPt5tB/s7zx+hGNt/QD87xMn00r4XvudbdgG+rvCvPn/tvDDd1414vx//+OtNPY4O+djvzuQVsL3zh9uo7k7zHMnO7hiQQnXZZhATKS7f/AcTx1vB+CV33yKhz9804Sc53pSvoyXfHQjez47tt+yS/717Pl3X3N4hDnP78dPn+LP+5oAKPB72LCkdMzr+PCvdhGOJXj2RAfPffy2MS//rceO8qMtpwAIhmJ86021Y17HpgPNHGnuxWUJb7umBiCtZA/Onrfa+2I8cqR9cHo0AUfbQnz6jwe444p5tPdG2LinkccPt2HbNodaevjS6y6jtSfCn/Y4n+H+hm6++egx4rbNwcYevv+2dYjIed/7qWPtbD/Vyan2PhK2oaUnwsVzC/mbK+fT0BVm26nOZCw225OPi3M8XLageIyfUPaZLdcKqddBP9nWwmfunMZgJkjqtYKa2SazSmcR0J18HARSz0ouY4wZ8tpI8wMgIneLyFYR2dra2joZMSulLkDnvwwb+3IjXNPNSpZ1gW2wUnLusS/I84778ZwHZMjf89FDTymVrsks4esCBuosFSSfD0ikPB54bWD+8DDzA2CMuRe4F6C2tta0pbw2E0v3AE587vZhS8dmmpPDbMclcwtpO8/8auqtLoZ9zo1hTn7udmprPwHAPS+6iCWlufxudwMLSnK4ZmkZHX1R3nTVQlp7o9TWFNPeF6WxK8TuM10sLc+jMMdLVUGAd1+/mLs2zOfdP96B323RF4lxtLUPr4G+iazSKc4/sQTMkCqd/XFC0eGrdAb7ovSEo+QHfIhAJGFYX1NCdZGfHaeDdPRGnlel0y2wvLqAUCROfsBNwgZjnIsr2xiagudW6Xx4fwunO/tZWpHHdcvLWVKeB8Brr5iHMYYdpztZUZk/WKXTY1l43KOX7kFmx9Cl+YypSuc7rl/GT589M1ilMx2/eMcVvPkHOyjP8/H9t28Ydf6v31XL9pQqnen43ttq+eTvD1BbU5gVpXsA975lHW+492mC4Sj3vX09MDHnuWIvdCZLp8daugew5z/Onn9r545cdfB87tqwkAK/m6KczEr3AL54x1p+vf0Md9bOy2j5d9+wlO5QbLBKZyZuvaiS6kI/lQV+8v0eAD76ohVjqtJZkuthzdxCTrb1ciYYweMyzCkI8OarF7F6nlOlc3V1ITWlp4nEE7zm8vkU+D0U+D28+JKqwSqdeT43e+uD3HXVghFL9wCuXlJKgd9NrrfqbJXOslyWVuRTU5pLvt9NQUqVzsoCP2vnz/zSPZg91wqp10FvvCI7zlnjlXqtoGY2OVvQNsErdtrwvcsY8y4R+QbwfWPMs8nXvoLTXm83sNEYc6OIfAhoBH4BPALcZoyJnGf1iEgrcCr5tAxm/PliNmwDnLsdlwPbmZ3bNpMNbMds2z+ZyPZt132U/ds+0/bRTIhzImMc2D8Tvd6Jlq2xTUVcM+0YGs1s246ZcgyNxWzbjoXGmPKRZpy0hA8g2Tvn5cAuY8x7ReSrxpj3icg84IdAAPiEMeYvIlIA/BQoAe41xnx/DO+z1Rgz9sr+WWQ2bAMMvx2zedtmoqHbMVu2KxMzZdtnSpyTYaZsu8Y5cSYrxmze9myNbSrjytbPYKxm83bM5m2bicayHZNZpRNjzAeGPH9f8u8Z4OYhr3UDL53MeJRSSimllFLqQqIDryullFJKKaXULDVbEr57pzuACTAbtgGG347ZvG0z0dDtmC3blYmZsu0zJc7JMFO2XeOcOJMVYzZve7bGNpVxZetnMFazeTtm87bNRGlvx6S24VNKKaWUUkopNX1mSwmfUkoppZRSSqkhNOFTSimllFJKqVlKEz6llFJKKaWUmqUmdViGqSQi64wxz013HGMhIquBhDHmYMq09caYZ6YxrDERkSuA00A7zrAaoeS4irlAMdBljOmdzhgVpO4PYAWwIeX508aYrdMWnHqe5HF1Qe8jEbkYuBg4lq3ndj3PTZzJ+Cz1OMpus3n/zKZrUmAvep7LGpmeK2dcpy0iMlyppAAPGmNum+p4MiUi/wVUAnGgFHi7MaZVRB42xtw88tLZQUS+g/PZR4ByoAHnS3gL8BzQDRQA+cBnjTGbpinUMRORDxpj/kdE1gJfBQzODZJ7jDGbpze69InIzcC/4uyLbuBqwAf8H7AZZ//cinOSf/90xTkVRMQFvJIhFxjAb40x8emL7Fwi8iWcfbQJCHJh7aMHjTEvEpEP4pxHNgLXAPXGmHumNbgUwxxXWXmemwnnscn6LLP5OBKRPODdOOeiIs6ei/7XGNNzIcSVzftnLGbzNSlwCfBr4HGy+Dw3mplwHkzHuM+VxpgZ9Q/oBx4GHkn+HXjcPt2xjXE7Hkt5vAZ4FFgHPDzdsWW4DXuSf59InZ6clgs8Od3xjnHbHk7+/QuwNPm4bAZuxxNATsrzx4fbH8Dj0x3rFHwWPwI+DFwOLAEuSz7/8XTHls6+uED20cBx9xhgpUx/YrpjGxLnOcdVclrWnedmwnlssj7LbD6OgN8DdwIlgAvnBtSdwAMXSlzZvH/GuB2z+Zp0J/DokPmy7jyXxrZl/Xkwze0Y17lyJlbpPAC8yhgTTJ0oIg9NUzyZcouI1xgTNcbsFpFXAT8GVk93YGOQ+v35aPJvBOfOYKpLgPBUBDSBSpJ3U0qMMUcBjDFtIjKzisSd/XEJMFBNeCtO4pMvIi/AuUN0C7B9esKbUjXGmDcNmbZDRLLtDt9WEfkWzp3vgbt4F8o+WiUiP8RJyH1AKDndP30hDWvocQXZeZ6bCeexyfoss/k4KgV+bYyxk887ReTXwAenLyRgauPK5v0zFrP5mvQIMH/IfNl4nhvNTDgPpmNc58qZWKWzGufOSXTIdLfJompZoxGRK4GTxpiWlGku4E5jzM+mL7L0Jet7HzTGJFKmLQC+hlPMbAEJYDfwRWNM/bQEmgER+UTK0y8bY7pEJB9nO949XXGNVfJ4uQfnjt3A/mjEqZMvOFV2thhjdkxXjFNFRD4M3IBz53LgAuMGYLMx5gvTGNrziMhlwFWcrVZ1oeyjhSlPG4wxsWQ1s+uMMX+arriGOs9xlXXnuZlwHpvMzzJbjyMReQNO1cndOOeiQpybvd82xvzkQokrW/fPWMzya9K5OM0//GTxeW40M+E8mI7xnitnXMKnlFKZEpEy4ErgCuAocNTMsIb1SqmZT0TcwHKcpKoLOJINCUK2xqWUGh8dlkFNCRH5ynTHMBFE5MvTHcNEmC3bMRbJDkHacC5m1uNc0LxfRD43vZGp2WKmnOdmwvE/Uz7LTCRr87wCeBvwjuS/VyaTLY1LZbXZcmzOhPNgOtLdDi3hUxNutnS1PFu2YzgicqUx5tnpjmMqDfSAKyKPATcNtFMRkSeMMddOc3hqhpkp54fzdLW+wRjz9DSGNayZMBTHRBCRHwF7eH4PlWuNMXdpXCpbzJTz3Ghmy3YMJ93rOU34JpiIJHBOmG6cxrxvMcb0n2feTwK9xpj/N3URTq4RulqOG2M+MJ2xjcUs2o5Z0WX0RBCRJpxeum4GlhljQsnpW40xtdMa3AwlIh8D3oDTlsAG3gX8LfDfxpj9ItJrjMkbZrkNwJdxjjEf8HNjzCenLPBxminnh5kw/M8IQ3GcMcZ8ZFqDmyQistkYc12606dKtsY10VKu0wb8zBiTVk0PEbkR+CdjzEvH8f6PJtcx5oRDRL4P/MEY86tM338M7zUjznOjmUXbMa7rOS2mn3ghY8ylACLyE5wG0P89rRFNrSuMMdcPmfYbEXl8WqLJ3GzZjl6ccZQEZ+wZko/XTFtE02d98u+/4lwAD4w79a/TFtEMJiJXAS8FLjfGRJLtI73GmHemsfgPgNcaY3Ylq5GtmMxYJ8FMOT/UGmNuABCRNcAvk50XZRNv8u+rOFvy/i0ReWIaY5psvxeRP/D8DqR+P51B8fy4CoHrgQemM6hJMHidNtWS57uZYqac50YzW7ZjXNdzmvBNrs0kd4SIvBn4J5ydtHto9/Ai8rfA3Tg/fkeBNxlj+kXkTuATOHfQg8aY65NVdL6XnNcCXmOMOTJF2zSa2dLV8mzZjtnSZfS4GWNODTOtF8ia3h9nmGqgzRgTAaeba3j+3etkKdNNQCfwemNMK1CB01ssyV5+9yfn/STOkAxzcboD/4Ix5ttTt0lpmynnh5kw/M9MGYpjwhhjvigiW3C6VO8BzuDcBFmcBXF9D6djq0LgNODPtl6MJ4uInAR+inO+8uBck/0nsBSnJ8RvJWctEJHf4Nyoehx4jzHGFpFv4oynHAB+ZYz5RMp6vwu8AKcX84H3s3Cu5U7jXOd9DrgR5zj4ujHmf0VEcAYLvxk4gXOBP1VmynluNLNlO8Z1PadVOifYQBWmZCPnXwMP4pwQ7geuSY79UWKM6Uit0ikipcaY9uQ6Pg00G2O+KiJ7gBcZY+pFpCjZnexXceof/0REvIBroHpaNpgNXS3D7NiO2dJltMo+ydLRJ4AcnB/SnxtjHktN+MQZ5+iu5Lnq34AKY8x7k4//Aack4UHgB8aYcPKc+Cqctha5wA5gvTGmYYo3b1Qz4fwgM2D4H5khQ3FMpORNkAqcG7lZU9VWnDFJU0sOAFYB+4YpIZmxhqnS+Z/GmJ8nE7PPG2O+mawGeAtO9WI/zmdQkazS+SDO53Iq+fh/jTG/Srm2cwF/Bd6fvNFyEvjGQOKcPEfeA3wA2GuM+YyI3I1zfvy0iPiAJ3EGvb8M+DvgRTjVs/cD75yKKp3JWLP+PJeO2bAd472e0xK+iRcQkZ3Jx5uB7+C0a/nVwB1wY0zHMMtdnEz0ioA84M/J6U8C3xeRX+AkjQBbgI+JyDzg/iwq3QMgeRDNqANpOLNhO4wxjeeZrsmeGhdjTG+yIfx1OHfEfy4i9wyZzQZ+nnz8Y5LnMGPMvyervL8Apw3g3+Dc2Qb4XfIGVkhEHsEpbfjtJG5KRmbC+WG4hvzJEtWsSPbggi15z9aqtr/BqZX0fWPMowAi8idjzIunNaqJN1KVzoFqtXuAPGNMD9AjImERKUq+9qwx5jiAiNwHXAv8CnhtMnFz49SAWIUzThqcPQ8O+F/gF8aYzySfvwBYIyJ3JJ8XAstwqtTelzxuG0Tk4Uw2OFMz4TyXjtmwHeO9ntNhGSZeyBhzafLf+5KZeGp92/P5PvBeY8wlwKdIVmcxzqCQH8ep3rQzWRL4U+DlOFVf/iwiWdH4XqnJJiKvEhEjIiunOxblJA/GmEeTVZfeC7xmtEVSlj1mjPkmzl30tSJSOnSe8zxXaqZzJ2vnYIzZjVOq/UmmuaqtMea/caoxrhKRn4nIy6cznmkSSf61Ux4PPB8oJHneOUpEFuE027nFGLMGp/Oh1GrJfUOWeQq4SUQG5hHgfSnXj4uMMX85z/spNWaa8E2Nv+Lc+SkFEJGSYebJBxpFxAO8cWCiiCwxxjxjjPk3oA2YLyKLgePGmK/g3I26EDvgUBemv8GpRvj66Q7kQiciK0RkWcqkS3GqOKWygIE71m/A2XeIyO3Jting3MVO4FSzAXiFiPiT58sbgVnbPb+6YP0DTm0eAIwxnTg3cae9x8Bke89vAHfhVDfdNc0hZaMrRWRRsg3e63DOawU4SV1QRCqB0UpFvwP8Ead0141Tq+vvkteAiMhyEcnFaRL0ehFxJav03TQ5m6RmO63SOQWMMftE5DPAY8m64zuAtw6Z7V+BZ3AumPbgJIAAX0xeVAlO4rgLp+73XSISA5r4/+3dW4zcZRnH8e9PRUQbG0iEcFDqsSmFVFglAgbXXmC4IGJASm0NTYuaRpLGRLwwTdxI1CuqtkBjVS4gahFN00IiLR5aUEuRtUuQKL2wRSgxIAkeMAQojxfPs7vDdGZ3W3dnZmd/n2SS2f3//9N3t9mZ9/2/zwG+PuM/xByhFmXmI2Jfd0dlMJYzdgn5gbcDGKoP3FvICncHyQXG7ZVPMUBWyJ1H3ixZ1S4kwo7LPGBThTm9Shab+jwZ2jTqRWCxpGGyHPay+v5ngW9L+m9duyIijtQa8GHy7vi7gJt6MX9vttAxtAma4ustIEvCnzs9I5ybZkmo7atkQZF+1Jh6A1nWvjkcfSJ7yQIr55ELsm1VtGU/8DjwVzIdZ0IRsUHSfOBO8kb/AuCPdTPsOeBKMsx2Kfl3fADYcwzj7AszMS+r3etzYortOCZ5rZbth3qNi7aYFWWZ+Q3AYLy+zLwnnD1A0kqybPsaSb8nQwjfA6wm2wOcSk5qPwdsJz8YP1nFEJYBn4iI1d0ZvU2F+rA3aTc1TkQqZ3K4wvYmu65lEQAv+Mysk/6feVmnitPNlgWfQzrNxh1VZj4inpE0IGmPpGFJOyWdLmm+pCckLYRM3Fa21rCZs5zxO+Bb6+uPAndHxGsR8XfgN3V8IXAucH/dyV0PnNXZ4Zr1lAeB90m6QtI+Sfsl/bLCz5A0JGmLpF3AHZJOk7RN0qP1uLhe542Svi/pcUm7JJ3UtZ/IzPpdu3nZoVr8IelDysqnrd7H9ilbmVHHd9ecbpWkW2oud6iihZD0VklPSTpB0nsl3VdzvwdHawdUOO9eSX+QdFOHfx/HzQs+s3G7yBzJA5Juk/SxiqffBFwdEQNkL51vRPZBuYGsoHotcHL0Zr+wvlD5XEuBHyhLXN9Ihge260kksoz2aAL8eRFxWWdGa8crIoa8uzf9KkfocjIs7LfARyLifPLGyVcaTh0gd8U/A2wE9kTEEuACMlQNMufy1ohYTOZdTlaox6ZI0hFJIw2PBd0ek1mXHTUvm8I1je9jW4FrYKytwRkRMTx6Ys3lHiXTQgCuAHZGxCvAFrKQzgBZkOe2Oue7wOaI+DCZVjUreMFnVqoU+ACZh/QcWUb5C7TZKYqI+8kJ1K3A9V0Y8lxyNXBHRJwdEQsi4p1kzt4/gKskvaF2Kgbr/CeAd1Q4CHW3rpeaTZt1wmiu0iPA38hCEWeR1Z0fI2+cNP5d7Ijxnq5Lgc0wVo11tNnvwYgYqefDZN6RTY/GKt8fjIhDk12g5Lmc9aVW8zJJqya5rPF97KdkP0PIhd/dLc6/i/H88mvr35gHXEwW1Rkh22icXudcAvyknt95LD9PN7loi1mDSpzfDeyuCdEXyZ2ii5rPrQ/ZRWR7jFOApzs41LlmOZkk3+jn5O//aeBPZEL7PuCfEfGysp/RxkqKfxPwHcZ3KczmgqP6jUnaBGyIiB3KJtJDDYebS8e30liq/gjgkM4ZUpPO7cDJwAnA+ojYXjt/vyBD2C8CrpR0DTmhPZEsIvK17ozabHq1mJddRxb7Gr3R8ZamS15suPawpOeV/S6XkTfxm+0AvqWsoD8A/Bp4G/DCBP0aZ10BFN8VMitqXWb+z7TfKfpSHV8O3F7hnzYDImIwIu5r+t7GiFgLfDkizgHWAB8gd12JiJGIuDQilkTEYofcmgHZ0PlwPb9ugvN+BawFUJaEf/tMD8xyR7Ye24CXgE9FxAVkdeKbpbF2JgvJqIfz6/n7gQvJz60BSZd2fvhm06vNvOxJ4BC5OIPJw8pHQ9fnR8RjzQdrF/FhMlTz3opo+BdwUNKnaxyStKQu+R3jraFWNL9er/IOn9m4dmXmt9C0U6RsiXE9cGFE/FvSA2S4p++qdt699X/2ZrKM/6yJqTfrgiEyTOkw8BDw7jbnrQO2SFpD7uStBdzWZGa9bke2biJ+sxZvrwFnAqfV4Scj4qF6flk99tfX88gF4AOdGLTZDGo3L1sE/FDSV8nInon8jFzMTVRg5S4y3HOw4XsrgM2S1pM77FvJfL91wI8lrSMjjWYFt2UwMzMz6zI1lXevXKXLgZUR8UoVrBqsw2PtMSTdDByIiO91dsRmNls4pNPMzMys98wHnq3F3seBs9uctxNYXTl/SDpT0qmdGqSZ9T6HdJqZmZn1nh8B90h6BBgB/tLqpIjYJWkRsLdS/P4DrASe7dA4zazHOaTTzMzMzMysTzmk08zMzMzMrE95wWdmZmZmZtanvOAzMzMzMzPrU17wmZmZmZmZ9Skv+MzMzMzMzPqUF3xmZmZmZmZ9ygs+MzMzMzOzPuUFn5mZmZmZWZ/6H2ob12c0Iq4aAAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<Figure size 1080x1080 with 64 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# scatter plot matrix\n",
    "scatter_matrix(titanic_new, figsize=[15,15])\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "dbe7785f",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 1311.88x1260 with 56 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.pairplot(titanic_new, hue='Survived')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 141,
   "id": "adb74078",
   "metadata": {},
   "outputs": [],
   "source": [
    "counts=titanic_new.groupby(['Survived','Sex'])['Survived'].count().values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 146,
   "id": "37bbf566",
   "metadata": {},
   "outputs": [],
   "source": [
    "label = ['Male Not-Survived','Female Not-Survived','Male Survived','Female Survived']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 147,
   "id": "afab1b39",
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.rcParams['figure.figsize'] = [6,6]  # the whole figure\n",
    "plt.rcParams[\"axes.titlesize\"] = 20     # for the top title\n",
    "plt.rcParams[\"xtick.labelsize\"] = 15    # for lables\n",
    "plt.rcParams[\"font.size\"] = 20          # for percentages"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 148,
   "id": "967419a0",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x432 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "explode = [0.05]*len(counts)\n",
    "colors = ['red','pink','green','royalblue']\n",
    "plt.title(\"Survival Rates by Gender\")\n",
    "plt.axis(\"equal\")\n",
    "plt.pie(counts, labels=label, autopct='%1.0f%%', startangle=-60, explode=explode, colors = colors, shadow=True)\n",
    "plt.plot();"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "bd51c853",
   "metadata": {},
   "source": [
    "## C. Modeling\n",
    "\n",
    "Now it is time to create some models for the data and estimate their accuracy on unseen data.\n",
    "\n",
    "Let's split our train and test data. I'll use 80/20 ratio. (train/test)\n",
    "\n",
    "I'll also fix the random state to some number so that we generate the same results every time."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 149,
   "id": "6e82a53d",
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "#This is our X values, and the last column will be th target values, Y.\n",
    "X = titanic_new.iloc[:,:-1]\n",
    "Y = titanic_new.iloc[:,-1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 150,
   "id": "0fb5f8b0",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Split the data\n",
    "X_train, X_test, Y_train, Y_test = \\\n",
    "    train_test_split(X, Y, train_size=0.8, random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 151,
   "id": "c1cc71ce",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "# of Test cases :  179 \n",
      "# of Train cases:  712\n"
     ]
    }
   ],
   "source": [
    "print('# of Test cases : ', len(X_test), '\\n# of Train cases: ', len(X_train))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8867220b",
   "metadata": {},
   "source": [
    "## 1. Gaussian Naive Bayes\n",
    "\n",
    "This will be our first model."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 152,
   "id": "01087ae7",
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "#clf_1 means the first classifier.\n",
    "clf_1 = GaussianNB()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 153,
   "id": "be322757",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0.86111111 0.79166667 0.69014085 0.92957746 0.83098592 0.76056338\n",
      " 0.73239437 0.76056338 0.69014085 0.87323944]\n",
      "\n",
      " Average accuracy of the model on the training data is: 79%\n"
     ]
    }
   ],
   "source": [
    "#we're using 10-fold cross validation\n",
    "accuracy_1_train = cross_val_score(clf_1, X_train, Y_train, scoring='accuracy', cv = 10)\n",
    "print(accuracy_1_train)\n",
    "\n",
    "#get the mean of accuracy scores\n",
    "print(\"\\n Average accuracy of the model on the training data is: {0:.0%}\".format(accuracy_1_train.mean()))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "65b0575c",
   "metadata": {},
   "source": [
    "The score is not that good. Let's see the score on the test data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 154,
   "id": "d467e50d",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Let's fit the model\n",
    "clf_1.fit(X_train, Y_train)\n",
    "\n",
    "#Let's get the predictions on the test data first.\n",
    "Y_pred_1 = clf_1.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 155,
   "id": "1beee6d5",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy on the test data: 78%\n"
     ]
    }
   ],
   "source": [
    "accuracy_1_test = accuracy_score(Y_pred_1, Y_test)\n",
    "\n",
    "print(\"Accuracy on the test data: {0:.0%}\".format(accuracy_1_test))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e54dbe64",
   "metadata": {},
   "source": [
    "Let's take a look at the confusion matrix."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 156,
   "id": "431dc295",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Confusion Matrix:\n",
      "[[82 23]\n",
      " [17 57]]\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x432 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.83      0.78      0.80       105\n",
      "           1       0.71      0.77      0.74        74\n",
      "\n",
      "    accuracy                           0.78       179\n",
      "   macro avg       0.77      0.78      0.77       179\n",
      "weighted avg       0.78      0.78      0.78       179\n",
      "\n"
     ]
    }
   ],
   "source": [
    "print(\"Confusion Matrix:\")\n",
    "cm = confusion_matrix(Y_test, Y_pred_1)\n",
    "print(cm)\n",
    "plt.matshow(cm)\n",
    "plt.title('Confusion matrix')\n",
    "plt.colorbar()\n",
    "plt.ylabel('True label')\n",
    "plt.xlabel('Predicted label')\n",
    "plt.show()\n",
    "print(classification_report(Y_test, Y_pred_1))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b886e660",
   "metadata": {},
   "source": [
    "Out of 74 survived, we predict 57 correctly. Out of 105 not survived, we predict 82 of them correctly."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b6a45870",
   "metadata": {},
   "source": [
    "## 2. Logistic Regresssion\n",
    "\n",
    "Now, let's try logistic regression."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 157,
   "id": "8b41d66a",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\ismai\\anaconda3\\lib\\site-packages\\sklearn\\linear_model\\_logistic.py:763: ConvergenceWarning: lbfgs failed to converge (status=1):\n",
      "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.\n",
      "\n",
      "Increase the number of iterations (max_iter) or scale the data as shown in:\n",
      "    https://scikit-learn.org/stable/modules/preprocessing.html\n",
      "Please also refer to the documentation for alternative solver options:\n",
      "    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression\n",
      "  n_iter_i = _check_optimize_result(\n"
     ]
    }
   ],
   "source": [
    "clf_2 = linear_model.LogisticRegression(C=1e5)\n",
    "clf_2.fit(X_train, Y_train)\n",
    "Y_pred_2 = clf_2.predict(X_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4f75d241",
   "metadata": {},
   "source": [
    "We are getting Convergence Warning: lbfgs failed to converge. \"TOTAL NO. of ITERATIONS REACHED LIMIT.\" To fix this, we have couple of options.\n",
    "\n",
    "  1) Increase the max no. of iterations. Default max_iter=100, but we can try different numbers like 200, 300, etc.\n",
    "  \n",
    "  2) We can use a different solver. Default solver='lbfgs'. But we 4 more solvers {‘newton-cg’, ‘liblinear’, ‘sag’, ‘saga’}\n",
    "  \n",
    "  3) We can also try normalizing the data using standard scaler.\n",
    "\n",
    "Let's start with the 1st option."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 158,
   "id": "578a6bf3",
   "metadata": {},
   "outputs": [],
   "source": [
    "clf_2 = linear_model.LogisticRegression(C=1e5, max_iter=500)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4ec3525d",
   "metadata": {},
   "source": [
    "We tried 200, 300, and 400 but it didn't work. 500 did not give us any warning."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 159,
   "id": "6b35b4dd",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0.84722222 0.79166667 0.71830986 0.92957746 0.84507042 0.73239437\n",
      " 0.74647887 0.71830986 0.73239437 0.92957746]\n",
      "\n",
      " Average accuracy of the model on the training data is: 80%\n"
     ]
    }
   ],
   "source": [
    "#we're using 10-fold cross validation\n",
    "accuracy_2 = cross_val_score(clf_2, X_train, Y_train, scoring='accuracy', cv = 10)\n",
    "print(accuracy_2)\n",
    "\n",
    "#get the mean of accuracy scores \n",
    "print(\"\\n Average accuracy of the model on the training data is: {0:.0%}\".format(accuracy_2.mean()))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 160,
   "id": "3f4261e4",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Let's fit the model\n",
    "clf_2.fit(X_train, Y_train)\n",
    "\n",
    "#Let's get the predictions on the test data first.\n",
    "Y_pred_2 = clf_2.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 161,
   "id": "ca353517",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy on the test data: 82%\n"
     ]
    }
   ],
   "source": [
    "accuracy_2 = accuracy_score(Y_pred_2, Y_test)\n",
    "\n",
    "print(\"Accuracy on the test data: {0:.0%}\".format(accuracy_2))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c3169497",
   "metadata": {},
   "source": [
    "The accuracy is a little bit more than Gaussian NB but still low.\n",
    "\n",
    "Let's use different solvers, the 2nd option."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 162,
   "id": "4b4f35d4",
   "metadata": {},
   "outputs": [],
   "source": [
    "clf_2 = linear_model.LogisticRegression(C=1e5, solver='liblinear')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5ce41f3d",
   "metadata": {},
   "source": [
    "solver='sag' or 'saga' was still giving a warning. But, newton-cg or liblinear seems fine."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 163,
   "id": "efbe8d25",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0.84722222 0.79166667 0.71830986 0.92957746 0.84507042 0.73239437\n",
      " 0.74647887 0.71830986 0.73239437 0.92957746]\n",
      "\n",
      " Average accuracy of the model on the training data is: 80%\n"
     ]
    }
   ],
   "source": [
    "accuracy_2 = cross_val_score(clf_2, X_train, Y_train, scoring='accuracy', cv = 10)\n",
    "print(accuracy_2)\n",
    "\n",
    "print(\"\\n Average accuracy of the model on the training data is: {0:.0%}\".format(accuracy_2.mean()))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 164,
   "id": "b6774ac2",
   "metadata": {},
   "outputs": [],
   "source": [
    "clf_2.fit(X_train, Y_train)\n",
    "Y_pred_2 = clf_2.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 165,
   "id": "2400a92a",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy on the test data: 82%\n"
     ]
    }
   ],
   "source": [
    "accuracy_2 = accuracy_score(Y_pred_2, Y_test)\n",
    "\n",
    "print(\"Accuracy on the test data: {0:.0%}\".format(accuracy_2))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e6644ca5",
   "metadata": {},
   "source": [
    "All of the scores are the same!"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "54b2a11b",
   "metadata": {},
   "source": [
    "Let's use the last option: normalizing the data using standard scaler. We'll see it will increase the accuracy or not."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 166,
   "id": "90c90247",
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "#We'll make a pipeline first.\n",
    "clf_2 = make_pipeline(StandardScaler(), linear_model.LogisticRegression(C=1e5))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5b3e969d",
   "metadata": {},
   "source": [
    "We didn't get any warning, that's good."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 167,
   "id": "a3429c2a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0.84722222 0.79166667 0.71830986 0.92957746 0.84507042 0.73239437\n",
      " 0.74647887 0.71830986 0.73239437 0.92957746]\n",
      "\n",
      " Average accuracy of the model on the training data is: 80%\n"
     ]
    }
   ],
   "source": [
    "accuracy_2 = cross_val_score(clf_2, X_train, Y_train, scoring='accuracy', cv = 10)\n",
    "print(accuracy_2)\n",
    "\n",
    "#get the mean of accuracy scores \n",
    "print(\"\\n Average accuracy of the model on the training data is: {0:.0%}\".format(accuracy_2.mean()))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 168,
   "id": "56f2206e",
   "metadata": {},
   "outputs": [],
   "source": [
    "clf_2.fit(X_train, Y_train)\n",
    "Y_pred_2 = clf_2.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 169,
   "id": "571e7c0a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy on the test data: 82%\n"
     ]
    }
   ],
   "source": [
    "accuracy_2 = accuracy_score(Y_pred_2, Y_test)\n",
    "\n",
    "print(\"Accuracy on the test data: {0:.0%}\".format(accuracy_2))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "466c5eca",
   "metadata": {},
   "source": [
    "Same score everywhere! So, any one of the three options can be used.\n",
    "\n",
    "Let's see the confusion matrix of the test data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 170,
   "id": "557df481",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Confusion Matrix:\n",
      "[[92 13]\n",
      " [19 55]]\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x432 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.83      0.88      0.85       105\n",
      "           1       0.81      0.74      0.77        74\n",
      "\n",
      "    accuracy                           0.82       179\n",
      "   macro avg       0.82      0.81      0.81       179\n",
      "weighted avg       0.82      0.82      0.82       179\n",
      "\n"
     ]
    }
   ],
   "source": [
    "print(\"Confusion Matrix:\")\n",
    "cm = confusion_matrix(Y_test, Y_pred_2)\n",
    "print(cm)\n",
    "plt.matshow(cm)\n",
    "plt.title('Confusion matrix')\n",
    "plt.colorbar()\n",
    "plt.ylabel('True label')\n",
    "plt.xlabel('Predicted label')\n",
    "plt.show()\n",
    "print(classification_report(Y_test, Y_pred_2))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "36cbcf3b",
   "metadata": {},
   "source": [
    "A little bit of improvement: Out of 74 survived, we predict 55 correctly. Out of 105 not survived, we predict 92 of them correctly."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "aa917561",
   "metadata": {},
   "source": [
    "## 3. SVC (Support Vector Classifier) Models\n",
    "\n",
    "We will try out 4 different SVM models:\n",
    "\n",
    "\"LinearSVC\"......\"SVC with linear kernel\"......\"SVC with quadratic polynomial kernel\"......\"SVC with RBF kernel\""
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1e702ed0",
   "metadata": {},
   "source": [
    "### 3.1 Linear SVC"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 171,
   "id": "eb54a6f6",
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "C = 1.0   # regularization parameter\n",
    "\n",
    "clf_3_1 = svm.LinearSVC(C=C, dual=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 172,
   "id": "d9af620b",
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0.83333333 0.77777778 0.71830986 0.94366197 0.83098592 0.70422535\n",
      " 0.77464789 0.71830986 0.77464789 0.92957746]\n",
      "\n",
      " Average accuracy of the model on the training data is: 80%\n"
     ]
    }
   ],
   "source": [
    "accuracy_3_1 = cross_val_score(clf_3_1, X_train, Y_train, scoring='accuracy', cv = 10)\n",
    "print(accuracy_3_1)\n",
    "\n",
    "print(\"\\n Average accuracy of the model on the training data is: {0:.0%}\".format(accuracy_3_1.mean()))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 173,
   "id": "efce0ec0",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy on the test data: 80%\n"
     ]
    }
   ],
   "source": [
    "clf_3_1.fit(X_train, Y_train)\n",
    "Y_pred_3_1 = clf_3_1.predict(X_test)\n",
    "\n",
    "accuracy_3_1 = accuracy_score(Y_pred_3_1, Y_test)\n",
    "\n",
    "print(\"Accuracy on the test data: {0:.0%}\".format(accuracy_3_1))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f44531e3",
   "metadata": {},
   "source": [
    "### 3.2 SVC with Linear Kernel"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 174,
   "id": "c8ba0d67",
   "metadata": {},
   "outputs": [],
   "source": [
    "clf_3_2 = svm.SVC(kernel=\"linear\", C=C)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 175,
   "id": "d0e9294c",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0.83333333 0.75       0.69014085 0.94366197 0.81690141 0.67605634\n",
      " 0.73239437 0.76056338 0.76056338 0.88732394]\n",
      "\n",
      " Average accuracy of the model on the training data is: 79%\n"
     ]
    }
   ],
   "source": [
    "accuracy_3_2 = cross_val_score(clf_3_2, X_train, Y_train, scoring = \"accuracy\", cv=10)\n",
    "print(accuracy_3_2)\n",
    "print(\"\\n Average accuracy of the model on the training data is: {0:.0%}\".format(accuracy_3_2.mean()))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 176,
   "id": "570bbd92",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy on the test data: 78%\n"
     ]
    }
   ],
   "source": [
    "clf_3_2.fit(X_train, Y_train)\n",
    "Y_pred_3_2 = clf_3_2.predict(X_test)\n",
    "\n",
    "accuracy_3_2 = accuracy_score(Y_pred_3_2, Y_test)\n",
    "\n",
    "print(\"Accuracy on the test data: {0:.0%}\".format(accuracy_3_2))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d3bf8804",
   "metadata": {},
   "source": [
    "### 3.3 SVC with Quadratic Polynomial Kernel"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 177,
   "id": "a6b6a438",
   "metadata": {},
   "outputs": [],
   "source": [
    "clf_3_3 = svm.SVC(kernel=\"poly\", degree=2, C=C)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 178,
   "id": "8a5b1184",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0.68055556 0.68055556 0.69014085 0.66197183 0.63380282 0.66197183\n",
      " 0.63380282 0.64788732 0.61971831 0.66197183]\n",
      "\n",
      " Average accuracy of the model on the training data is: 66%\n"
     ]
    }
   ],
   "source": [
    "accuracy_3_3 = cross_val_score(clf_3_3, X_train, Y_train, scoring = \"accuracy\", cv=10)\n",
    "print(accuracy_3_3)\n",
    "print(\"\\n Average accuracy of the model on the training data is: {0:.0%}\".format(accuracy_3_3.mean()))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 179,
   "id": "f00b2049",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy on the test data: 64%\n"
     ]
    }
   ],
   "source": [
    "clf_3_3.fit(X_train, Y_train)\n",
    "Y_pred_3_3 = clf_3_3.predict(X_test)\n",
    "\n",
    "accuracy_3_3 = accuracy_score(Y_pred_3_3, Y_test)\n",
    "\n",
    "print(\"Accuracy on the test data: {0:.0%}\".format(accuracy_3_3))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "342a18b6",
   "metadata": {},
   "source": [
    "### 3.4 SVC with RBF (Radial Basis Function) Kernel"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 180,
   "id": "385808ed",
   "metadata": {},
   "outputs": [],
   "source": [
    "clf_3_4 = svm.SVC(kernel='rbf')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 181,
   "id": "b7b99725",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0.69444444 0.68055556 0.69014085 0.70422535 0.64788732 0.63380282\n",
      " 0.64788732 0.64788732 0.61971831 0.70422535]\n",
      "\n",
      " Average accuracy of the model on the training data is: 67%\n"
     ]
    }
   ],
   "source": [
    "accuracy_3_4 = cross_val_score(clf_3_4, X_train, Y_train, scoring= \"accuracy\", cv=10)\n",
    "print(accuracy_3_4)\n",
    "print(\"\\n Average accuracy of the model on the training data is: {0:.0%}\".format(accuracy_3_4.mean()))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 182,
   "id": "d3deff5f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy on the test data: 66%\n"
     ]
    }
   ],
   "source": [
    "clf_3_4.fit(X_train, Y_train)\n",
    "Y_pred_3_4 = clf_3_4.predict(X_test)\n",
    "\n",
    "accuracy_3_4 = accuracy_score(Y_pred_3_4, Y_test)\n",
    "\n",
    "print(\"Accuracy on the test data: {0:.0%}\".format(accuracy_3_4))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5385074f",
   "metadata": {},
   "source": [
    "## 4. Decision Tree Classifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 183,
   "id": "e67671df",
   "metadata": {},
   "outputs": [],
   "source": [
    "clf_4 = DecisionTreeClassifier(max_depth=4, criterion='entropy', max_features=0.6, splitter='best')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 184,
   "id": "fce3c64f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0.81944444 0.77777778 0.73239437 0.91549296 0.88732394 0.70422535\n",
      " 0.81690141 0.81690141 0.76056338 0.91549296]\n",
      "\n",
      " Average accuracy of the model on the training data is: 81%\n"
     ]
    }
   ],
   "source": [
    "accuracy_4 = cross_val_score(clf_4, X_train, Y_train, scoring= \"accuracy\", cv=10)\n",
    "print(accuracy_4)\n",
    "print(\"\\n Average accuracy of the model on the training data is: {0:.0%}\".format(accuracy_4.mean()))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 185,
   "id": "a64c488c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy on the test data: 80%\n"
     ]
    }
   ],
   "source": [
    "clf_4.fit(X_train, Y_train)\n",
    "\n",
    "Y_pred_4 = clf_4.predict(X_test)\n",
    "\n",
    "accuracy_4 = accuracy_score(Y_pred_4, Y_test)\n",
    "\n",
    "print(\"Accuracy on the test data: {0:.0%}\".format(accuracy_4))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ec8123ef",
   "metadata": {},
   "source": [
    "## 5. Random Forest Classifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 186,
   "id": "e9ced4c8",
   "metadata": {},
   "outputs": [],
   "source": [
    "clf_5 = RandomForestClassifier(n_estimators=100, random_state=100)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 187,
   "id": "9cf6e1ec",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0.81944444 0.77777778 0.74647887 0.84507042 0.84507042 0.76056338\n",
      " 0.77464789 0.77464789 0.81690141 0.84507042]\n",
      "\n",
      " Average accuracy of the model on the training data is: 80%\n"
     ]
    }
   ],
   "source": [
    "accuracy_5 = cross_val_score(clf_5, X_train, Y_train, scoring= \"accuracy\", cv=10)\n",
    "print(accuracy_5)\n",
    "print(\"\\n Average accuracy of the model on the training data is: {0:.0%}\".format(accuracy_5.mean()))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 188,
   "id": "e8b55c3b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy on the test data: 83%\n"
     ]
    }
   ],
   "source": [
    "clf_5.fit(X_train, Y_train)\n",
    "\n",
    "Y_pred_5 = clf_5.predict(X_test)\n",
    "\n",
    "accuracy_5 = accuracy_score(Y_pred_5, Y_test)\n",
    "\n",
    "print(\"Accuracy on the test data: {0:.0%}\".format(accuracy_5))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "644bf88a",
   "metadata": {},
   "source": [
    "## 6. Linear Discriminant Analysis (LDA)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 189,
   "id": "25cabfc7",
   "metadata": {},
   "outputs": [],
   "source": [
    "clf_6 = LinearDiscriminantAnalysis()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 190,
   "id": "72d157d5",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0.80555556 0.77777778 0.71830986 0.94366197 0.83098592 0.67605634\n",
      " 0.77464789 0.73239437 0.76056338 0.92957746]\n",
      "\n",
      " Average accuracy of the model on the training data is: 79%\n"
     ]
    }
   ],
   "source": [
    "accuracy_6 = cross_val_score(clf_6, X_train, Y_train, scoring= \"accuracy\", cv=10)\n",
    "print(accuracy_6)\n",
    "print(\"\\n Average accuracy of the model on the training data is: {0:.0%}\".format(accuracy_6.mean()))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 191,
   "id": "07ab0a93",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy on the test data: 79%\n"
     ]
    }
   ],
   "source": [
    "clf_6.fit(X_train, Y_train)\n",
    "\n",
    "Y_pred_6 = clf_6.predict(X_test)\n",
    "\n",
    "accuracy_6 = accuracy_score(Y_pred_6, Y_test)\n",
    "\n",
    "print(\"Accuracy on the test data: {0:.0%}\".format(accuracy_6))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "15ca1c3a",
   "metadata": {},
   "source": [
    "## 7. KNN"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 192,
   "id": "1887d351",
   "metadata": {},
   "outputs": [],
   "source": [
    "clf_7 = KNeighborsClassifier(n_neighbors=6)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 193,
   "id": "a95edb26",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0.63888889 0.66666667 0.67605634 0.73239437 0.74647887 0.70422535\n",
      " 0.61971831 0.70422535 0.63380282 0.73239437]\n",
      "\n",
      " Average accuracy of the model on the training data is: 69%\n"
     ]
    }
   ],
   "source": [
    "accuracy_7 = cross_val_score(clf_7, X_train, Y_train, scoring= \"accuracy\", cv=10)\n",
    "print(accuracy_7)\n",
    "print(\"\\n Average accuracy of the model on the training data is: {0:.0%}\".format(accuracy_7.mean()))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 194,
   "id": "112e55c3",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy on the test data: 75%\n"
     ]
    }
   ],
   "source": [
    "clf_7.fit(X_train, Y_train)\n",
    "\n",
    "Y_pred_7 = clf_7.predict(X_test)\n",
    "\n",
    "accuracy_7 = accuracy_score(Y_pred_7, Y_test)\n",
    "\n",
    "print(\"Accuracy on the test data: {0:.0%}\".format(accuracy_7))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c84a4736",
   "metadata": {},
   "source": [
    "## D. Test Data and Submissions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 195,
   "id": "16c1c784",
   "metadata": {},
   "outputs": [],
   "source": [
    "titanic_test = pd.read_csv('Test.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 196,
   "id": "68260a25",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "PassengerId      0\n",
       "Pclass           0\n",
       "Name             0\n",
       "Sex              0\n",
       "Age             86\n",
       "SibSp            0\n",
       "Parch            0\n",
       "Ticket           0\n",
       "Fare             1\n",
       "Cabin          327\n",
       "Embarked         0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 196,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "titanic_test.isnull().sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8d7679be",
   "metadata": {},
   "source": [
    "Let's fill out the missing Fare value with mean"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 201,
   "id": "1a17a1e0",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "35.6271884892086"
      ]
     },
     "execution_count": 201,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "titanic_test.Fare.mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 202,
   "id": "0472bc25",
   "metadata": {},
   "outputs": [],
   "source": [
    "titanic_test['Fare'] = titanic_test['Fare'].fillna(35.62)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a1998d59",
   "metadata": {},
   "source": [
    "Let's convert nonnumeric categorical values to numeric ones."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 204,
   "id": "b94cb50c",
   "metadata": {},
   "outputs": [],
   "source": [
    "titanic_test['Sex'] = titanic_test['Sex'].map({'female': 1, 'male': 0})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 205,
   "id": "6824efb7",
   "metadata": {},
   "outputs": [],
   "source": [
    "titanic_test['Embarked'] = titanic_test['Embarked'].map({'C':0, 'Q':1, 'S': 2})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 207,
   "id": "02ad3c95",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Embarked</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>892</td>\n",
       "      <td>3</td>\n",
       "      <td>Kelly, Mr. James</td>\n",
       "      <td>0</td>\n",
       "      <td>34.5</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>330911</td>\n",
       "      <td>7.8292</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>893</td>\n",
       "      <td>3</td>\n",
       "      <td>Wilkes, Mrs. James (Ellen Needs)</td>\n",
       "      <td>1</td>\n",
       "      <td>47.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>363272</td>\n",
       "      <td>7.0000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>894</td>\n",
       "      <td>2</td>\n",
       "      <td>Myles, Mr. Thomas Francis</td>\n",
       "      <td>0</td>\n",
       "      <td>62.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>240276</td>\n",
       "      <td>9.6875</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>895</td>\n",
       "      <td>3</td>\n",
       "      <td>Wirz, Mr. Albert</td>\n",
       "      <td>0</td>\n",
       "      <td>27.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>315154</td>\n",
       "      <td>8.6625</td>\n",
       "      <td>NaN</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>896</td>\n",
       "      <td>3</td>\n",
       "      <td>Hirvonen, Mrs. Alexander (Helga E Lindqvist)</td>\n",
       "      <td>1</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>3101298</td>\n",
       "      <td>12.2875</td>\n",
       "      <td>NaN</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>413</th>\n",
       "      <td>1305</td>\n",
       "      <td>3</td>\n",
       "      <td>Spector, Mr. Woolf</td>\n",
       "      <td>0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>A.5. 3236</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>414</th>\n",
       "      <td>1306</td>\n",
       "      <td>1</td>\n",
       "      <td>Oliva y Ocana, Dona. Fermina</td>\n",
       "      <td>1</td>\n",
       "      <td>39.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>PC 17758</td>\n",
       "      <td>108.9000</td>\n",
       "      <td>C105</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>415</th>\n",
       "      <td>1307</td>\n",
       "      <td>3</td>\n",
       "      <td>Saether, Mr. Simon Sivertsen</td>\n",
       "      <td>0</td>\n",
       "      <td>38.5</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>SOTON/O.Q. 3101262</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>416</th>\n",
       "      <td>1308</td>\n",
       "      <td>3</td>\n",
       "      <td>Ware, Mr. Frederick</td>\n",
       "      <td>0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>359309</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>417</th>\n",
       "      <td>1309</td>\n",
       "      <td>3</td>\n",
       "      <td>Peter, Master. Michael J</td>\n",
       "      <td>0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2668</td>\n",
       "      <td>22.3583</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>418 rows × 11 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     PassengerId  Pclass                                          Name  Sex  \\\n",
       "0            892       3                              Kelly, Mr. James    0   \n",
       "1            893       3              Wilkes, Mrs. James (Ellen Needs)    1   \n",
       "2            894       2                     Myles, Mr. Thomas Francis    0   \n",
       "3            895       3                              Wirz, Mr. Albert    0   \n",
       "4            896       3  Hirvonen, Mrs. Alexander (Helga E Lindqvist)    1   \n",
       "..           ...     ...                                           ...  ...   \n",
       "413         1305       3                            Spector, Mr. Woolf    0   \n",
       "414         1306       1                  Oliva y Ocana, Dona. Fermina    1   \n",
       "415         1307       3                  Saether, Mr. Simon Sivertsen    0   \n",
       "416         1308       3                           Ware, Mr. Frederick    0   \n",
       "417         1309       3                      Peter, Master. Michael J    0   \n",
       "\n",
       "      Age  SibSp  Parch              Ticket      Fare Cabin  Embarked  \n",
       "0    34.5      0      0              330911    7.8292   NaN         1  \n",
       "1    47.0      1      0              363272    7.0000   NaN         2  \n",
       "2    62.0      0      0              240276    9.6875   NaN         1  \n",
       "3    27.0      0      0              315154    8.6625   NaN         2  \n",
       "4    22.0      1      1             3101298   12.2875   NaN         2  \n",
       "..    ...    ...    ...                 ...       ...   ...       ...  \n",
       "413   NaN      0      0           A.5. 3236    8.0500   NaN         2  \n",
       "414  39.0      0      0            PC 17758  108.9000  C105         0  \n",
       "415  38.5      0      0  SOTON/O.Q. 3101262    7.2500   NaN         2  \n",
       "416   NaN      0      0              359309    8.0500   NaN         2  \n",
       "417   NaN      1      1                2668   22.3583   NaN         0  \n",
       "\n",
       "[418 rows x 11 columns]"
      ]
     },
     "execution_count": 207,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "titanic_test"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "405181ef",
   "metadata": {},
   "source": [
    "We will use the same method above to fill out missing age values."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 356,
   "id": "6e3a64a1",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Embarked</th>\n",
       "      <th>Age</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>40.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>41.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>41.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>37.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>42.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>29.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>53.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>30.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>20.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>25.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10</th>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>22.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11</th>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>24.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>12</th>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>25.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>13</th>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>24.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>14</th>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>26.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>15</th>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>22.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "    Pclass  Sex  Embarked   Age\n",
       "0        1    0         0  40.0\n",
       "1        1    0         2  41.0\n",
       "2        1    1         0  41.0\n",
       "3        1    1         1  37.0\n",
       "4        1    1         2  42.0\n",
       "5        2    0         0  29.0\n",
       "6        2    0         1  53.0\n",
       "7        2    0         2  30.0\n",
       "8        2    1         0  20.0\n",
       "9        2    1         2  25.0\n",
       "10       3    0         0  22.0\n",
       "11       3    0         1  24.0\n",
       "12       3    0         2  25.0\n",
       "13       3    1         0  24.0\n",
       "14       3    1         1  26.0\n",
       "15       3    1         2  22.0"
      ]
     },
     "execution_count": 356,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "fill_age_test = titanic_test[[\"Pclass\", \"Sex\", \"Embarked\", \"Age\"]].groupby([\"Pclass\", \"Sex\", \"Embarked\"], as_index=False).mean().round(0)\n",
    "fill_age_test"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 357,
   "id": "4ba4fbed",
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "for p in [1,2,3]:\n",
    "    for s in [0,1]:    \n",
    "        for e in [0,1,2]:\n",
    "            titanic_test.loc[(titanic_test['Pclass']==p) & (titanic_test['Sex']==s) & (titanic_test['Embarked']==e) & (pd.isnull(titanic_test['Age'])), ['Age']] = fill_age_test.loc[(fill_age_test['Pclass']==p) & (fill_age_test['Sex']==s) & (fill_age_test['Embarked']==e), ['Age']].values"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "fec40019",
   "metadata": {},
   "source": [
    "Let's see if it worked or not."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 358,
   "id": "6d3f350d",
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Embarked</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "Empty DataFrame\n",
       "Columns: [PassengerId, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked]\n",
       "Index: []"
      ]
     },
     "execution_count": 358,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "titanic_test[pd.isnull(titanic_test['Age'])]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 359,
   "id": "c0475afe",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Embarked</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>418.000000</td>\n",
       "      <td>418.000000</td>\n",
       "      <td>418.000000</td>\n",
       "      <td>418.000000</td>\n",
       "      <td>418.000000</td>\n",
       "      <td>418.000000</td>\n",
       "      <td>418.000000</td>\n",
       "      <td>418.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>1100.500000</td>\n",
       "      <td>2.265550</td>\n",
       "      <td>0.363636</td>\n",
       "      <td>29.470096</td>\n",
       "      <td>0.447368</td>\n",
       "      <td>0.392344</td>\n",
       "      <td>35.627171</td>\n",
       "      <td>1.401914</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>120.810458</td>\n",
       "      <td>0.841838</td>\n",
       "      <td>0.481622</td>\n",
       "      <td>13.031378</td>\n",
       "      <td>0.896760</td>\n",
       "      <td>0.981429</td>\n",
       "      <td>55.840500</td>\n",
       "      <td>0.854496</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>892.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.170000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>996.250000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>22.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>7.895800</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>1100.500000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>26.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>14.454200</td>\n",
       "      <td>2.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>1204.750000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>36.875000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>31.500000</td>\n",
       "      <td>2.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>1309.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>76.000000</td>\n",
       "      <td>8.000000</td>\n",
       "      <td>9.000000</td>\n",
       "      <td>512.329200</td>\n",
       "      <td>2.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       PassengerId      Pclass         Sex         Age       SibSp  \\\n",
       "count   418.000000  418.000000  418.000000  418.000000  418.000000   \n",
       "mean   1100.500000    2.265550    0.363636   29.470096    0.447368   \n",
       "std     120.810458    0.841838    0.481622   13.031378    0.896760   \n",
       "min     892.000000    1.000000    0.000000    0.170000    0.000000   \n",
       "25%     996.250000    1.000000    0.000000   22.000000    0.000000   \n",
       "50%    1100.500000    3.000000    0.000000   26.000000    0.000000   \n",
       "75%    1204.750000    3.000000    1.000000   36.875000    1.000000   \n",
       "max    1309.000000    3.000000    1.000000   76.000000    8.000000   \n",
       "\n",
       "            Parch        Fare    Embarked  \n",
       "count  418.000000  418.000000  418.000000  \n",
       "mean     0.392344   35.627171    1.401914  \n",
       "std      0.981429   55.840500    0.854496  \n",
       "min      0.000000    0.000000    0.000000  \n",
       "25%      0.000000    7.895800    1.000000  \n",
       "50%      0.000000   14.454200    2.000000  \n",
       "75%      0.000000   31.500000    2.000000  \n",
       "max      9.000000  512.329200    2.000000  "
      ]
     },
     "execution_count": 359,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "titanic_test.describe()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9b72b972",
   "metadata": {},
   "source": [
    "It worked. Now let's rearrange columns."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 360,
   "id": "6a9d6194",
   "metadata": {},
   "outputs": [],
   "source": [
    "titanic_test_X = titanic_test[['Pclass',  'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 361,
   "id": "219e1c8e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Embarked</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>34.5</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7.8292</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>47.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>7.0000</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>62.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>9.6875</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>27.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>8.6625</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>12.2875</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>413</th>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>25.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>414</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>39.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>108.9000</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>415</th>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>38.5</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>416</th>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>25.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>417</th>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>22.3583</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>418 rows × 7 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     Pclass  Sex   Age  SibSp  Parch      Fare  Embarked\n",
       "0         3    0  34.5      0      0    7.8292         1\n",
       "1         3    1  47.0      1      0    7.0000         2\n",
       "2         2    0  62.0      0      0    9.6875         1\n",
       "3         3    0  27.0      0      0    8.6625         2\n",
       "4         3    1  22.0      1      1   12.2875         2\n",
       "..      ...  ...   ...    ...    ...       ...       ...\n",
       "413       3    0  25.0      0      0    8.0500         2\n",
       "414       1    1  39.0      0      0  108.9000         0\n",
       "415       3    0  38.5      0      0    7.2500         2\n",
       "416       3    0  25.0      0      0    8.0500         2\n",
       "417       3    0  22.0      1      1   22.3583         0\n",
       "\n",
       "[418 rows x 7 columns]"
      ]
     },
     "execution_count": 361,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "titanic_test_X"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "064727f4",
   "metadata": {},
   "source": [
    "I made 7 submissions, one for each model."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8ff1a131",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Gaussioan NB 75%\n",
    "titanic_test_Y_pred_1 = clf_1.predict(titanic_test_X)\n",
    "#Logistic Regression 76%\n",
    "titanic_test_Y_pred_2 = clf_2.predict(titanic_test_X)\n",
    "#Linear SVC 76%\n",
    "titanic_test_Y_pred_3 = clf_3_1.predict(titanic_test_X)\n",
    "#Decision Tree 77%\n",
    "titanic_test_Y_pred_4 = clf_4.predict(titanic_test_X)\n",
    "#Random Forest 76%\n",
    "titanic_test_Y_pred_5 = clf_5.predict(titanic_test_X)\n",
    "#LDA 76%\n",
    "titanic_test_Y_pred_6 = clf_6.predict(titanic_test_X)\n",
    "#KNN 65%\n",
    "titanic_test_Y_pred_7 = clf_7.predict(titanic_test_X)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "aaa49390",
   "metadata": {},
   "source": [
    "The higest score one was the decision tree."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 398,
   "id": "eb9b3135",
   "metadata": {},
   "outputs": [],
   "source": [
    "titanic_test_Y_pred_4 = clf_4.predict(titanic_test_X)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 399,
   "id": "2c7cb57d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1,\n",
       "       1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1,\n",
       "       0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0,\n",
       "       0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0,\n",
       "       0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,\n",
       "       0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n",
       "       0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0,\n",
       "       0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n",
       "       0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,\n",
       "       0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0,\n",
       "       1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0,\n",
       "       1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1,\n",
       "       1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0,\n",
       "       0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,\n",
       "       0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0,\n",
       "       0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,\n",
       "       0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0,\n",
       "       1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,\n",
       "       0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0],\n",
       "      dtype=int64)"
      ]
     },
     "execution_count": 399,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "titanic_test_Y_pred_4"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 400,
   "id": "166b9f23",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>892</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>893</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>894</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>895</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>896</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>413</th>\n",
       "      <td>1305</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>414</th>\n",
       "      <td>1306</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>415</th>\n",
       "      <td>1307</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>416</th>\n",
       "      <td>1308</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>417</th>\n",
       "      <td>1309</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>418 rows × 2 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     PassengerId  Survived\n",
       "0            892         0\n",
       "1            893         0\n",
       "2            894         0\n",
       "3            895         0\n",
       "4            896         0\n",
       "..           ...       ...\n",
       "413         1305         0\n",
       "414         1306         1\n",
       "415         1307         0\n",
       "416         1308         0\n",
       "417         1309         0\n",
       "\n",
       "[418 rows x 2 columns]"
      ]
     },
     "execution_count": 400,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "submission_df_4 = pd.DataFrame({\n",
    "                  \"PassengerId\": titanic_test['PassengerId'], \n",
    "                  \"Survived\": titanic_test_Y_pred_4})\n",
    "submission_df_4"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 401,
   "id": "e7c44d86",
   "metadata": {},
   "outputs": [],
   "source": [
    "submission_df_4.to_csv('submission_4.csv', index=False)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "36f0637e",
   "metadata": {},
   "source": [
    "I hope this code helps. My  submission to the competition site Kaggle results in scoring 4,259 of 13,892 competition entries."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bd725cc6",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
