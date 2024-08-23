import streamlit as st
import pandas as pd
import requests
st.title('Prédiction feux de forêt USA 🔥')

from collections import Counter
from imblearn.ensemble import EasyEnsembleClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score
from imblearn.over_sampling import RandomOverSampler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler,  ClusterCentroids
from imblearn.metrics import classification_report_imbalanced, geometric_mean_score
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold, cross_val_score
from sklearn import model_selection
from sklearn import tree
from sklearn.metrics import recall_score
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
import warnings
import csv

warnings.filterwarnings("ignore")

df = pd.read_csv('Firesclean.csv',index_col=0)
st.dataframe(df.head(5))

st.sidebar.title("Sommaire")
pages=["Contexte et présentation du projet", "Jeu de donnée et preprocessing", "DataVizualization", "Modélisation", "Conclusion"]
page=st.sidebar.radio("Aller vers", pages)

if page == pages[0] : 
  st.write("### Contexte et présentation du projet")
  st.write("### Jeu de donnée et preprocessing")
  if st.checkbox("Afficher les na") :
    st.dataframe(df.isna().sum())
      
