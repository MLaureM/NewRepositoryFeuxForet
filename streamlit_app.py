import streamlit as st
import pandas as pd
import requests
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(layout="wide")
st.title('Prédiction feux de forêt USA 🔥')

from collections import Counter
from imblearn.ensemble import EasyEnsembleClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler,  ClusterCentroids
from imblearn.metrics import classification_report_imbalanced, geometric_mean_score

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold, cross_val_score
from sklearn import model_selection
from sklearn import tree
from sklearn.metrics import recall_score, f1_score, confusion_matrix, classification_report
from sklearn.datasets import make_classification
from xgboost import XGBClassifier
import xgboost as xgb
import warnings
import csv
import dropbox
from io import BytesIO
warnings.filterwarnings("ignore")

#from google.colab import drive
#drive.mount('/content/drive')
#pd.read_csv('/content/drive/My Drive/20082024 Projet Datascientest Feux de forêt USA')


#import dropbox
# Configuration de l'accès à Dropbox
#DROPBOX_ACCESS_TOKEN = 'sl.B7knc246PEcajtdw4_VJQrM4DcPH7catk9JW5M5kOAIdqYZbKUUDRZOW7cQ4eElOOY8H1NlAc7qWLn9nuNa3TZSzKuxY9TqumGpH9xaTVpqdRzJ2YXBwvPE43G5-GJbA4MdokNTGQn2kga3uUhvVw-0'
#dbx = dropbox.Dropbox(DROPBOX_ACCESS_TOKEN)
# Fonction pour télécharger le fichier CSV depuis Dropbox
#def download_csv_from_dropbox(dropbox_path, local_path):
#    with open(local_path, "wb") as f:
#        metadata, res = dbx.files_download(path=dropbox_path)
#        f.write(res.content)
# Chemin du fichier CSV sur Dropbox et chemin local temporaire
#dropbox_path = '/Firesclean.csv'
#local_path = 'temp.csv'
# Télécharger le fichier CSV
#download_csv_from_dropbox(dropbox_path, local_path)
# Charger le fichier CSV dans un DataFrame
#df1 = pd.read_csv(local_path)
# Afficher le DataFrame avec Streamlit
#st.title('Affichage du DataFrame')
#st.write(df1)

@st.cache_data(persist=True)
def load_data():
  data=pd.read_csv('Firesclean.csv',index_col=0)
  return data
df=load_data()
st.sidebar.title("Sommaire")
pages=["Contexte et présentation du projet", "Jeu de données et preprocessing", "DataVizualization", "Prédiction causes de feux", "Prédiction classes de feux", "Conclusion"]
page=st.sidebar.radio("Aller vers", pages)

if page == pages[0] : 
  st.write("### Contexte et présentation du projet")
  if st.checkbox("Afficher jeu donnée") :
    st.dataframe(df.head(5))

      
if page == pages[1] : 
  st.write("### Jeu de données et preprocessing")
  if st.checkbox("Afficher jeu données") :
    st.write("### Jeu de données et statistiques")
    st.dataframe(df.head(5))
    st.write("### statistiques")
    st.dataframe(df.describe(), use_container_width=True)
  if st.checkbox("Afficher la dimension") :
     st.write(f"La dimension : {df.shape}")
  if st.checkbox("Afficher les na") :
    st.dataframe(df.isna().sum(), width=300, height=640)

if page == pages[2] : 
  st.write("### DataVizualisation")

if page == pages[3] : 
  st.write("### Prédiction causes de feux")
  # Création de la variable DURATION en jours
  # Suppression des variables non utiles au ML et utilisation de l'année de feu comme index
  Drop_col_ML = ["FPA_ID", "DISCOVERY_DATE", "DISCOVERY_DOY", "CONT_DOY", "FIRE_SIZE"] #"OBJECTID", "FOD_ID", "DISCOVERY_TIME", "CONT_DATE", "CONT_TIME", "Shape", "STAT_CAUSE_DESCR"
  Fires35 = df.dropna()
  Fires_ML = Fires35.drop(Drop_col_ML, axis = 1)
  # Suppression des lignes de "COUNTY", "AVG_TEMP [°C]", "AVG_PCP [mm]" ayant des données manquantes 
  Fires_ML = Fires_ML.dropna(subset = ["STATE", "AVG_TEMP [°C]", "AVG_PCP [mm]"])
  if st.checkbox("Afficher jeu données pour Machine learning") :
    st.dataframe(Fires_ML.head(5))

  st.divider()
  # Distribution des causes de feux
  #with st.container():
  st.write("### Distribution initiale des causes de feux")
  col1, col2= st.columns([1, 2])
  with col1:
      count = Fires_ML["STAT_CAUSE_DESCR"].value_counts()
      color = ["blue", "magenta", "orange", "yellow", "magenta", "blue", "blue", "blue", "blue", "blue", "blue", "blue", "blue"]
      fig, ax = plt.subplots(figsize=(15, 10)) 
      ax.bar(count.index, count.values, color=color)
      ax.set_ylabel("COUNT", fontsize=20)
      ax.set_xticklabels(count.index, rotation=45, fontsize=20)
      ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
      st.pyplot(fig)
  with col2:
      st.write()
  st.write("""
  On observe un grand déséquilibre du jeu de données. Ce qui va rendre complexe la prédiction de l'analyse.
  Les feux manquants, non-définis et autres représentent environ le quart des feux. Compte tenu de leur caractère inerte 
  par rapport à l'objectif de l'étude, nous les supprimerons.
  De même, compte tenu des diverses qui peuvent parfois se ressembler, nous procéderons à une fusion des causes qui peuvent 
  être regroupés dans une cause parent.
  """)

  st.divider()
  # Suppression des causes non-définies et fusion des autres
  st.write("### XXXXXXXXXXXXXXXX")
  st.write("Nous allons regrouper les feux en 3 catégories pour réduire le nombre de causes:")
  st.write("**Humaine (20)** : debris burning (5), Campfire (4), Children (8), Smoking (3), Equipment Use (2), Railroad (6), Powerline (11), Structure(12), Fireworks (10)")
  st.write("**Criminelle (21)** : Arson (7)")
  st.write("**Naturelle (22)** : Ligthning (1)")

  Fires_ML = Fires_ML[(Fires_ML.loc[:, "STAT_CAUSE_CODE"] != 9) & (Fires_ML.loc[:, "STAT_CAUSE_CODE"] != 13)]
  Fires_ML["STAT_CAUSE_CODE"] = Fires_ML["STAT_CAUSE_CODE"].replace([3, 4, 5, 8, 2, 6, 10, 11, 12], 20)
  Fires_ML["STAT_CAUSE_CODE"] = Fires_ML["STAT_CAUSE_CODE"].replace(7, 21)
  Fires_ML["STAT_CAUSE_CODE"] = Fires_ML["STAT_CAUSE_CODE"].replace(1, 22)
  Fires_ML["STAT_CAUSE_CODE"] = Fires_ML["STAT_CAUSE_CODE"].replace({20: 0, 21: 1, 22: 2})

  st.divider()
  # Nouvelle distribution des causes suite au regroupement des causes initiales
  st.write("### Distribution des causes après regroupement")
  col1, col2= st.columns([1, 2])
  with col1:
      count2 = Fires_ML["STAT_CAUSE_CODE"].value_counts()
      color = ["blue", "orange", "yellow"]
      fig, ax = plt.subplots(figsize=(15, 13))  # Adjust the figure size as needed
      ax.bar(count.index, count2.values, color=color)
      ax.set_ylabel("COUNT", fontsize=20)
      ax.set_xticklabels([0, 1, 2], ["Humaine", "Criminelle", "Naturelle"], rotation = 30, fontsize=20)
      ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
      st.pyplot(fig)
  with col2:
      st.write()

if page == pages[4] : 
  st.write("### Prédiction classes de feux")
  Fires34=df.dropna()
  FiresML2= Fires34.loc[:,['MONTH_DISCOVERY','FIRE_SIZE_CLASS','STAT_CAUSE_DESCR','AVG_TEMP [°C]','AVG_PCP [mm]','LONGITUDE','LATITUDE','STATE']]
  FiresML2['FIRE_SIZE_CLASS'] = FiresML2['FIRE_SIZE_CLASS'].replace({"A":0,"B":0,"C":0,"D":1,"E":1,"F":1,"G":1})
  if st.checkbox("Afficher jeu données pour Machine learning") :
    st.dataframe(FiresML2.head(5))
  
  feats = FiresML2.drop('FIRE_SIZE_CLASS', axis=1)
  target = FiresML2['FIRE_SIZE_CLASS'].astype('int')
  X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=0.25, random_state = 42,stratify=target)
  num_train=X_train.drop(['STAT_CAUSE_DESCR','MONTH_DISCOVERY','STATE'],axis=1)
  num_test=X_test.drop(['STAT_CAUSE_DESCR','MONTH_DISCOVERY','STATE'],axis=1)
  sc = StandardScaler()
  num_train= sc.fit_transform(num_train)
  num_test= sc.transform(num_test)
  oneh = OneHotEncoder(drop = 'first', sparse_output=False)
  cat_train=X_train.drop(['AVG_TEMP [°C]','AVG_PCP [mm]','MONTH_DISCOVERY','LONGITUDE','LATITUDE'],axis=1)
  cat_test=X_test.drop(['AVG_TEMP [°C]','AVG_PCP [mm]','MONTH_DISCOVERY','LONGITUDE','LATITUDE'],axis=1)
  cat_train=oneh.fit_transform(cat_train)
  cat_test= oneh.transform(cat_test)
  circular_cols = ['MONTH_DISCOVERY']
  circular_train = X_train[circular_cols]
  circular_test = X_test[circular_cols]
  circular_train['MONTH_DISCOVERY'] = circular_train['MONTH_DISCOVERY'].apply(lambda h : np.sin(2 * np.pi * h / 12))
  circular_train['MONTH_DISCOVERY'] = circular_train['MONTH_DISCOVERY'].apply(lambda h : np.cos(2 * np.pi * h / 12))
  circular_test['MONTH_DISCOVERY'] = circular_test['MONTH_DISCOVERY'].apply(lambda h : np.sin(2 * np.pi * h /12))
  circular_test['MONTH_DISCOVERY'] = circular_test['MONTH_DISCOVERY'].apply(lambda h : np.cos(2 * np.pi * h / 12))
  X_train=np.concatenate((num_train,cat_train,circular_train),axis=1)
  X_test=np.concatenate((num_test,cat_test,circular_test), axis=1)
  le = LabelEncoder()
  y_train = le.fit_transform(y_train)
  y_test = le.transform(y_test)
  
  classifier=st.selectbox("classificateur",("XGBoost","BalancedRandomForest"))

  if classifier == "XGboost":
    st.sidebar.subheader("Hyperparamètres du modèle XGBoost")
  max_bin_test = st.sidebar.number_input("Max_Bin selection",100, 700, step=10)
  scale_pos_weight_test = st.sidebar.number_input("scale_pos_weight selection",0, 50, step=1)
  subsample_test = st.sidebar.number_input("subsample selection",0.00, 1.00, step=0.01)
  colsample_bytree_test = st.sidebar.number_input("colsample_bytree selection",0.00, 1.00, step=0.01)
  learning_rate_test = st.sidebar.number_input("learning_rate selection",0.00, 1.00, step=0.01)
  tree_method_test = st.sidebar.radio("tree_method selection",("hist","approx"))
  if st.sidebar.button("Execution",key="classify"):
    st.subheader("XGBoost Results")
    model=XGBClassifier(max_bin=max_bin_test,
                        scale_pos_weight=scale_pos_weight_test,
                        subsample=subsample_test,
                        colsample_bytree=colsample_bytree_test,
                        learning_rate=learning_rate_test,
                        tree_method=tree_method_test ).fit(X_train,y_train)
    y_pred=model.predict(X_test)
    #Métriques
    accuracy=model.score(X_test,y_test).round(4)
    #precision=precision_score(y_test,y_pred).round(4)
    recall=recall_score(y_test,y_pred).round(4)

    #Afficher
    st.write("Accuracy",accuracy.round(4))
    #st.write("precision",precision.round(4))
    st.write("recall",recall.round(4))
  sns.histplot(data=FiresML2, x="FIRE_SIZE_CLASS",bins=2,stat="percent",discrete=False)
  plt.show()

  if classifier == "BalancedRandomForest":
    st.sidebar.subheader("Hyperparamètres du modèle BalancedRandomForest")
  min_samples_split_test = st.sidebar.number_input("Min_samples_split selection",1, 10, step=1)
  max_depth_test = st.sidebar.number_input("max_depth selection",0, 100, step=1)
  sampling_strategy_test = st.sidebar.radio("sampling_strategy selection",("not minority","not majority"))
  replacement_test= st.sidebar.radio("replacement selection",(True,False))
  Bootstrap_test= st.sidebar.radio("bootstrap selection",(True,False))
  random_state_test=st.sidebar.number_input("random_state selection",200,200, step=1)
  n_estimator_test=st.sidebar.number_input("n_estimator selection",100,1000, step=10)
  criterion_test=st.sidebar.radio("criterion selection",("entropy","giny"))
  class_weight_test=st.sidebar.radio("class_weight selection",("balanced_subsample","None"))
  max_features_test=st.sidebar.number_input("max_features selection",100,200, step=10)
  if st.sidebar.button("Execution",key="classify2"):
    st.subheader("BalancedRandomForest Results")
    model=BalancedRandomForestClassifier(min_samples_split=min_samples_split_test,
                        max_depth=max_depth_test,
                        sampling_strategy=sampling_strategy_test,
                        replacement=replacement_test,
                        bootstrap=Bootstrap_test,
                        random_state=random_state_test,
                        n_estimators=n_estimator_test,
                        criterion=criterion_test,
                        class_weight=class_weight_test,
                        max_features=max_features_test).fit(X_train,y_train)
    y_pred=model.predict(X_test)
    #Métriques
    accuracy=model.score(X_test,y_test).round(4)
    #precision=precision_score(y_test,y_pred).round(4)
    recall=recall_score(y_test,y_pred).round(4)

    #Afficher
    st.write("Accuracy",accuracy.round(4))
    #st.write("precision",precision.round(4))
    st.write("recall",recall.round(4))
  sns.histplot(data=FiresML2, x="FIRE_SIZE_CLASS",bins=2,stat="percent",discrete=False)
  plt.show()


if page == pages[5] : 
  st.write("### Conclusion")

 