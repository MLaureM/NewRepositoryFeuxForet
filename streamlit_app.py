import streamlit as st
st.set_page_config(layout="wide",)
import pandas as pd
import requests
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
st.title('Prédiction feux de forêt USA 🔥')
from collections import Counter
from imblearn.ensemble import EasyEnsembleClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler,  ClusterCentroids
from imblearn.metrics import classification_report_imbalanced, geometric_mean_score
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.metrics import RocCurveDisplay
from sklearn import svm
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import roc_curve
from sklearn.model_selection import GridSearchCVassifiere
from sklearn import model_selection
from sklearn import tree
from sklearn.metrics import recall_sc, roc_auc_score
from sklearn.metrics import precision_recall_curve, auc
from sklearn.preprocessing import MinMaxScaler, train_test_split, StratifiedKFold, cross_val_scor
from sklearn.ensemble import GradientBoostingCl
import folium
from streamlit_folium import st_folium
from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics
from sklearn.metrics import make_scorer, f1_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
import joblib

# Mise en forme couleur du fond de l'application
page_bg_img="""<style>
[data-testid="stAppViewContainer"]{
background-color: #F4E4AA;
opacity: 1;
background-image:  repeating-radial-gradient( circle at 0 0, transparent 0, #F4E4AA 10px ), repeating-linear-gradient( #F4E4AA55, #F4E4AA );}
[data-testid="stHeader"]{
background-color: #F4E4AA;
opacity: 1;
background-image:  repeating-radial-gradient( circle at 0 0, transparent 0, #F4E4AA 10px ), repeating-linear-gradient( #F4E4AA55, #F4E4AA );}
[data-testid="stSidebarContent"]{
background-color: #F4E4AA;
opacity: 0.8;
background-image:  repeating-radial-gradient( circle at 0 0, transparent 0, #F4E4AA 10px ), repeating-linear-gradient( #F4E4AA55, #F4E4AA );}
[data-testid="stAppViewContainer"] { padding-top: 0rem; }
div[class^='block-container'] { padding-side: 2rem; }
</style>"""
st.markdown(page_bg_img,unsafe_allow_html=True)
st.markdown("""<style>.block-container {padding-top: 3rem;padding-bottom: 2rem;padding-left: 5rem;padding-right: 5rem;}</style>""", unsafe_allow_html=True)
#Chargement dataframe sous alias df
@st.cache_data(persist=True)
def load_data():
  data=pd.read_csv('Firesclean.csv',index_col=0)
  return data
df=load_data()

st.sidebar.title("Sommaire")
pages=["Contexte et présentation", "Preprocessing", "DataVizualization", "Prédiction causes de feux", "Prédiction classes de feux", "Conclusion"]
page=st.sidebar.radio("Aller vers", pages)

# Création contenu de la première page (page 0) avec le contexte et présentation du projet
if page == pages[0] : 
  st.write("### Contexte et présentation du projet")
  #st.image("ImageFeu.jpg")  
  st.write("Nous sommes en réorientation professionnelle et cherchons à approfondir nos compétences en data analysis. Ce projet nous permet de mettre en pratique les méthodes et outils appris durant notre formation, de l’exploration des données à la modélisation et la data visualisation.")
  st.markdown("""
    ### Étapes du projet :
    - **Nettoyage et pré-processing des données**
    - **Storytelling avec DataViz** (Plotly, seaborn, matplotlib)
    - **Construction de modèles de Machine Learning**
    - **Restitution via Streamlit**
    ### Objectif :
    Le projet vise à prédire les incendies de forêt pour améliorer la prévention et l’intervention. Il s’inscrit dans un contexte de préservation de l’environnement et de sécurité publique, avec des impacts économiques significatifs.
    ### Données utilisées :
    Nous utilisons des données provenant du **US Forest Service**, qui centralise les informations sur les incendies de forêt aux États-Unis. Ces données incluent les causes des incendies, les surfaces touchées, et leurs localisations. Nous intégrons également des données météorologiques (vent, température, humidité) provenant du **National Interagency Fire Center** pour évaluer les risques de départ et de propagation des feux.
    ### Applications :
    - Détection précoce des incendies pour cibler les interventions
    - Prévention des incendies criminels et anticipation des feux dus à la foudre.
    """)

#if st.checkbox("Afficher jeu donnée") :
#    st.dataframe(df.head(5))

#Création de la page 1 avec explication du préprocessing     
if page == pages[1] : 
  st.write("### Preprocessing")
  if st.checkbox("Afficher jeu données") :
    st.write("### Jeu de données et statistiques")
    st.dataframe(df.head(5))
    st.write("### statistiques")
    st.dataframe(df.describe(), use_container_width=True)
  if st.checkbox("Afficher la dimension") :
     st.write(f"La dimension : {df.shape}")
  if st.checkbox("Afficher les na") :
    st.dataframe(df.isna().sum(), width=300, height=640)

#Création de la page 2 Datavizualisation
if page == pages[2] : 
  st.header("DataVizualisation")
  st.write("Nous avons analysé le dataset sous différents angles afin d’en faire ressortir les principales caractéristiques.")
  st.subheader("1 - Analyse des outliers et de la répartitions des valeurs numériques")
 
  col1, col2 =st.columns([0.55, 0.45],gap="small",vertical_alignment="center")
  with col1 :
    fig, axes = plt.subplots(2, 3,figsize=(12,7))
    sns.set_theme()
    sns.violinplot(ax=axes[0, 0], x=df['DURATION'])
    sns.violinplot(ax=axes[0, 1],x=df['FIRE_SIZE'])
    sns.violinplot(ax=axes[0, 2],x=df['AVG_PCP [mm]'])
    sns.violinplot(ax=axes[1,0],x=df['LATITUDE'])
    sns.violinplot(ax=axes[1, 1],x=df['LONGITUDE'])
    sns.violinplot(ax=axes[1, 2],x=df['AVG_TEMP [°C]'])
    st.pyplot(fig)
  with col2 :
    st.divider()

    st.markdown("Certaines variables comme les tailles de feux et les durées présentent des valeurs particulièrement extrêmes.")  
    st.markdown("Certaines valeurs extrêmes paraissent impossibles et devront être écartées des analyses (exemple feux de plus de 1 an). A l'inverse les feux de taille extrêmes restent des valeurs possibles (méga feux) et seront partie intégrante de nos analyses")
    
    st.markdown("Les autres variables numériques (précipitations, températures ou coordonnées géographiques) nous indiquent certaines tendances marquées sur la répartition des feux que nous allons analyser plus en détail.")
    
    st.divider()

  st.subheader("2 - Répartition des feux par cause et classe")
  col1, col2 = st.columns([0.55, 0.45],gap="small",vertical_alignment="center")
  with col1 :
    #with st.container(height=400):
    Fires_cause = df.groupby("STAT_CAUSE_DESCR").agg({"FPA_ID":"count", "FIRE_SIZE":"sum"}).reset_index()
    Fires_cause = Fires_cause.rename({"FPA_ID":"COUNT_FIRE", "FIRE_SIZE":"FIRE_SIZE_SUM"}, axis = 1)
    Indic = ["≈ Hawaii + Massachusetts", "≈ Hawaii + Massachusetts", "≈ Washington + Georgia", "≈ Maine", "≈ New Jersey + Massachusetts"]
    Fires_cause["Text"] = Indic
    fig = make_subplots(rows = 1, cols = 2,specs = [[{"type":"domain"}, {"type":"domain"}]])
    fig.add_trace(go.Pie(labels = Fires_cause["STAT_CAUSE_DESCR"],values = Fires_cause["COUNT_FIRE"],hole = 0.6,
        direction = "clockwise", title = dict(text = "Nombre", font=dict(size=20))),row = 1, col = 1,)
    fig.add_trace(go.Pie(labels = Fires_cause["STAT_CAUSE_DESCR"],values = Fires_cause["FIRE_SIZE_SUM"],hovertext = Fires_cause["Text"],
         hole = 0.6,direction = "clockwise",title = dict(text = "Surfaces (acres)", font=dict(size=20))),row = 1, col = 2)
    fig.update_traces(textfont_size=15,sort=False,marker=dict(colors=['#F1C40F', '#F39C12', '#e74c3c','#E67E22','#d35400']))
    fig.update_layout(title_text="Répartition des feux par causes (1992 - 2015)", title_x = 0.2, title_y = 0.95,paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',legend=dict(x=0.5, y=0.83,orientation="h",xanchor="center",yanchor="bottom",font=dict(
            family="Arial",size=13,color="black")),margin=dict(l=0, r=0, t=0, b=0),titlefont=dict(size=15),width=900,height=500)
    st.plotly_chart(fig)
  #Pie Chart répartition par cause
  with col2 :
  #if st.checkbox("Afficher graphiques par cause") :
    st.divider()

    st.markdown(":orange[Les feux d’origine humaine] (volontaire et involontaire) représentent :orange[50% des départs].")
    
    st.markdown(":orange[Les causes naturelles] (foudre) représentent :orange[62,1% des surfaces brûlées].")

    st.divider()
  
  col1, col2 = st.columns([0.55, 0.45],gap="small",vertical_alignment="center")
  with col1 :
    #with st.container(height=400):
    Fires_class = df.groupby("FIRE_SIZE_CLASS").agg({"FPA_ID":"count", "FIRE_SIZE":"sum"}).reset_index()
    Fires_class = Fires_class.rename({"FPA_ID":"COUNT_FIRE", "FIRE_SIZE":"FIRE_SIZE_SUM"}, axis = 1)
    Indic = ["≈ ", "≈ ","≈ Connecticut", "≈ New Jersey", "≈ Maryland", "≈ Virginie Occidentale + Delaware", "≈ Californie + Hawaii"]
    Fires_class["Text"] = Indic
    fig1= make_subplots(rows = 1, cols = 2, specs = [[{"type":"domain"}, {"type":"domain"}]])
    fig1.add_trace(go.Pie(labels = Fires_class["FIRE_SIZE_CLASS"],values = Fires_class["COUNT_FIRE"],
           hole = 0.6, rotation = 0,title = dict(text = "Nombre", font=dict(size=20))),row = 1, col = 1)
    fig1.add_trace(go.Pie(labels = Fires_class["FIRE_SIZE_CLASS"],values = Fires_class["FIRE_SIZE_SUM"],
           hovertext = Fires_class["Text"],hole = 0.6, rotation = -120,title = dict(text = "Surfaces (acres)", font=dict(size=20))),
      row = 1, col = 2)
    fig1.update_traces(textfont_size=15,sort=False,marker=dict(colors=['yellow','brown','#F1C40F', '#F39C12', '#e74c3c','#E67E22','#d35400']))
    fig1.update_layout(title_text="Répartition des feux suivant leur taille (1992 - 2015)", title_x = 0.2, title_y = 0.95,paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',legend=dict(x=0.5, y=0.83,orientation="h",xanchor="center",yanchor="bottom",font=dict(
            family="Arial",size=12,color="black")),margin=dict(l=0, r=0, t=0, b=0),titlefont=dict(size=15),width=900,height=500)
    st.plotly_chart(fig1)
  with col2 :
    st.divider()
    st.write(":orange[Les feux de petite taille (A et B, <9,9 acres)] représentent :orange[62 % du nombre de départs] mais seulement :orange[2% des surfaces brûlées].  :orange[78 % des surfaces brûlées sont liées aux feux de la classe G] (avec des feux allant de 5000 à 600 000 acres).")
    st.divider()

  st.subheader("3 - Répartition temporelle des feux")
  st.write("Cet axe révèle assez clairement des périodes à risque sur les départs et la gravité des feux")
#Histogrammes année
  st.write("Certaines années semblent clairement plus propices aux départs de feux. Cela peut s’expliquer par les conditions météorologiques. On observe notamment que les années où les surfaces brûlées sont significativement supérieures à la moyenne cela est dû à la foudre")
  #if st.checkbox("Afficher graphiques année") :
    #fig2 = make_subplots(rows=1, cols=2, shared_yaxes=False,subplot_titles=("Surfaces brûlées (acres)","Nombre de départs"))
    #fig2.add_trace(go.Histogram(histfunc="sum",
    #  name="Surface brûlées (acres) ",
    #  x=df['FIRE_YEAR'],y=df['FIRE_SIZE'], marker_color='red'),1,1)
    #fig2.add_trace(go.Histogram(histfunc="count",
    #  name="Nombre de feux",
    #  x=df['FIRE_YEAR'],marker_color='blue'),1,2)
    #fig2.update_layout(title_text="Départs de feux par année",bargap=0.2,height=400, width=1100, coloraxis=dict(colorscale='Bluered_r'), showlegend=False,paper_bgcolor='rgba(0,0,0,0)',
    #plot_bgcolor='rgba(0,0,0,0)')
    #st.plotly_chart(fig2)

  df1=df.groupby(['STAT_CAUSE_DESCR', 'FIRE_YEAR']).agg({"FIRE_SIZE":"sum"}).reset_index()
  df1bis=df.groupby(['STAT_CAUSE_DESCR', 'FIRE_YEAR']).agg({"FPA_ID":"count"}).reset_index()
  col1, col2 = st.columns([0.5, 0.5],gap="small",vertical_alignment="center")
  with col1 :
    fig2bis = px.area(df1, 'FIRE_YEAR' , "FIRE_SIZE", color="STAT_CAUSE_DESCR", line_group="STAT_CAUSE_DESCR")    
    fig2bis.update_layout(title_text="Répartition des feux par année et cause (en acres)", title_x = 0.3, title_y = 1,paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',width=1000, height=350,legend=dict(x=0.0, y=0.93,orientation="h",xanchor="center",yanchor="bottom",font=dict(
            family="Arial",size=11,color="black")),margin=dict(l=0, r=0, t=0, b=0),titlefont=dict(size=15))
    st.plotly_chart(fig2bis)
  with col2 :
    fig3bis = px.area(df1bis, 'FIRE_YEAR' , "FPA_ID", color="STAT_CAUSE_DESCR", line_group="STAT_CAUSE_DESCR")
    fig3bis.update_layout(title_text="Répartition des feux par année et cause (en nombre)", title_x = 0.3, title_y = 1,paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',width=1000, height=350,legend=dict(x=0.0, y=0.93,orientation="h",xanchor="center",yanchor="bottom",font=dict(
            family="Arial",size=11,color="black")),margin=dict(l=0, r=0, t=0, b=00),titlefont=dict(size=15))
    st.plotly_chart(fig3bis)
#Histogrammes mois
  st.write("Les mois de juin à août sont les plus dévastateurs ce qui qui peut sous-entendre 2 facteurs : un climat plus favorable aux départs de feux, des activités humaines à risque plus élevées pendant les périodes de vacances")
  if st.checkbox("Afficher graphiques mois") :
    fig3= make_subplots(rows=1, cols=2, shared_yaxes=False,subplot_titles=("En surfaces brûlées (acres)","En Nombre de départs"))
    fig3.add_trace(go.Histogram(histfunc="sum",
      name="Surface brûlées (acres) ",
      x=df['MONTH_DISCOVERY'],y=df['FIRE_SIZE'], marker_color='red'),1,1)
    fig3.add_trace(go.Histogram(histfunc="count",
      name="Nombre de feux",
      x=df['MONTH_DISCOVERY'],marker_color='blue'),1,2)
    fig3.update_layout(title_text="Départs de feux par mois",bargap=0.2,height=400, width=1100, coloraxis=dict(colorscale='Bluered_r'), showlegend=False,paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)')

    st.plotly_chart(fig3)

#Histogrammes jour semaine
  st.write("On observe également des départs de feux significativement plus élevés le week-end. Ce qui peut être mis en corrélation avec les feux d'origine humaine déclenchés par des activités à risque plus propices en périodes de week-end (feux de camps...)")
  if st.checkbox("Afficher graphiques jour de la semaine") :
    df['DAY_OF_WEEK_DISCOVERYName'] = pd.to_datetime(df['DISCOVERY_DATE']).dt.day_name()
    Fires2=df.loc[df['STAT_CAUSE_DESCR']!="Foudre"]
    Fires3 = Fires2.groupby('DAY_OF_WEEK_DISCOVERYName')['FPA_ID'].value_counts().groupby('DAY_OF_WEEK_DISCOVERYName').sum()
    week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    Fires3 = Fires3.reindex(week)
    Fires4 = Fires2.groupby('DAY_OF_WEEK_DISCOVERYName')['FIRE_SIZE'].sum()
    week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    Fires4 = Fires4.reindex(week)
    fig4 = make_subplots(rows=1, cols=2, shared_yaxes=False,subplot_titles=("En surfaces brûlées (acres)","En Nombre de départs"))
    fig4.add_trace(go.Histogram(histfunc="sum",
      name="Surface brûlées (acres) ",
      x=Fires4.index,y=Fires4.values, marker_color='red'),1,1)
    fig4.add_trace(go.Histogram(histfunc="sum",
      name="Nombre de feux",
      x=Fires3.index,y=Fires3.values,marker_color='blue'),1,2)
    fig4.update_layout(title_text="Départs de feux en fonction du jour de la semaine",bargap=0.2,height=400, width=1000, coloraxis=dict(colorscale='Bluered_r'), showlegend=False,paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig4)

    df3=df.groupby(['STAT_CAUSE_DESCR', 'DISCOVERY_DOY']).agg({"FIRE_SIZE":"sum"}).reset_index()
 
    fig4bis = px.area(df3, 'DISCOVERY_DOY' , "FIRE_SIZE", color="STAT_CAUSE_DESCR", line_group="STAT_CAUSE_DESCR")
    fig4bis.update_layout(title_text="Répartition des feux jours de l'année et cause (en acres)", title_x = 0.3, title_y = 1,paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',width=1000, height=400,legend=dict(x=0.5, y=0.93,orientation="h",xanchor="center",yanchor="bottom",font=dict(
            family="Arial",size=15,color="black")),margin=dict(l=100, r=100, t=25, b=50),titlefont=dict(size=20))
    st.plotly_chart(fig4bis)
# Durée moyenne
  st.write('L’analyse de la durée des feux par cause montre une certaine hétérogénéité de la durée des feux en fonction de la cause. Les feux liés à la foudre sont en moyenne deux fois plus longs à contenir que les autres types de feux')
  if st.checkbox("Afficher graphiques par durée") :
    Fires_burn = df.dropna(subset=['CONT_DOY', 'DISCOVERY_DOY']).copy()
    Fires_burn['CONT_DOY'] = Fires_burn['CONT_DOY'].astype(int)
    Fires_burn['DISCOVERY_DOY'] = Fires_burn['DISCOVERY_DOY'].astype(int)
    Fires_burn['BURN_TIME'] = Fires_burn['CONT_DOY'] - Fires_burn['DISCOVERY_DOY']
    Fires_burn.loc[Fires_burn['BURN_TIME'] < 0, 'BURN_TIME'] += 365
    cause_avg_burn_time = Fires_burn.groupby('STAT_CAUSE_DESCR', dropna=True)['BURN_TIME'].mean().reset_index()
    cause_avg_burn_time.sort_values(by='BURN_TIME', ascending=False, inplace=True)
    fig5 = px.bar(
      cause_avg_burn_time,
      x='BURN_TIME',
      y='STAT_CAUSE_DESCR',
      labels={"STAT_CAUSE_DESCR": "Cause", "BURN_TIME": "Durée moyenne (jours)"},
      title='Durée moyenne de feux par cause',
      orientation='h',  # Horizontal orientation
      color='STAT_CAUSE_DESCR',
      color_discrete_sequence=px.colors.sequential.Reds_r)
    fig5.update_layout(xaxis=dict(tickmode='linear', dtick=0.5),title_x = 0.3, title_y = 1,paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',width=1000, height=400,showlegend=False,margin=dict(l=170, r=200, t=50, b=50),titlefont=dict(size=20))
    st.plotly_chart(fig5)
  
  st.subheader("4 - Répartition géographique")
  #if st.checkbox("Afficher graphiques répartition géographique") :
  #Réprésentation géographique des feux de classe E à G (> 300 acres) par taille (taille de la bulle) et par cause (couleur de la bulle)
  Fires_bis = df
  modif1 = ["Campfire", "Debris Burning", "Smoking", "Fireworks", "Children"]
  modif2 = ["Powerline", "Railroad", "Structure", "Equipment Use"]
  Fires_bis["STAT_CAUSE_DESCR"] = Fires_bis["STAT_CAUSE_DESCR"].replace("Missing/Undefined", "Miscellaneous")
  Fires_bis["STAT_CAUSE_DESCR"] = Fires_bis["STAT_CAUSE_DESCR"].replace(modif1, "Origine humaine")
  Fires_bis["STAT_CAUSE_DESCR"] = Fires_bis["STAT_CAUSE_DESCR"].replace(modif2, "Equipnt Infrastr")
  Fires_bis["FIRE_SIZE_CLASS"] = Fires_bis["FIRE_SIZE_CLASS"].replace(["A", "B", "C"], "ABC")
  Fires_bis["FIRE_SIZE_CLASS"] = Fires_bis["FIRE_SIZE_CLASS"].replace(["D", "E", "F"], "DEF")
  FiresClasse = df[(Fires_bis['FIRE_SIZE_CLASS'] != "ABC")&(df['FIRE_SIZE_CLASS'] != "B")&(df['FIRE_SIZE_CLASS'] != "C")&(df['FIRE_SIZE_CLASS'] != "D")]
  FiresClasse = df[(Fires_bis['FIRE_SIZE_CLASS'] != "ABC")]
    #fig6 = px.scatter_geo(FiresClasse,
    #      lon = FiresClasse['LONGITUDE'],
    #      lat = FiresClasse['LATITUDE'],
    #      color="STAT_CAUSE_DESCR",
    #          #facet_col="FIRE_YEAR", #pour créer un graph par année
     #   #facet_col_wrap,# pour définir le nombre de graph par ligne
      #  #animation_frame="FIRE_YEAR",#pour créer une animation sur l'année
    #      color_discrete_sequence=["blue","orange","red","grey","purple"],
    #      labels={"STAT_CAUSE_DESCR": "Cause"},
    #      hover_name="STATE", # column added to hover information
    #      size=FiresClasse['FIRE_SIZE']/1000, # size of markers
    #      projection='albers usa',
    #      locationmode = 'USA-states',
    #      width=800,
    #      height=500,
    #      title="Répartition géographique des feux par cause et taille",basemap_visible=True)
    #fig6.update_geos(resolution=50,lataxis_showgrid=True, lonaxis_showgrid=True,bgcolor='rgba(0,0,0,0)',framecolor='blue',showframe=True,showland=True,landcolor='#e0efe7',projection_type="albers usa")
    #fig6.update_layout(title_text="Répartition géographique des feux par cause et taille", title_x = 0.3, title_y = 0.95,paper_bgcolor='rgba(0,0,0,0)',
    #plot_bgcolor='rgba(0,0,0,0)',width=1000, height=500,legend=dict(x=0.5, y=1.05,orientation="h",xanchor="center",yanchor="bottom",font=dict(
    #        family="Arial",size=15,color="black")),margin=dict(l=50, r=50, t=100, b=50),titlefont=dict(size=20))      ,
    
    #st.plotly_chart(fig6)
  if st.checkbox("Afficher graphiques répartition géographique et année") :
    col1, col2 = st.columns(2)

    with col1:
      fig7 = px.scatter_geo(FiresClasse,
         lon = FiresClasse['LONGITUDE'],
          lat = FiresClasse['LATITUDE'],
          color="STAT_CAUSE_DESCR",
    #    #facet_col="FIRE_YEAR", #pour créer un graph par année
     #   #facet_col_wrap,# pour définir le nombre de graph par ligne
        #animation_frame="FIRE_YEAR",#pour créer une animation sur l'année
          color_discrete_sequence=["blue","orange","red","grey","purple"],
          labels={"STAT_CAUSE_DESCR": "Cause"},
          hover_name="STATE", # column added to hover information
          size=FiresClasse['FIRE_SIZE']/1000, # size of markers
          projection='albers usa',
          width=800,
          height=500,
          title="Répartition géographique des feux par cause, taille",basemap_visible=True)
      fig7.update_geos(resolution=50,lataxis_showgrid=True, lonaxis_showgrid=True,bgcolor='rgba(0,0,0,0)',framecolor='blue',showframe=True,showland=True,landcolor='#e0efe7',projection_type="albers usa")
      fig7.update_layout(title_text="Répartition géographique des feux par cause et taille", title_x = 0.1, title_y = 0.95,paper_bgcolor='rgba(0,0,0,0)',
      plot_bgcolor='rgba(0,0,0,0)',width=1000, height=700,legend=dict(x=0.5, y=0.95,orientation="h",xanchor="center",yanchor="bottom",font=dict(
            family="Arial",size=11,color="black")),margin=dict(l=50, r=50, t=100, b=150),titlefont=dict(size=18))   
    
      st.plotly_chart(fig7)

    with col2:
      fig7_ = px.scatter_geo(FiresClasse,
         lon = FiresClasse['LONGITUDE'],
          lat = FiresClasse['LATITUDE'],
          color="STAT_CAUSE_DESCR",
    #    #facet_col="FIRE_YEAR", #pour créer un graph par année
     #   #facet_col_wrap,# pour définir le nombre de graph par ligne
          animation_frame="FIRE_YEAR",#pour créer une animation sur l'année
          color_discrete_sequence=["blue","orange","red","grey","purple"],
          labels={"STAT_CAUSE_DESCR": "Cause"},
          hover_name="STATE", # column added to hover information
          size=FiresClasse['FIRE_SIZE']/1000, # size of markers
          projection='albers usa',
          width=800,
          height=500,
          title="Focus par année",basemap_visible=True)
      fig7_.update_geos(resolution=50,lataxis_showgrid=True, lonaxis_showgrid=True,bgcolor='rgba(0,0,0,0)',framecolor='blue',showframe=True,showland=True,landcolor='#e0efe7',projection_type="albers usa")
      fig7_.update_layout(title_text="Focus par année", title_x = 0.4, title_y = 0.95,paper_bgcolor='rgba(0,0,0,0)',
      plot_bgcolor='rgba(0,0,0,0)',width=1000, height=500,legend=dict(x=0.5, y=0.95,orientation="h",xanchor="center",yanchor="bottom",font=dict(
            family="Arial",size=11,color="black")),margin=dict(l=50, r=50, t=100, b=50),titlefont=dict(size=18))   
    
      st.plotly_chart(fig7_)
         
     
    
  st.subheader("5 - Analyse corrélations entre variables")
# Plot heatmap - correlation matrix for all numerical columns
#style.use('ggplot')
  
  if st.checkbox("Afficher heatmap") :
    df_Fires_ML_num = df.select_dtypes(include=[np.number])
    plt.subplots(figsize = (10,7))
    sns.set_style(style='white')
    sns.set(rc={"axes.facecolor": "#F4E4AA", "figure.facecolor": "#F4E4AA"})
    df_Fires_ML_num = df.select_dtypes(include=[np.number])
    mask = np.zeros_like(df_Fires_ML_num.corr(), dtype='bool')
    mask[np.triu_indices_from(mask)] = True
    fig7b, ax = plt.subplots(figsize = (10,7))
    sns.heatmap(df_Fires_ML_num.corr(), cmap=sns.diverging_palette(20, 220, n=200), annot=True, center = 0, mask=mask, annot_kws={"size": 8})
    plt.title("Heatmap of all the selected features of data set", fontsize = 15)
    st.write(fig7b)

    #df_Fires_ML_num = df.select_dtypes(include=[np.number])
    #fig7b, ax = plt.subplots(figsize = (10,7))
    #sns.heatmap(df_Fires_ML_num.corr(), ax=ax)
    #st.write(fig7b)

    #df_Fires_ML_num = df.select_dtypes(include=[np.number])
    #fig7b = px.imshow(df_Fires_ML_num)
    #st.plotly_chart(fig7b)


if page == pages[3] : 
  st.write("### Prédiction causes de feux")

  # Suppression des variables non utiles au ML
  Drop_col_ML = ["NWCG_REPORTING_UNIT_NAME", "FPA_ID","DISCOVERY_DATE","DISCOVERY_DOY","DISCOVERY_TIME","CONT_DOY","CONT_DATE","CONT_TIME","FIRE_SIZE","STAT_CAUSE_DESCR","COUNTY","FIPS_NAME"] 
  Fires35 = df.dropna()
  Fires_ML = Fires35.drop(Drop_col_ML, axis = 1)
  # Suppression des lignes de "STATE", "AVG_TEMP [°C]", "AVG_PCP [mm]" ayant des données manquantes 
  Fires_ML = Fires_ML.dropna(subset = ["STATE", "AVG_TEMP [°C]", "AVG_PCP [mm]"])
  # Création d'une checkbox pour afficher ou non le jeu de données ML
  if st.checkbox("Affichage du jeu de données pour Machine Learning") :
    st.dataframe(Fires_ML.head(5))
    st.write(Fires_ML.columns)
  
  # Création d'une checkbox pour afficher la distribution des causes avant et après regroupement
  # Nouvelle distribution des causes suite au regroupement des causes initiales
  Fires_ML = Fires_ML[(Fires_ML.loc[:, "STAT_CAUSE_CODE"] != 9) & (Fires_ML.loc[:, "STAT_CAUSE_CODE"] != 13)]

  Fires_ML["STAT_CAUSE_CODE"] = Fires_ML["STAT_CAUSE_CODE"].replace([3, 4, 5, 8, 2, 6, 10, 11, 12], 20)
  Fires_ML["STAT_CAUSE_CODE"] = Fires_ML["STAT_CAUSE_CODE"].replace(7, 21)
  Fires_ML["STAT_CAUSE_CODE"] = Fires_ML["STAT_CAUSE_CODE"].replace(1, 22)
  Fires_ML["STAT_CAUSE_CODE"] = Fires_ML["STAT_CAUSE_CODE"].replace({20: 0, 21: 1, 22: 2})
  if st.checkbox("Regroupement des causes de feux"):
    col1, col2= st.columns(spec = 2, gap = "large")
    with col1:
      st.write("### Distribution initiale des causes de feux")
      count = Fires_ML["STAT_CAUSE_DESCR_1"].value_counts()
      color = ["blue", "orange", "yellow", "blue", "blue", "blue", "blue", "blue", "blue", "blue", "blue"]
      fig, ax = plt.subplots(figsize=(16, 12), facecolor='none') 
      ax.bar(count.index, count.values, label = Fires_ML["STAT_CAUSE_DESCR_1"].unique(), color=color)
      ax.set_facecolor('none') 
      fig.patch.set_alpha(0.0) 
      ax.set_ylabel("COUNT", fontsize=25)
      ax.set_xticks(range(len(count.index)))
      ax.set_xticklabels(count.index, rotation=75, fontsize=25)
      # ax.set_yticklabels(count.values, fontsize=40)
      ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
      st.pyplot(fig)
      st.write("""On observe un grand déséquilibre du jeu de données. Ce qui va rendre complexe la prédiction de l'analyse.
               Les feux Missing/Undefined et Miscellaneous représentent environ le quart des données. 
               Compte tenu de leur caractère inerte par rapport à l'objectif de l'étude, nous les supprimerons.
               Pour les diverses qui peuvent se ressembler, nous procéderons à leur regroupement dans une cause parente""")
    with col2:
      st.write("### Distribution des causes apès regroupement")
      count2 = Fires_ML["STAT_CAUSE_CODE"].value_counts()
      color = ["blue", "orange", "yellow"]
      fig, ax = plt.subplots(figsize=(16, 12), facecolor='none')  
      ax.bar(count2.index, count2.values, color=color)
      ax.set_facecolor('none') 
      fig.patch.set_alpha(0.0) 
      ax.set_ylabel("COUNT", fontsize = 25)
      ax.set_xticks([0, 1, 2])
      ax.set_xticklabels(["Humaine", "Criminelle", "Naturelle"], rotation = 25, fontsize = 25)
      ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
      st.pyplot(fig)

      st.write("Suppresion des causses non-définies : Missing/Undefined, Miscellaneous, Others")
      st.write("""
               Regroupement des feux en 3 principales causes :
               - **Humaine (0)** : Debris burning, Campfire, Children, Smoking, Equipment Use, Railroad, Powerline, Structure, Fireworks"
               - **Criminelle (1)** : Arson"
               - **Naturelle (2)** : Ligthning
               """)

  # Preprocessing des données pour le ML
  Fires_ML = Fires_ML.drop("STAT_CAUSE_DESCR_1", axis = 1)
  @st.cache_data(persist=True)
  def data_labeling(data):
    # Remplacement des jours de la semaine par 1 à 7 au lieu de 0 à 6
    data["DAY_OF_WEEK_DISCOVERY"] = data["DAY_OF_WEEK_DISCOVERY"].replace({0:1, 1:2, 2:3, 3:4, 4:5, 5:6, 6:7})
    # Data preparation for time-series split
    data.sort_values(["FIRE_YEAR", "MONTH_DISCOVERY", "DISCOVERY_WEEK", "DAY_OF_WEEK_DISCOVERY"], inplace = True)
    # Fires_ML.set_index("FIRE_YEAR", inplace = True)
    feats, target = data.drop("STAT_CAUSE_CODE", axis = 1), data["STAT_CAUSE_CODE"]
    # OneHotEncoding des variables catégorielles avec get_dummies avec le train_test_split
    feats = pd.get_dummies(feats, dtype = "int")
    return feats, target
  feats, target = data_labeling(Fires_ML)
  # st.dataframe(feats.head())

  @st.cache_data(persist=True)
  def data_split(X, y):
    # Data split of features and target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, shuffle = False)
    # display(feats.shape, X_train.shape, X_test.shape)
    return X_train, X_test, y_train, y_test
  X_train, X_test, y_train, y_test = data_split(feats, target)
  # st.dataframe(X_train.head())

  @st.cache_data(persist=True)
  def cyclic_transform(X_train, X_test):
    # Séparation des variables suivant leur type
    circular_cols_init = ["MONTH_DISCOVERY", "DISCOVERY_WEEK", "DAY_OF_WEEK_DISCOVERY"]
    circular_train, circular_test = X_train[circular_cols_init], X_test[circular_cols_init]
    
    # Encodage des variables temporelles cycliques
    circular_train["SIN_MONTH"] = circular_train["MONTH_DISCOVERY"].apply(lambda m: np.sin(2*np.pi*m/12))
    circular_train["COS_MONTH"] = circular_train["MONTH_DISCOVERY"].apply(lambda m: np.cos(2*np.pi*m/12))
    circular_train["SIN_WEEK"] = circular_train["DISCOVERY_WEEK"].apply(lambda w: np.sin(2*np.pi*w/53))
    circular_train["COS_WEEK"] = circular_train["DISCOVERY_WEEK"].apply(lambda w: np.cos(2*np.pi*w/53))
    circular_train["SIN_DAY"] = circular_train["DAY_OF_WEEK_DISCOVERY"].apply(lambda d: np.sin(2*np.pi*d/7))
    circular_train["COS_DAY"] = circular_train["DAY_OF_WEEK_DISCOVERY"].apply(lambda d: np.cos(2*np.pi*d/7))

    circular_test["SIN_MONTH"] = circular_test["MONTH_DISCOVERY"].apply(lambda m: np.sin(2*np.pi*m/12))
    circular_test["COS_MONTH"] = circular_test["MONTH_DISCOVERY"].apply(lambda m: np.cos(2*np.pi*m/12))
    circular_test["SIN_WEEK"] = circular_test["DISCOVERY_WEEK"].apply(lambda w: np.sin(2*np.pi*w/53))
    circular_test["COS_WEEK"] = circular_test["DISCOVERY_WEEK"].apply(lambda w: np.cos(2*np.pi*w/53))
    circular_test["SIN_DAY"] = circular_test["DAY_OF_WEEK_DISCOVERY"].apply(lambda d: np.sin(2*np.pi*d/7))
    circular_test["COS_DAY"] = circular_test["DAY_OF_WEEK_DISCOVERY"].apply(lambda d: np.cos(2*np.pi*d/7))

    # Suppression des variables cycliques sources pour éviter le doublon d'informations
    circular_train = circular_train.drop(circular_cols_init, axis = 1).reset_index(drop = True)
    circular_test = circular_test.drop(circular_cols_init, axis = 1).reset_index(drop = True)

    # Récupération des noms de colonnes des nouvelles variables
    circular_cols = circular_train.columns
    return circular_train, circular_test
  circular_train, circular_test = cyclic_transform(X_train, X_test)
  # st.dataframe(circular_train.head())

  @st.cache_data(persist=True)
  def num_imputer(X_train, X_test):
    circular_cols_init = ["MONTH_DISCOVERY", "DISCOVERY_WEEK", "DAY_OF_WEEK_DISCOVERY"]
    num_cols = feats.drop(circular_cols_init, axis = 1).columns
    num_train, num_test = X_train[num_cols], X_test[num_cols]
    # Instanciation de la méthode SimpleImputer
    numeric_imputer = SimpleImputer(strategy = "median")
    # Initialisation des variables
    CLASS = ["FIRE_SIZE_CLASS_A", "FIRE_SIZE_CLASS_B", "FIRE_SIZE_CLASS_C", "FIRE_SIZE_CLASS_D", "FIRE_SIZE_CLASS_E", 
             "FIRE_SIZE_CLASS_F", "FIRE_SIZE_CLASS_G"]
    sub_col = ["DURATION","FIRE_SIZE_CLASS_A", "FIRE_SIZE_CLASS_B", "FIRE_SIZE_CLASS_C", "FIRE_SIZE_CLASS_D", 
               "FIRE_SIZE_CLASS_E", "FIRE_SIZE_CLASS_F", "FIRE_SIZE_CLASS_G"]
    sub_num_train_data = num_train[sub_col]
    sub_num_test_data = num_test[sub_col]
    train, test = sub_num_train_data, sub_num_test_data
    for fire_class in CLASS:
        num_train_imputed = numeric_imputer.fit_transform(sub_num_train_data[sub_num_train_data[fire_class] == 1])
        num_test_imputed = numeric_imputer.transform(sub_num_test_data[sub_num_test_data[fire_class] == 1])
        train[train[fire_class] == 1] = num_train_imputed
        test[test[fire_class] == 1] = num_test_imputed
    num_train["DURATION"], num_test["DURATION"] = train["DURATION"], test["DURATION"]
    num_train, num_test = num_train.reset_index(drop = True), num_test.reset_index(drop = True)
    return num_train, num_test
  num_train_imputed, num_test_imputed = num_imputer(X_train, X_test)
  # st.dataframe(num_train_imputed.head())

  @st.cache_data(persist=True)
  def X_concat(X_train_num, X_test_num, circular_train, circular_test):
    X_train_final = pd.concat([X_train_num, circular_train], axis = 1)
    X_test_final = pd.concat([X_test_num, circular_test], axis = 1)
    X_total = pd.concat([X_train_final, X_test_final], axis = 0)
    y_total = pd.concat([y_train, y_test], axis = 0)
    return X_train_final, X_test_final
  X_train_final, X_test_final = X_concat(num_train_imputed, num_test_imputed, circular_train, circular_test)
  # st.dataframe(X_train_final.head(10))

  # Tracé des courbes Precison_Recall
  @st.cache_data(persist=True)
  def multiclass_PR_curve(classifier, X_test, y_test):
    """Cette fonction de tracer les courbes Precision_Recall pour une classification multi-classe.
    """
    from itertools import cycle
    from sklearn import metrics
    from collections import Counter
    import matplotlib.pyplot as plt
    from sklearn.metrics import PrecisionRecallDisplay, average_precision_score, precision_recall_curve
    
    # Pour chaque classe
    precision, recall, average_precision = dict(), dict(), dict()
    y_score = classifier.predict_proba(X_test)
    
    n_classes = y_test.shape[1]
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], y_score[:, i])
        average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])
    # une "micro-moyenne": pour quantifier le score pour chaque classe
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(), y_score.ravel())
    average_precision["micro"] = average_precision_score(y_test, y_score, average = "micro")
    display_PR = PrecisionRecallDisplay(recall = recall["micro"], 
                                     precision = precision["micro"], 
                                     average_precision = average_precision["micro"])

    # Tracé de la courbe Precision-Recall pour chaque classe et aussi les courbes iso-f1
    # Définition des paramètres des courbes
    colors = cycle(["navy", "turquoise", "darkorange", "cornflowerblue", "teal"])

    _, ax = plt.subplots(figsize=(10, 8))
    # Tracé des courbes des iso_f1_score
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines, labels = [], []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
        plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))
    # Tracé de la courbe Precision_Recall de l'ensemble des classe
    display_PR.plot(ax=ax, name="Micro-average precision-recall", color="gold")
    # Tracé de la courbe Precision_Recall des différentes classes
    for i, color in zip(range(n_classes), colors):
        display_PR = PrecisionRecallDisplay(recall=recall[i], precision=precision[i], average_precision=average_precision[i])
        display_PR.plot(ax=ax, name=f"Precision-recall for class {i}", color=color)
    # add the legend for the iso-f1 curves
    handles, labels = display_PR.ax_.get_legend_handles_labels()
    handles.extend([l])
    labels.extend(["iso-f1 curves"])
    # set the legend and the axes
    ax.set_ylim([0, 1])
    ax.legend(handles=handles, labels=labels, loc="best")
    ax.set_title("Extension of Precision-Recall curve to multi-class")
    # plt.show()
    st.plotly_chart(display_PR)

  # Analyse de la peformance des modèles
  recall_scorer = make_scorer(recall_score, needs_proba = True, average = "micro")
  def plot_perf(graph):
    if 'Matrice confusion' in graph:
     y_pred = model.predict(X_test)
     cm = confusion_matrix(y_test, y_pred)
     figML = px.imshow(cm,labels={"x": "Classes Prédites", "y": "Classes réelles"},width = 400, height = 400, text_auto = True)#color_continuous_scale='hot'
     layout = go.Layout(title ='Confusion Matrix', paper_bgcolor = 'rgba(0,0,0,0)', plot_bgcolor = 'rgba(0,0,0,0)')
     figML.update_layout(paper_bgcolor = 'rgba(0,0,0,0)', plot_bgcolor = 'rgba(0,0,0,0)', width = 1000, height = 450, legend = dict(x = 0.5, y = 1.05, orientation = "h", xanchor = "center", yanchor = "bottom", font = dict(
            family = "Arial", size = 15, color = "black")), margin = dict(l = 100, r = 100, t = 100, b = 100), titlefont = dict(size = 20))
     st.plotly_chart(figML)      

    if 'Courbe Recall' in graph:
      st.subheader('Courbe Recall')
      multiclass_PR_curve(xgboost, X_test_final, y_test)

  classifier=st.selectbox("Choix du modèle",("Regression Logistique","Arbre de Décision", "Random Forest", "KNN", "Gradient Boosting", "XGBoost"))

  # Entrainement et sauvegarde des modèles
  # Modèle 1 : Logistic Regression
  if classifier == "XGBoost":
    # XGBoost est KO quand les noms de colonnes contiennent des symboles de type [ or ] or <. On procède donc à leur suppression
    X_train_final = X_train_final.rename(columns={"AVG_TEMP [°C]": "AVG_TEMP", "AVG_PCP [mm]": "AVG_PCP"})
    X_test_final = X_test_final.rename(columns={"AVG_TEMP [°C]": "AVG_TEMP", "AVG_PCP [mm]": "AVG_PCP"})
    st.sidebar.subheader("Veuillez sélectionner les paramètres")

    #Graphiques performances 
    graphes_perf = st.sidebar.multiselect("Choix graphiques",("Matrice confusion","Courbe ROC","Courbe Recall"))

    class_weights = st.sidebar.radio("Voulez-vous rééquilibrer les classes ?", ["Oui", "Non"], horizontal=True)
    if class_weights == "Oui":
      classes_weights = class_weight.compute_sample_weight(class_weight='balanced', y=y_train)
      st.write("Les classes sont ré-équilibrées")
    elif class_weights == "Non":
      classes_weights = None
      st.write("Les classes ne sont pas ré-équilibrées.")
    # Réduction du modèle avec la méthode feature importances
    @st.cache_data(persist=True)
    def model_reduction(X_train, y_train):
      clf = XGBClassifier(tree_meethod = "approx",
                          objective = "multi:softprob").fit(X_train, y_train)
      feat_imp_data = pd.DataFrame(list(clf.get_booster().get_fscore().items()),
                              columns=["feature", "importance"]).sort_values('importance', ascending=True)
      feat_imp = list(feat_imp_data["feature"][-10:])
      # plt.figure()
      # plt.barh(feat_imp_data["feature"][-10:], feat_imp_data["importance"][-10:])
      # plt.title("Feature Importance")
      # plt.xlabel("F score:Gain")
      return feat_imp
    Feature_importances = st.sidebar.radio("Voulez-vous réduire la dimension du jeu ?", ("Oui", "Non"), horizontal=True)
    if Feature_importances == "Oui":
      feat_imp = model_reduction(X_train_final, y_train)
      st.write("Les variables les plus importantes sont", feat_imp)
      X_train_final, X_test_final = X_train_final[feat_imp], X_test_final[feat_imp]
    else:
      X_train_final, X_test_final = X_train_final, X_test_final

    n_estimators = st.sidebar.slider("Veuillez choisir le nombre d'estimateurs", 10, 100, 10, 5)
    tree_method = st.sidebar.radio("Veuillez choisir la méthode", ("approx", "hist"), horizontal=True)
    max_depth = st.sidebar.slider("Veuillez choisir la profondeur de l'arbre", 3, 20, 5)
    learning_rate = st.sidebar.slider("Veuillez choisir le learning rate", 0.005, 0.5, 0.1, 0.005)
    
    
  # Création d'un bouton pour le modèle avec les meilleurs paramètres
  if st.sidebar.button("Best Model Execution"):
    st.subheader("XGBoost Result")
    best_params = {"learning_rate": 0.1, 
                   "max_depth": 5, 
                   "n_estimators": 10}
    clf_xgb_best = XGBClassifier(objective = "multi:softprob",
                                 tree_method = "approx",
                                 **best_params).fit(X_train_final[feat_imp], y_train)
    # Enrégistrement du meilleur modèle
    joblib.dump(clf_xgb_best, "clf best model.joblib")
    # Chargement du meilleur modèle
    clf_best_model = joblib.load("clf best model.joblib")
    # Prédiction avec le meilleur modèle
    best_clf_pred = clf_best_model.predict(X_test_final[feat_imp])
    # st.write("Le score de prédiction est de :", clf_best_model.score(X_test_final[feat_imp], y_test))
    # Métriques
    accuracy = clf_best_model.score(X_test_final[feat_imp], y_test)
    # recall = recall_scorer(y_test, best_clf_pred)
    # recall_scorer = make_scorer(recall_score, average = "micro")

    #Afficher
    st.write("Accuracy",round(accuracy,4))
    # st.write("recall",round(recall,4))

    #Afficher les graphique performance
    plot_perf(graphes_perf)

  # Création d'un bouton utilisateur pour l'interactivité
  if st.sidebar.button("User Execution", key = "classify"):
    st.subheader("XGBoost User Results")
    model = XGBClassifier(n_estimators = n_estimators,
                          objective = "multi:softprob",
                          tree_method = tree_method, 
                          max_depth = max_depth,
                          learning_rate = learning_rate,                  
                          sample_weight = classes_weights).fit(X_train_final, y_train)
    st.write("Le score d'entrainement est :", model.score(X_train_final, y_train))
    y_pred = model.predict(X_test_final)
    st.write("Le score de test est :", model.score(X_test_final, y_test))


if page == pages[4] : 
  #st.write("### Prédiction classes de feux")

  Fires34=df.dropna()
  FiresML2= Fires34.loc[:,['MONTH_DISCOVERY','FIRE_SIZE_CLASS','STAT_CAUSE_DESCR','AVG_TEMP [°C]','AVG_PCP [mm]','LONGITUDE','LATITUDE']]
  #FiresML2= Fires34.loc[:,['MONTH_DISCOVERY','FIRE_SIZE_CLASS','STAT_CAUSE_DESCR','AVG_TEMP [°C]','AVG_PCP [mm]','LONGITUDE','LATITUDE','STATE']]
  FiresML2['FIRE_SIZE_CLASS'] = FiresML2['FIRE_SIZE_CLASS'].replace({"A":0,"B":0,"C":0,"D":1,"E":1,"F":1,"G":1})
  feats = FiresML2.drop('FIRE_SIZE_CLASS', axis=1)
  target = FiresML2['FIRE_SIZE_CLASS'].astype('int')
  X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=0.25, random_state = 42,stratify=target)
  num_train=X_train.drop(['STAT_CAUSE_DESCR','MONTH_DISCOVERY'],axis=1)
  num_test=X_test.drop(['STAT_CAUSE_DESCR','MONTH_DISCOVERY'],axis=1)
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



  
  #if st.checkbox("Afficher jeu données pour Machine learning") :
  #  st.dataframe(FiresML2.head(5))
  
  classifier=st.selectbox("Sélection du modèle",("XGBoost","BalancedRandomForest"))


  if classifier == "XGBoost":
    st.sidebar.subheader("Hyperparamètres XGBoost")
    max_bin_test = st.sidebar.slider("Max_Bin selection",100, 700, 400)
    scale_pos_weight_test = st.sidebar.slider("scale_pos_weight selection",0, 50, 29)

  
    with st.sidebar :
      st.header("Paramètres d'entrée")
      mois=st.slider('mois',1,12,6)
      Cause=st.selectbox("Cause",('Non défini', 'Origine humaine', 'Équipements', 'Criminel', 'Foudre'))
      Température=st.slider('Température',-25.00,40.00,16.00)
      Précipitations=st.slider('Précipitation',0.00,917.00,63.00)
      Longitude=st.slider('Longitude',-178.00,-65.00,-96.00)
      Latitude=st.slider('Latitude',17.00,71.00,37.00)
         
    data={'MONTH_DISCOVERY':mois,'STAT_CAUSE_DESCR':Cause,'AVG_TEMP [°C]':Température,'AVG_PCP [mm]':Précipitations,"LONGITUDE":Longitude,"LATITUDE":Latitude}
    input_df=pd.DataFrame(data,index=[0])
    input_array=np.array(input_df)
    input_fires=pd.concat([input_df,feats],axis=0)
    
    #Data Prep
    #encode x
    num_input_fires=input_fires.drop(['STAT_CAUSE_DESCR','MONTH_DISCOVERY'],axis=1)
    #num_input_fires=input_fires.drop(['STAT_CAUSE_DESCR','MONTH_DISCOVERY','STATE'],axis=1)
    num_input_fires= sc.fit_transform(num_input_fires)    
    oneh = OneHotEncoder(drop = 'first', sparse_output=False)
    cat_input_fires=input_fires.drop(['AVG_TEMP [°C]','AVG_PCP [mm]','MONTH_DISCOVERY','LONGITUDE','LATITUDE'],axis=1)
    cat_input_fires=oneh.fit_transform(cat_input_fires)
    circular_cols = ['MONTH_DISCOVERY']
    circular_input_fires = input_fires[circular_cols]
    circular_input_fires['MONTH_DISCOVERY'] = circular_input_fires['MONTH_DISCOVERY'].apply(lambda h : np.sin(2 * np.pi * h / 12))
    circular_input_fires['MONTH_DISCOVERY'] = circular_input_fires['MONTH_DISCOVERY'].apply(lambda h : np.cos(2 * np.pi * h / 12))
    df_fires_encoded=np.concatenate((num_input_fires,cat_input_fires,circular_input_fires),axis=1)
    x=df_fires_encoded[1:]

  if st.sidebar.button("Execution",key="classify"):
    st.subheader("XGBoost Results")
  model=XGBClassifier(max_bin=max_bin_test,
                        scale_pos_weight=scale_pos_weight_test,
                        subsample=0.92,
                        colsample_bytree=0.96,
                        learning_rate=0.31,
                        tree_method='hist' ).fit(X_train,y_train)
  y_pred=model.predict(X_test)
  y_pred_input=model.predict(df_fires_encoded[:1])
  prediction=model.predict(df_fires_encoded[:1])
  prediction_proba=model.predict_proba(df_fires_encoded[:1])
  df_prediction_proba=pd.DataFrame(prediction_proba)
  df_prediction_proba.columns=['Petite Classe','Grande Classe']
  df_prediction_proba.rename(columns={0:"Petite Classe",1:"Grande Classe"})  
    #Métriques
  accuracy=model.score(X_test,y_test)
    #precision=precision_score(y_test,y_pred).round(4)
  recall=recall_score(y_test,y_pred)
  LAT=input_df[:1].LATITUDE.to_numpy()
  LONG=input_df[:1].LONGITUDE.to_numpy()

  st.subheader("Prédiction classe de feux en fonction des paramètres d'entrée", divider="blue") 
  col1, col2 = st.columns([0.5,0.5],gap="small",vertical_alignment="center")
  with col1 :
    with st.container(height=350):
        m = folium.Map(location=[36.966428, -95.844032],zoom_start=3.49)
        folium.Marker([LAT, LONG],icon=folium.Icon(color='orange', icon='fire', prefix='fa')).add_to(m)
        #folium.Marker([LAT, LONG], popup=input_df[:1].STATE, tooltip=input_df[:1].STATE,icon=folium.Icon(color='orange', icon='fire', prefix='fa')).add_to(m)
        st_data = st_folium(m, width=1200)
    
  with col2 :
    st.write("Paramètres d'entrée")
    input_df=pd.DataFrame(input_df)
    input_df.columns=['Mois','Cause','Température','Précipitation','Longitude','Latitude']
    input_df.rename(columns={'MONTH_DISCOVERY':'Mois','STAT_CAUSE_DESCR':'Cause','AVG_TEMP [°C]':"Température",'AVG_PCP [mm]':"Précipitation",'LONGITUDE':'Longitude','LATITUDE':"Latitude"})  
    input_df[:1]
    st.write("Probabilité de classe")
    st.dataframe(df_prediction_proba,
                 column_config={
                'Petite Classe':st.column_config.ProgressColumn('Petite Classe',format='%.2f',width='medium',min_value=0,max_value=1),
                'Grande Classe':st.column_config.ProgressColumn('Grande Classe',format='%.2f',width='medium',min_value=0,max_value=1)},hide_index=True)
    Fires_class_pred=np.array(['Petite Classe','Grande Classe'])
    st.success(str(Fires_class_pred[prediction][0]))


  #with st.container(height=160,border=None):   
  st.subheader("Scores de performance du modèle optimisé XGBoost", divider="blue")   
    #Afficher
  st.write("Accuracy",round(model.score(X_test,y_test),4))
    #st.write("precision",precision.round(4))
  st.write("Recall",round(recall,4))   
    

  col1, col2,col3 = st.columns(3,gap="small",vertical_alignment="center")
  with col1:
    with st.container(height=350):
      precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
      fpr, tpr, thresholds = roc_curve(y_test, y_pred)
      roc_auc = auc(fpr, tpr)
      figML2 = px.area(x=fpr, y=tpr,title=f'Courbe ROC (AUC={auc(fpr, tpr):.4f})',labels=dict(x='Taux faux positifs', y='Taux vrais positifs'))
      figML2.add_shape(type='line', line=dict(dash='dash'),x0=0, x1=1, y0=0, y1=1)
      figML2.update_yaxes(scaleanchor="x", scaleratio=1)
      figML2.update_xaxes(constrain='domain')
      figML2.update_layout(title_x = 0.2, title_y =0.98,paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',width=900, height=320,margin=dict(l=0, r=0, t=20, b=0))
      st.plotly_chart(figML2)

  with col2:
    with st.container(height=350):
      #st.subheader('Matrice de confusion') 
      cm = confusion_matrix(y_test, y_pred)
      figML1 = px.imshow(cm,labels={"x": "Predicted Label", "y": "True Label"},width=800,height=800,text_auto=True)#color_continuous_scale='hot'
      #layout = go.Layout(title='Matrice de confusion',paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')
      figML1.update_layout(title='Matrice de confusion',title_x = 0.35, title_y =0.98,paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',width=900, height=320,legend=dict(x=0.5, y=1,orientation="h",xanchor="center",yanchor="bottom",font=dict(
      family="Arial",size=15,color="black")),margin=dict(l=0, r=0, t=2, b=0))
      figML1.update_traces(dict(showscale=False,coloraxis=None), selector={'type':'heatmap'})
      #titlefont=dict(size=20)
      st.plotly_chart(figML1)

  with col3 : 
    with st.container(height=350):     
      feats1 = {}
      for feature, importance in zip(feats.columns,model.feature_importances_):
        feats1[feature] = importance
      importances= pd.DataFrame.from_dict(feats1, orient='index').rename(columns={0: 'Importance'})
      importances.sort_values(by='Importance', ascending=False).head(8)
      
      fig = px.bar(importances, x='Importance', y=importances.index)
      fig.update_layout(title='Feature Importance',title_x = 0.4, title_y = 0.98,paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',width=900, height=320,legend=dict(x=0.5, y=0.93,orientation="h",xanchor="center",yanchor="bottom",font=dict(
      family="Arial",size=15,color="black")),margin=dict(l=0, r=0, t=50, b=0),titlefont=dict(size=15))
      st.plotly_chart(fig)


 


  if classifier == "BalancedRandomForest":
    st.sidebar.subheader("Hyperparamètres BalancedRandomForest")
    min_samples_split_test = st.sidebar.number_input("Min_samples_split selection",2, 10, step=1)
    max_depth_test = st.sidebar.number_input("max_depth selection",1, 100, step=1)
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



if page == pages[5] : 
  #  st.write("### Conclusion")
  st.write("### Conclusion et propositions d’optimisation")
  st.markdown("""
Le projet “Feux de forêts” nous a permis de mettre en pratique les compétences acquises durant notre formation en data analysis, en abordant toutes les étapes d’un projet de Data Science, de l’exploration des données à la modélisation et la data visualisation. Nous avons également abordé les étapes de prédiction, en utilisant des modèles avancés pour prévoir les occurrences de feux de forêts et ainsi mieux comprendre les facteurs qui les influencent.
### Résultats obtenus :
- **Amélioration des performances des modèles** : Grâce à l'utilisation de différentes méthodes de rééchantillonnage et de classificateurs spécifiques pour les jeux de données déséquilibrés, nous avons significativement amélioré les performances des modèles. Le peaufinage des hyperparamètres avec Grid Search a permis d’optimiser encore plus ces performances.
- **Modèle le plus performant** : Le modèle BalancedRandomForest s'est distingué parmi les huit modèles testés, avec un Recall de presque 80% pour la classe 1 et un score ROC AUC de 0,77. Ce modèle utilise principalement quatre des sept features : température, précipitation, mois et cause du feu.
### Pistes d’optimisation :
- **Méthodes de resampling** : Utiliser des méthodes plus précises que les méthodes aléatoires, comme SMOTETomek, SMOTEEN, KmeansSMOTE.
- **Données plus précises** : Ajouter des données plus détaillées sur les températures, par exemple des données journalières au lieu de moyennes mensuelles.
- **Réglage des hyperparamètres** : Continuer à optimiser les hyperparamètres, car toutes les combinaisons n’ont pas pu être testées.
### Impact et perspectives :
Ce projet a démontré l'importance de la data analysis et du machine learning dans la prévention et la gestion des incendies de forêt. En permettant une détection précoce des incendies et en ciblant les interventions là où elles sont le plus nécessaires, nous pouvons contribuer à réduire les coûts économiques et les impacts environnementaux des feux de forêt.
En termes d'expertise, ce projet nous a permis de développer nos compétences en Python et en modélisation via le Machine Learning, des domaines nouveaux pour la plupart d'entre nous. Nous avons également appris à utiliser des outils interactifs comme Streamlit pour la restitution de nos résultats.
Pour aller plus loin, il serait bénéfique de collaborer avec des spécialistes en lutte contre les incendies de forêt pour affiner nos modèles et mieux comprendre les enjeux opérationnels. De plus, l'intégration de données météorologiques plus précises pourrait améliorer encore davantage les performances de nos modèles.
En conclusion, ce projet nous a permis de mettre en pratique les compétences acquises durant notre formation et de contribuer à un enjeu crucial de préservation de l'environnement et de sécurité publique.
""")
