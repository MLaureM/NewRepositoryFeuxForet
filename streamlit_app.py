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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold, cross_val_score
from sklearn import model_selection
from sklearn import tree
from sklearn.metrics import recall_score, f1_score, confusion_matrix, classification_report
from sklearn.datasets import make_classification
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve, auc

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
  st.text("Le projet “Feux de Forêts” s’inscrit dans un contexte crucial de préservation de l’environnement et de sécurité publique.") 
  st.text("Ces forêts ont des conséquences dévastatrices sur les écosystèmes, les communautés locales et l’économie.")

  if st.checkbox("Afficher jeu donnée") :
    st.dataframe(df.head(5))

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
  st.subheader("1 - Analyse des Outliers & Distribution")
  st.write("On remarque une distribution hétérogène des variables avec de nombreux outliers.")
  st.write("Certains outliers sont liés à des erreurs de données (feux de plus de 1 an), d’autres restent des valeurs possibles (feux de très grande taille) qui seront conservés dans nos différentes analyses")
  #if st.checkbox("Afficher Boxplots") :
  # Graphique Gigi

  st.subheader("2 - Répartition des feux par cause et classe")

  st.write("Les feux d’origine humaine (volontaire et involontaire) représentent 50% des départs, tandis que les causes naturelles (foudre) représentent 62,1% des surfaces brûlées.")

  #Pie Chart répartition par cause
  if st.checkbox("Afficher graphiques par cause") :
    Fires_cause = df.groupby("STAT_CAUSE_DESCR").agg({"FPA_ID":"count", "FIRE_SIZE":"sum"}).reset_index()
    Fires_cause = Fires_cause.rename({"FPA_ID":"COUNT_FIRE", "FIRE_SIZE":"FIRE_SIZE_SUM"}, axis = 1)
    Indic = ["≈ Hawaii + Massachusetts", "≈ Hawaii + Massachusetts", "≈ Washington + Georgia", "≈ Maine", "≈ New Jersey + Massachusetts"]
    Fires_cause["Text"] = Indic
    fig = make_subplots(rows = 1, cols = 2,specs = [[{"type":"domain"}, {"type":"domain"}]])
    fig.add_trace(go.Pie(labels = Fires_cause["STAT_CAUSE_DESCR"],values = Fires_cause["COUNT_FIRE"],hole = 0.5,
          direction = "clockwise", title = dict(text = "Nombre", font=dict(size=20))),row = 1, col = 1,)
    fig.add_trace(go.Pie(labels = Fires_cause["STAT_CAUSE_DESCR"],values = Fires_cause["FIRE_SIZE_SUM"],hovertext = Fires_cause["Text"],
           hole = 0.5,direction = "clockwise",title = dict(text = "Surfaces (acres)", font=dict(size=20))),row = 1, col = 2)
    fig.update_traces(textfont_size=15,sort=False,marker=dict(colors=['#F1C40F', '#F39C12', '#e74c3c','#E67E22','#d35400']))
    fig.update_layout(title_text="Répartition des feux par causes (1992 - 2015)", title_x = 0.3, title_y = 0.95,paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',width=1000, height=450,legend=dict(x=0.5, y=1.05,orientation="h",xanchor="center",yanchor="bottom",font=dict(
            family="Arial",size=15,color="black")),margin=dict(l=100, r=100, t=100, b=100),titlefont=dict(size=20))

    st.plotly_chart(fig)
  
  st.write("Les feux de petite taille (A et B, <9,9 acres) représentent 62 % du nombre de départs mais seulement 2% des surfaces brûlées.  78 % des surfaces brûlées sont liées aux feux de la classe G (avec des feux allant de 5000 à 600 000 acres).")
  if st.checkbox("Afficher graphiques par classe") :
  #Pie Chart répartition par classe
    Fires_class = df.groupby("FIRE_SIZE_CLASS").agg({"FPA_ID":"count", "FIRE_SIZE":"sum"}).reset_index()
    Fires_class = Fires_class.rename({"FPA_ID":"COUNT_FIRE", "FIRE_SIZE":"FIRE_SIZE_SUM"}, axis = 1)
    Indic = ["≈ ", "≈ ","≈ Connecticut", "≈ New Jersey", "≈ Maryland", "≈ Virginie Occidentale + Delaware", "≈ Californie + Hawaii"]
    Fires_class["Text"] = Indic
    fig1= make_subplots(rows = 1, cols = 2, specs = [[{"type":"domain"}, {"type":"domain"}]])
    fig1.add_trace(
      go.Pie(labels = Fires_class["FIRE_SIZE_CLASS"],
           values = Fires_class["COUNT_FIRE"],
           hole = 0.5, rotation = 0,
           title = dict(text = "Nombre", font=dict(size=20))),
      row = 1, col = 1)
    fig1.add_trace(
      go.Pie(labels = Fires_class["FIRE_SIZE_CLASS"],
           values = Fires_class["FIRE_SIZE_SUM"],
           hovertext = Fires_class["Text"],
           hole = 0.5, rotation = -120,
           title = dict(text = "Surfaces (acres)", font=dict(size=20))),
      row = 1, col = 2)
    fig1.update_traces(textfont_size=15,sort=False,marker=dict(colors=['yellow','brown','#F1C40F', '#F39C12', '#e74c3c','#E67E22','#d35400']))
    fig1.update_layout(title_text="Répartition des feux suivant leur taille (1992 - 2015)", title_x = 0.3, title_y = 0.95,paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',width=1000, height=450,legend=dict(x=0.5, y=1.05,orientation="h",xanchor="center",yanchor="bottom",font=dict(
            family="Arial",size=15,color="black")),margin=dict(l=100, r=100, t=100, b=100),titlefont=dict(size=20))
    st.plotly_chart(fig1)

  st.subheader("3 - Répartition temporelle des feux")
  st.write("Cet axe révèle assez clairement des périodes à risque sur les départs et la gravité des feux")
#Histogrammes année
  st.write("Certaines années semblent clairement plus propices aux départs de feux. Cela peut s’expliquer par les conditions météorologiques. On observe notamment que les années où les surfaces brûlées sont significativement supérieures à la moyenne cela est dû à la foudre")
  if st.checkbox("Afficher graphiques année") :
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

    fig2bis = px.area(df1, 'FIRE_YEAR' , "FIRE_SIZE", color="STAT_CAUSE_DESCR", line_group="STAT_CAUSE_DESCR")
    fig2bis.update_layout(title_text="Répartition des feux par année et cause (en acres)", title_x = 0.3, title_y = 1,paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',width=1000, height=400,legend=dict(x=0.5, y=0.93,orientation="h",xanchor="center",yanchor="bottom",font=dict(
            family="Arial",size=15,color="black")),margin=dict(l=100, r=100, t=25, b=50),titlefont=dict(size=20))
    st.plotly_chart(fig2bis)

  
    fig3bis = px.area(df1bis, 'FIRE_YEAR' , "FPA_ID", color="STAT_CAUSE_DESCR", line_group="STAT_CAUSE_DESCR")
    fig3bis.update_layout(title_text="Répartition des feux par année et cause (en nombre)", title_x = 0.3, title_y = 1,paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',width=1000, height=400,legend=dict(x=0.5, y=0.93,orientation="h",xanchor="center",yanchor="bottom",font=dict(
            family="Arial",size=15,color="black")),margin=dict(l=100, r=100, t=25, b=50),titlefont=dict(size=20))
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
    fig7 = px.scatter_geo(FiresClasse,
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
          title="Répartition géographique des feux par cause, taille et année",basemap_visible=True)
    fig7.update_geos(resolution=50,lataxis_showgrid=True, lonaxis_showgrid=True,bgcolor='rgba(0,0,0,0)',framecolor='blue',showframe=True,showland=True,landcolor='#e0efe7',projection_type="albers usa")
    fig7.update_layout(title_text="Répartition géographique des feux par cause et taille", title_x = 0.3, title_y = 0.95,paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',width=1000, height=500,legend=dict(x=0.5, y=1.05,orientation="h",xanchor="center",yanchor="bottom",font=dict(
            family="Arial",size=15,color="black")),margin=dict(l=50, r=50, t=100, b=50),titlefont=dict(size=20))   
    
    st.plotly_chart(fig7)
    
  st.subheader("5 - Analyse corrélations entre variables")
# Plot heatmap - correlation matrix for all numerical columns
#style.use('ggplot')
  
  if st.checkbox("Afficher heatmap") :
    sns.set_style(style='white')
    sns.set(rc={"axes.facecolor": "#F4E4AA", "figure.facecolor": "#F4E4AA"})
    plt.subplots(figsize = (10,7))
    df_Fires_ML_num = df.select_dtypes(include=[np.number])
    mask = np.zeros_like(df_Fires_ML_num.corr(), dtype='bool')
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(df_Fires_ML_num.corr(), cmap=sns.diverging_palette(20, 220, n=200), annot=True, center = 0, mask=mask, annot_kws={"size": 8})
    plt.title("Heatmap of all the selected features of data set", fontsize = 15)
    st.pyplot()

if page == pages[3] : 
  st.write("### Prédiction causes de feux")
 
  # Suppression des variables non utiles au ML et utilisation de l'année de feu comme index
  Drop_col_ML = ["FPA_ID","DISCOVERY_DATE","DISCOVERY_DOY","DISCOVERY_TIME","CONT_DOY","CONT_DATE","CONT_TIME","FIRE_SIZE","STAT_CAUSE_DESCR","COUNTY","FIPS_NAME"] 
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
      count = Fires_ML["STAT_CAUSE_DESCR_1"].value_counts()
      color = ["blue", "magenta", "orange", "yellow", "blue", "blue", "blue", "blue", "blue", "blue", "blue", "blue", "blue"]
      fig, ax = plt.subplots(figsize=(20, 15), facecolor='none') 
      ax.bar(count.index, count.values, color=color)
      ax.set_facecolor('none') 
      fig.patch.set_alpha(0.0) 
      ax.set_ylabel("COUNT", fontsize=40)
      ax.set_xticks(range(len(count.index)))
      ax.set_xticklabels(count.index, rotation=80, fontsize=40)
      ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
      st.pyplot(fig)
  with col2:
      st.write("")
  st.write("""
  On observe un grand déséquilibre du jeu de données. Ce qui va rendre complexe la prédiction de l'analyse.
  Les feux manquants, non-définis et autres représentent environ le quart des feux. Compte tenu de leur caractère inerte 
  par rapport à l'objectif de l'étude, nous les supprimerons.
  De même, compte tenu des diverses qui peuvent parfois se ressembler, nous procéderons à une fusion des causes qui peuvent 
  être regroupés dans une cause parent.
  """)

  st.divider()
  # Suppression des causes non-définies et fusion des autres
  st.write("### Suppression des causes non-définies et fusion des autres")
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
  st.write("### Distribution des causes apès regroupement")
  col1, col2= st.columns([1, 2])
  with col1:
      count2 = Fires_ML["STAT_CAUSE_CODE"].value_counts()
      color = ["blue", "magenta", "yellow"]
      fig, ax = plt.subplots(figsize=(20, 15), facecolor='none')  
      ax.bar(count2.index, count2.values, color=color)
      ax.set_facecolor('none') 
      fig.patch.set_alpha(0.0) 
      ax.set_ylabel("COUNT", fontsize=40)
      ax.set_xticks([0, 1, 2])
      ax.set_xticklabels(["Humaine", "Criminelle", "Naturelle"], rotation = 30, fontsize=40)
      ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
      st.pyplot(fig)
  with col2:
      st.write("")

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

  # Analyse de la peformance des modèles
  def plot_perf(graph):
  
    if 'Matrice confusion' in graph:
     cm = confusion_matrix(y_test, y_pred)
     figML1 = px.imshow(cm,labels={"x": "Predicted Label", "y": "True Label"},width=400,height=400,text_auto=True)#color_continuous_scale='hot'
     layout = go.Layout(title='Confusion Metrix',paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')
     figML1.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',width=1000, height=450,legend=dict(x=0.5, y=1.05,orientation="h",xanchor="center",yanchor="bottom",font=dict(
            family="Arial",size=15,color="black")),margin=dict(l=100, r=100, t=100, b=100),titlefont=dict(size=20))
     st.plotly_chart(figML1)      


    if 'Courbe ROC' in graph:
      st.subheader('Courbe ROC')

      precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
      fpr, tpr, thresholds = roc_curve(y_test, y_pred)
      roc_auc = auc(fpr, tpr)

      figML2 = px.area(x=fpr, y=tpr,title=f'(AUC={auc(fpr, tpr):.4f})',labels=dict(x='Taux faux positifs', y='Taux vrais positifs'))
      figML2.add_shape(type='line', line=dict(dash='dash'),x0=0, x1=1, y0=0, y1=1)
      figML2.update_yaxes(scaleanchor="x", scaleratio=1)
      figML2.update_xaxes(constrain='domain')
      figML2.update_layout(title_x = 0.4, title_y = 0.95,paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',width=1000, height=450,legend=dict(x=0.5, y=0.93,orientation="h",xanchor="center",yanchor="bottom",font=dict(
            family="Arial",size=15,color="black")),margin=dict(l=100, r=100, t=100, b=50),titlefont=dict(size=20))
      st.plotly_chart(figML2)

      
  
    if 'Courbe Recall' in graph:
      st.subheader('Courbe Recall')
      precision, recall, _ = precision_recall_curve(y_test, y_pred)
      model_disp = PrecisionRecallDisplay(precision=precision, recall=recall)
      st.pyplot()

  if classifier == "XGBoost":
    st.sidebar.subheader("Hyperparamètres XGBoost")
    max_bin_test = st.sidebar.slider("Max_Bin selection",100, 700, 400)
    scale_pos_weight_test = st.sidebar.slider("scale_pos_weight selection",0, 50, 29)
    #subsample_test = st.sidebar.slider("subsample selection",0.00, 1.00, 0.92)
    #colsample_bytree_test = st.sidebar.slider("colsample_bytree selection",0.00, 1.00, 0.96)
    #learning_rate_test = st.sidebar.slider("learning_rate selection",0.00, 1.00, 0.31)
    #tree_method_test = st.sidebar.radio("tree_method selection",("hist","approx"))

  #Graphiques performances 
    graphes_perf = st.sidebar.multiselect("Choix graphiques",("Matrice confusion","Courbe ROC","Courbe Recall"))

  if st.sidebar.button("Execution",key="classify"):
    st.subheader("XGBoost Results")
    model=XGBClassifier(max_bin=max_bin_test,
                        scale_pos_weight=scale_pos_weight_test,
                        subsample=0.92,
                        colsample_bytree=0.96,
                        learning_rate=0.31,
                        tree_method='hist' ).fit(X_train,y_train)
    y_pred=model.predict(X_test)
    #Métriques
    accuracy=model.score(X_test,y_test)
    #precision=precision_score(y_test,y_pred).round(4)
    recall=recall_score(y_test,y_pred)

    #Afficher
    st.write("Accuracy",round(accuracy,4))
    #st.write("precision",precision.round(4))
    st.write("recall",round(recall,4))

    #Afficher les graphique performance
    plot_perf(graphes_perf)

  sns.histplot(data=FiresML2, x="FIRE_SIZE_CLASS",bins=2,stat="percent",discrete=False)
  plt.show()

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


  sns.histplot(data=FiresML2, x="FIRE_SIZE_CLASS",bins=2,stat="percent",discrete=False)
  plt.show()


if page == pages[5] : 
  st.write("### Conclusion")
