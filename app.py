import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.express as px
from streamlit_option_menu import option_menu
import geopandas as gpd
import matplotlib.pyplot as plt
#
from flask import Flask, request, jsonify
import joblib
#

# Charger le modèle sauvegardé et les encoders // disponibilites
model = joblib.load(r'xgboost_model.pkl')
label_encoders = joblib.load(r'label_encoders.pkl')
scaler_features = joblib.load(r'scaler_features.pkl')

# Charger le modèle sauvegardé et les encoders // population
model_pop = joblib.load(r'xgboost_model_population_.pkl')
label_encoders_pop = joblib.load(r'label_encoders_population_.pkl')
scaler_features_pop = joblib.load(r'scaler_features_population_.pkl')

# Charger le modèle sauvegardé et les encoders // PDI
model_pdi = joblib.load(r'xgboost_model_PDI_.pkl')
label_encoders_pdi = joblib.load(r'label_encoders_PDI_.pkl')
scaler_features_pdi = joblib.load(r'scaler_features_PDI_.pkl')

# Définir les colonnes de caractéristiques
features = ['Annee', 'Region_encoded', 'Province_encoded', 'Culture_encoded']
features_p = ['Annee', 'Region_encoded', 'Province_encoded']

# Créer une application Flask
flask_app = Flask(__name__)
## -------------------------------------------------------------------------------------------

# Route pour la prédiction
@flask_app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.json
        
        # Encoder les valeurs d'entrée
        data_encoded = {
            'Annee': data['Annee'],
            'Region_encoded': label_encoders['Region'].transform([data['Region']])[0],
            'Province_encoded': label_encoders['Province'].transform([data['Province']])[0],
            'Culture_encoded': label_encoders['Culture'].transform([data['Culture']])[0]
        }

        # Convertir les données encodées en DataFrame
        df = pd.DataFrame([data_encoded])
        
        # Appliquer la standardisation aux caractéristiques
        df[features] = scaler_features.transform(df[features])

        # Faire la prédiction
        prediction = model.predict(df[features])[0]
        
        return jsonify({'prediction': float(prediction)})
    


st.set_page_config(
    page_title = "Migration Analysis by WASCAL",
    layout = "wide",
    initial_sidebar_state = "expanded"
)

#######################
# CSS styling
st.markdown("""
<style>

[data-testid="block-container"] {
    padding-left: 1rem;
    padding-right: 1rem;
    padding-top: 1rem;
    padding-bottom: 0rem;
    margin-bottom: -7rem;
}

[data-testid="stVerticalBlock"] {
    padding-left: 0rem;
    padding-right: 0rem;
}

[data-testid="stMetric"] {
    background-color: grey;
    text-align: center;
    padding: 15px 0;
}

[data-testid="stMetricLabel"] {
  display: flex;
  justify-content: center;
  align-items: center;
}

[data-testid="stMetricDeltaIcon-Up"] {
    position: relative;
    left: 38%;
    -webkit-transform: translateX(-50%);
    -ms-transform: translateX(-50%);
    transform: translateX(-50%);
}

[data-testid="stMetricDeltaIcon-Down"] {
    position: relative;
    left: 38%;
    -webkit-transform: translateX(-50%);
    -ms-transform: translateX(-50%);
    transform: translateX(-50%);
}

</style>
""", unsafe_allow_html=True)

def make_donut(input_response, input_text, input_color):
  
  if input_color == 'blue':
      chart_color = ['#29b5e8', '#155F7A']
  if input_color == 'green':
      chart_color = ['#27AE60', '#12783D']
  if input_color == 'orange':
      chart_color = ['#F39C12', '#875A12']
  if input_color == 'red':
      chart_color = ['#E74C3C', '#781F16']
    
  source = pd.DataFrame({
      "Topic": ['', input_text],
      "% value": [100-input_response, input_response]
  })
  source_bg = pd.DataFrame({
      "Topic": ['', input_text],
      "% value": [100, 0]
  })
    
  plot = alt.Chart(source).mark_arc(innerRadius=45, cornerRadius=25).encode(
      theta="% value",
      color= alt.Color("Topic:N",
                      scale=alt.Scale(
                          #domain=['A', 'B'],
                          domain=[input_text, ''],
                          # range=['#29b5e8', '#155F7A']),  # 31333F
                          range=chart_color),
                      legend=None),
  ).properties(width=130, height=130)
    
  text = plot.mark_text(align='center', color="#29b5e8", font="Lato", fontSize=32, fontWeight=700, fontStyle="italic").encode(text=alt.value(f'{input_response} %'))
  plot_bg = alt.Chart(source_bg).mark_arc(innerRadius=45, cornerRadius=20).encode(
      theta="% value",
      color= alt.Color("Topic:N",
                      scale=alt.Scale(
                          # domain=['A', 'B'],
                          domain=[input_text, ''],
                          range=chart_color),  # 31333F
                      legend=None),
  ).properties(width=130, height=130)
  return plot_bg + plot + text

df = pd.read_csv(r"migration_data.csv")
df = df.drop("Unnamed: 0", axis=1)

# Charger le fichier Shapefile avec geopandas
shapefile_path = r"./gis/BFA_adm2.shp"
gdf = gpd.read_file(shapefile_path)

#path = r"./data/Donnees_compilees.csv"
#data = pd.read_csv(path, sep=";")
consumption = pd.read_csv(r"Donnees_compilees.csv", sep=";" )
# consumption = consumption.drop("Unnamed: 0", axis=1)

with st.sidebar:
    st.title("Migration Analisis by WASCAL")
    selected_page = option_menu("Pages List", ["Generality", "Starting area", 'Consumption', 'Forcasting'], icons=['gear', 'house', '', 'world'], menu_icon="cast", default_index=0)
    st.write("TapsY")
    # selected


def generality():
    # Dashboard Main Panel
    # col = st.columns((1.5, 4.5, 2), gap='medium')
    st.markdown("## Generality / Study context")
    col = st.columns((1,2))
    with col[0]:
        st.markdown("### Number of migrants")
        st.metric(label="Migrants women", value=df['FEMININ'].sum(), delta="")
        st.metric(label="Migrants men", value=df['MASCULIN'].sum(), delta="")
        st.metric(label="Total migrants", value=df['total'].sum(), delta="")
    with col[1]:
        st.markdown("### Study Area (Burkina Faso)")
        # Créer une figure Matplotlib
        fig, ax = plt.subplots(figsize=(15, 10))
        # Plot avec Geopandas
        gdf.plot(ax=ax,column="NAME_1", cmap="Oranges", legend=True)
        ax.axis("off")
        # ax.set_title("Map of Burkina Faso")
        st.pyplot(fig)

    st.markdown("### Times Serie")
    times_serie = df.groupby(['annee_collecte', 'mois_num']).sum().reset_index()
    #times_serie = times_serie.drop(columns=['commune_depart', 'region_accueil', 'province_accueil', 'commune_accueil', 'province_depart'])
    times_serie = times_serie.sort_values(['annee_collecte', 'mois_num'])
    times_serie['date'] = pd.to_datetime(times_serie['annee_collecte'].astype(str) + '-' + times_serie['mois_num'].astype(str), format='%Y-%m')
    times_serie = times_serie.sort_values(by='date')
    times_serie.set_index('date')
    st.line_chart(times_serie.set_index('date')[['MASCULIN', 'FEMININ', 'total']], color=["#FF0000", "#0000FF", "#00FF00"])
        
    st.markdown("### Study data")
    st.dataframe(df)

def generality2():
    # Dashboard Main Panel
    # col = st.columns((1.5, 4.5, 2), gap='medium')
    st.markdown("## Generality / Study context")
    st.markdown("### Number of migrants")
    col = st.columns(3)
    with col[0]:
        st.metric(label="Migrants women", value=df['FEMININ'].sum(), delta="")
    with col[1]:
        st.metric(label="Migrants men", value=df['MASCULIN'].sum(), delta="")
    with col[2]:
        st.metric(label="Total migrants", value=df['total'].sum(), delta="")

    st.markdown("### Study Area (Burkina Faso)")
    # Créer une figure Matplotlib
    fig, ax = plt.subplots(figsize=(10, 10))
    # Plot avec Geopandas
    gdf.plot(ax=ax,column="NAME_1", cmap="Oranges", legend=True)
    ax.axis("off")
    ax.set_title("Map of Burkina Faso")
    st.pyplot(fig)
        
    st.markdown("### Study data")
    st.dataframe(df)


def starting():
    st.markdown("## Starting area")
    province = []
    commune = []
    col2 = st.columns(4)
    with col2[0]:
        province = st.multiselect("Province de départ ", df['province_depart'].unique())
        if(province != []):
            df_new = df[df['province_depart'].isin(province)]
        else:
            df_new = df
    with col2[1]:
        commune = st.multiselect("Commune de départ ", df_new['commune_depart'].unique())
        if(commune != []):
            df_new = df_new[df_new['commune_depart'].isin(commune)]
        else:
            df_new = df_new
    with col2[2]:
        annee = st.multiselect("Annee collectée ", df_new['annee_collecte'].unique())
        if(annee != []):
            df_new = df_new[df_new['annee_collecte'].isin(annee)]
        else:
            df_new = df_new
    with col2[3]:
        # Obtenez les valeurs uniques dans la colonne 'mois_num'
        valeurs_uniques_mois = df_new['mois_num'].unique()

        # Triez les valeurs uniques
        valeurs_uniques_mois.sort()
        mois = st.multiselect("Mois collectée ", valeurs_uniques_mois)
        if(mois != []):
            df_new = df_new[df_new['mois_num'].isin(mois)]
        else:
            df_new = df_new
    
    groupe_province = df_new.groupby(['province_depart']).sum().reset_index()
    # groupe_province = groupe_province.drop(columns=['commune_depart', 'region_accueil', 'province_accueil', 'commune_accueil', 'annee_collecte', 'mois_num'])

    # Streamlit App
    st.markdown('### Visualization of migration data by province of departure')
    st.write("These graphs show the total number of migrations by province of departure.")

    # Create bar chart
    st.write("Women and Men")
    st.bar_chart(groupe_province.drop('total', axis=1).set_index('province_depart'))
    st.write("All migrants")
    st.bar_chart(groupe_province.drop(columns=['MASCULIN', 'FEMININ']).set_index('province_depart'), color=['#FF6600', '#00FF00', '#0000FF'])
    
        #fig, ax = plt.subplots(figsize=(10, 8))
        #ax.hist(groupe_province, 30, stacked=True, density=False)
        #ax.set_xlabel('Province de depart')
        #ax.set_ylabel('Population')
        #ax.set_title('Répartition de la population immigrée')
        #ax.legend()
        #ax.set_title("Map of Burkina Faso")
        #st.pyplot(fig)
    # groupe_df = df.groupby(['annee_collecte', 'mois_num']).sum().reset_index()
    # Trier le DataFrame résultant en fonction de 'colonne1' et 'colonne2'
    # groupe_df_trie = groupe_df.sort_values(by=['annee_collecte', 'mois_num']).reset_index(drop=True)

    times_serie = df.groupby(['annee_collecte', 'mois_num']).sum().reset_index()
    # times_serie = times_serie.drop(columns=['commune_depart', 'region_accueil', 'province_accueil', 'commune_accueil', 'province_depart'])
    times_serie = times_serie.sort_values(['annee_collecte', 'mois_num'])
    times_serie.set_index(['annee_collecte', 'mois_num'], inplace=True)
    # st.dataframe(times_serie)
    
def Consumption() :
    #st.write(consumption)
    colc = st.columns(2)
    with colc[0]:
        # Graphique 1 : Production totale par région
        # st.subheader('Production totale par région')
        # production_region = consumption.groupby('Region')['Production (t)'].sum().reset_index()
        # fig1, ax1 = plt.subplots()
        # ax1.bar(production_region['Region'], production_region['Production (t)'])
        # plt.xticks(rotation=90)
        # st.pyplot(fig1)
        # Graphique circulaire 1 : Production totale par région
        st.subheader('Production totale par région')
        production_region = consumption.groupby('Region')['Production (t)'].sum().reset_index()
        fig1, ax1 = plt.subplots()
        ax1.pie(production_region['Production (t)'], labels=production_region['Region'], autopct='%1.1f%%', startangle=90)
        ax1.axis('equal')  # Assure que le pie chart est dessiné comme un cercle.
        st.pyplot(fig1)

        # Graphique 5 : Production totale par culture
        # st.subheader('Production totale par culture')
        # production_culture = consumption.groupby('Culture')['Production (t)'].sum().reset_index()
        # fig5, ax5 = plt.subplots()
        # ax5.bar(production_culture['Culture'], production_culture['Production (t)'])
        # plt.xticks(rotation=90)
        # st.pyplot(fig5)
    with colc[1]:
        # Graphique 4 : Population totale par région
        st.subheader('Population totale par région')
        population_region = consumption.groupby('Region')['Population'].sum().reset_index()
        fig4, ax4 = plt.subplots()
        ax4.bar(population_region['Region'], population_region['Population'])
        plt.xticks(rotation=90)
        st.pyplot(fig4)
    # Graphique : Production par culture par région
    st.subheader('Production par culture par région')
    # Pivot des données pour avoir les cultures comme colonnes, les régions comme index, et la production comme valeurs
    pivot_df = consumption.pivot_table(values='Production (t)', index='Region', columns='Culture', aggfunc='sum').fillna(0)
    # Création du graphique à barres groupées
    fig, ax = plt.subplots(figsize=(14, 8))
    # Définir le nombre de barres
    bar_width = 0.2
    # Définir les positions des barres
    positions = np.arange(len(pivot_df))
    # Tracer les barres pour chaque culture
    for i, culture in enumerate(pivot_df.columns):
        ax.bar(positions + i * bar_width, pivot_df[culture], bar_width, label=culture)
    # Ajouter les labels et le titre
    ax.set_xlabel('Région')
    ax.set_ylabel('Production (t)')
    ax.set_title('Production par culture par région')
    ax.set_xticks(positions + bar_width * (len(pivot_df.columns) - 1) / 2)
    ax.set_xticklabels(pivot_df.index, rotation=90)
    ax.legend()
    # Afficher le graphique dans Streamlit
    st.pyplot(fig)
    colc2 = st.columns(2)
    with colc2[0]:
        # Graphique 2 : Disponibilité par tête par région
        st.subheader('Disponibilité par tête par région')
        disponibilite_tete_region = consumption.groupby('Region')['Disponibilite/tete'].mean().reset_index()
        fig2, ax2 = plt.subplots()
        ax2.bar(disponibilite_tete_region['Region'], disponibilite_tete_region['Disponibilite/tete'])
        plt.xticks(rotation=90)
        st.pyplot(fig2)
    with colc2[1]:
        # Graphique 3 : Équivalent Kcal par région
        st.subheader('Équivalent Kcal par région')
        kcal_region = consumption.groupby('Region')['Equivalent Kcal'].sum().reset_index()
        fig3, ax3 = plt.subplots()
        ax3.bar(kcal_region['Region'], kcal_region['Equivalent Kcal'])
        plt.xticks(rotation=90)
        st.pyplot(fig3)

def Forcasting():
    # Interface utilisateur Streamlit
    st.title('Prediction of crop availability')
    colF = st.columns(4)
    with colF[0]:
        annee = st.number_input('Année', min_value=2018, max_value=2100, value=2023)
    with colF[3]:
        cultures = ['SORGHO', 'MAIS', 'MIL', 'RIZ', 'NIEBE']
        culture = st.selectbox('Culture', cultures)
    with colF[1]:
        # Dictionnaire complet pour les régions et les provinces
        regions = {
            'Boucle du Mouhoun': ['Balé', 'Banwa', 'Kossi', 'Mouhoun', 'Nayala', 'Sourou'],
            'Cascades': ['Comoé', 'Léraba'],
            'Centre': ['Kadiogo'],
            'Centre Est': ['Boulgou', 'Koulpélogo', 'Kouritenga'],
            'Centre Nord': ['Bam', 'Namentenga', 'Sanmatenga'],
            'Centre Ouest': ['Boulkiemdé', 'Sanguié', 'Sissili', 'Ziro'],
            'Centre Sud': ['Bazèga', 'Nahouri', 'Zoundwéogo'],
            'Est': ['Gnagna', 'Gourma', 'Komondjari', 'Kompienga', 'Tapoa'],
            'Hauts Bassins': ['Houet', 'Kénédougou', 'Tuy'],
            'Nord': ['Loroum', 'Passoré', 'Yatenga', 'Zondoma'],
            'Plateau Central': ['Ganzourgou', 'Kourwéogo', 'Oubritenga'],
            'Sahel': ['Oudalan', 'Séno', 'Soum', 'Yagha'],
            'Sud-Ouest': ['Bougouriba', 'Ioba', 'Noumbiel', 'Poni'],
            # Ajoutez d'autres régions et provinces ici si nécessaire
        }
        region = st.selectbox('Région', list(regions.keys()))
    with colF[2]:
        if region:
            province = st.selectbox('Province', regions[region])
        else:
            province = st.selectbox('Province', [])

    if st.button('Prédire maintenant'):
        data_encoded = {
            'Annee': annee,
            'Region': region,
            'Province': province,
            'Culture': culture
        }
        data_encoded_pop = {
            'Annee': annee,
            'Region': region,
            'Province': province,
        }
        data_encoded_pdi = {
            'Annee': annee,
            'Region': region,
            'Province': province,
        }

        # Encoder les valeurs d'entrée // disponibiltes
        data_encoded['Region_encoded'] = label_encoders['Region'].transform([region])[0]
        data_encoded['Province_encoded'] = label_encoders['Province'].transform([province])[0]
        data_encoded['Culture_encoded'] = label_encoders['Culture'].transform([culture])[0]

        # Encoder les valeurs d'entrée // population
        data_encoded_pop['Region_encoded'] = label_encoders_pop['Region'].transform([region])[0]
        data_encoded_pop['Province_encoded'] = label_encoders_pop['Province'].transform([province])[0]

        # Encoder les valeurs d'entrée // PDI
        data_encoded_pdi['Region_encoded'] = label_encoders_pdi['Region'].transform([region])[0]
        data_encoded_pdi['Province_encoded'] = label_encoders_pdi['Province'].transform([province])[0]

        # Convertir les données encodées en DataFrame
        df = pd.DataFrame([data_encoded])
        df_pop = pd.DataFrame([data_encoded_pop])
        df_pdi = pd.DataFrame([data_encoded_pdi])

        # Appliquer la standardisation aux caractéristiques
        df[features] = scaler_features.transform(df[features])
        df_pop[features_p] = scaler_features_pop.transform(df_pop[features_p])
        df_pdi[features_p] = scaler_features_pdi.transform(df_pdi[features_p])

        # Faire la prédiction
        prediction = model.predict(df[features])[0]
        prediction_pop = model_pop.predict(df_pop[features_p])[0]
        prediction_pdi = model_pdi.predict(df_pdi[features_p])[0]

        st.success(f'Crop availability prediction : {float(prediction):.2f}')
        st.success(f'Population prediction in the area : {float(prediction_pop):.2f}')
        st.success(f'Prediction of PDI in the area : {float(prediction_pdi):.2f}')


# Show page
if(selected_page == "Generality"):
    generality()
elif(selected_page == "Starting area"):
    #
    starting()
elif(selected_page == 'Land of welcome'):
    #
    st.markdown("# Land of welcome")
elif(selected_page == 'Consumption'):
    #
    st.markdown("# Consumption")
    Consumption()
elif(selected_page == 'Forcasting'):
    #
    st.markdown("# Forcasting")
    Forcasting()