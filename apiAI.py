import json
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.preprocessing import StandardScaler
import pandas as pd
from flask import Flask, jsonify

app = Flask(__name__)

# Charger les données de déchets et de zones industrielles
Dechets = pd.read_json('dechets.json')
with open('zoneIndus.json', 'r') as fichier:
    zoneIndus = json.load(fichier)

# Prétraitement des données de déchets 
Dechets['Gouvernorat'] = Dechets['Gouvernorat'].str.lower()

# Séparer 
X = Dechets.drop(['Type dechets', 'Gouvernorat'], axis=1).values
y = Dechets['Gouvernorat'].values

# Normalisation des features
norme = StandardScaler()
X = norme.fit_transform(X)

# Recherche des meilleurs hyperparamètres pour le modèle RandomForestClassifier
parametres = {'n_estimators': [50, 100, 200],
              'max_depth': [None, 10, 20]}

modele = RandomForestClassifier(random_state=42)
cvStratifie = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42) 
espaceRech = GridSearchCV(modele, param_grid=parametres, cv=cvStratifie)
espaceRech.fit(X, y)

# Meilleur modèle après la recherche des hyperparamètres
meilleurModele = espaceRech.best_estimator_

@app.route('/MeteoEtZone/<ville>', methods=['GET'])
def MeteoEtZone(ville):
    villeMin = ville.lower()

    # Obtenir les données météorologiques de la ville spécifiée
    meteoVille = getDonneesMeteo(ville)

    if meteoVille.get('Erreur'):
        return jsonify({"Erreur": meteoVille["Erreur"]})

    # Préparer les données pour la prédiction du modèle IA
    X_pred = Dechets[Dechets['Gouvernorat'] == villeMin].drop(['Type dechets', 'Gouvernorat'], axis=1).values
    X_pred = norme.transform(X_pred)

    # Prédire les gouvernorats 
    gouvernoratsPred = meilleurModele.predict(X_pred)

    # Compter le nombre d'occurrences de chaque gouvernorat prédit
    occurences = {}
    for gouvernorat in gouvernoratsPred:
        occurences[gouvernorat] = occurences.get(gouvernorat, 0) + 1

    # Sélectionner le gouvernorat avec le plus grand nombre d'occurrences
    gouvernoratPlusPredicte = max(occurences, key=occurences.get)

    # Filtrer les informations sur les déchets 
    dechetsPredits = Dechets[Dechets['Gouvernorat'].str.lower() == gouvernoratPlusPredicte.lower()]

    # Filtrer les informations sur la zone industrielle 
    estZoneIndustrielle = any(villeMin == zone['Zone'].lower() for zone in zoneIndus['Zones_industrielles'])

    # Créer la réponse JSON
    resultat = {
        "infosDechets": dechetsPredits.to_dict(orient='records'),
        "estZoneIndustrielle": estZoneIndustrielle,
        "meteo": meteoVille,
        "gouvernoratsPred": gouvernoratsPred.tolist()
    }

    return jsonify(resultat)

def getDonneesMeteo(ville):
    apiCle = "9c27d097c7b46ce34ad6b105aba64bac"
    meteoUrl = f"https://api.openweathermap.org/data/2.5/weather?q={ville}&appid={apiCle}"
    repMeteo = requests.get(meteoUrl)
    if repMeteo.status_code == 200:
        donneesMeteo = repMeteo.json()
        return {
            "temp": donneesMeteo["main"]["temp"],
            "humidite": donneesMeteo["main"]["humidity"],
            "rapiditeVent": donneesMeteo["wind"]["speed"],
        }
    else:
        return {"Erreur pendant la récupération des données météorologiques"}

if __name__ == '__main__':
    app.run(debug=True)
