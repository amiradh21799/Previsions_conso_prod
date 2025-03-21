# Prévisions Conso Prod

Ce repository contient deux scripts principaux qui permettent de traiter des données de consommation et de production énergétique, et de réaliser des prévisions à l'aide de modèles LSTM.

## Contenu du Repository

- **Prevision_conso.py**  
  Ce script :
  - Charge et nettoie les données de consommation depuis `conso_historiques_clean.csv`.
  - Prépare les données (normalisation, création de séquences temporelles).
  - Construit et entraîne un modèle LSTM multi-sortie pour prédire plusieurs indicateurs de puissance.
  - Effectue des prédictions, inverse l'échelle des données et affiche des graphiques comparant les valeurs réelles et prédites.
  - Génère un fichier CSV (`test_vs_real.csv`) contenant les dates et les valeurs réelles et prédites pour une analyse ultérieure.

- **Prevision_prod.py**  
  Ce script :
  - Charge des données météorologiques et historiques de production photovoltaïque depuis `pv_prev_meteo_clean.csv` et `pv_historiques_clean.csv`.
  - Réalise un traitement et un agrégat horaire des données, puis fusionne les jeux de données.
  - Prépare les features (y compris des variables temporelles et des lags) et normalise les données.
  - Construit et entraîne un modèle LSTM bidirectionnel pour la prévision de la production photovoltaïque.
  - Évalue le modèle (MAE, RMSE, R2) et génère des visualisations (courbes d'apprentissage, comparaisons entre valeurs réelles et prédites, distribution des erreurs).
  - Enregistre les résultats dans un fichier CSV (`test_vs_pred_timewindow.csv`).

## Prérequis

Assurez-vous d'avoir installé les bibliothèques Python suivantes :

- [pandas](https://pandas.pydata.org/)
- [numpy](https://numpy.org/)
- [matplotlib](https://matplotlib.org/)
- [scikit-learn](https://scikit-learn.org/)
- [tensorflow](https://www.tensorflow.org/) (inclut Keras)

Vous pouvez installer ces dépendances via pip :

```bash
pip install pandas numpy matplotlib scikit-learn tensorflow
