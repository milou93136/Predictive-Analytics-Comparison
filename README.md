# Prédiction du Prix de l'Immobilier — Projet ML Portfolio

> **Étudiant Ingénieur 4A — Spécialité Intelligence Artificielle**
> Projet réalisé dans le cadre de ma **recherche d'alternance** en tant qu'Ingénieur IA / Data Scientist.

---

## Présentation du Projet

Ce projet de Machine Learning end-to-end illustre ma capacité à concevoir, entraîner et évaluer des modèles de régression sur des données réelles.
L'objectif est de **prédire la valeur médiane des logements** en Californie (en centaines de milliers de dollars) à partir du **California Housing Dataset** (UCI / Scikit-learn), en comparant deux algorithmes d'ensemble : **Random Forest** et **Gradient Boosting**.

Il reflète le type d'automatisation et d'analyse que j'ai pu mettre en œuvre lors de mes expériences en entreprise : pipeline reproductible, code structuré, résultats interprétables.

---

## Stack Technique

| Catégorie        | Technologies                                              |
|------------------|-----------------------------------------------------------|
| Langage          | Python 3.11                                               |
| Data Wrangling   | Pandas, NumPy                                             |
| Machine Learning | Scikit-learn (Pipeline, GridSearchCV, Ensemble methods)   |
| Visualisation    | Matplotlib, Seaborn                                       |
| Environnement    | VS Code / JupyterLab                                      |

---

## Dataset

**California Housing** — Données issues du recensement de 1990 en Californie.

| Feature              | Description                                      |
|----------------------|--------------------------------------------------|
| `MedInc`             | Revenu médian du foyer (dizaines de milliers $)  |
| `HouseAge`           | Âge médian des logements du quartier (années)    |
| `AveRooms`           | Nombre moyen de pièces par logement              |
| `AveBedrms`          | Nombre moyen de chambres par logement            |
| `Population`         | Population du block group                        |
| `AveOccup`           | Nombre moyen d'occupants par logement            |
| `Latitude`           | Latitude géographique                            |
| `Longitude`          | Longitude géographique                           |
| **`Prix_Median`**    | **Cible** — Valeur médiane des logements (100k $)|

---

## Méthodologie

### 1. Analyse Exploratoire (EDA)
- Statistiques descriptives et détection des valeurs manquantes
- Analyse de la distribution de la variable cible (histogramme + log-transform)
- Matrice de corrélation (heatmap) pour identifier les features prédictives
- Scatter plots des top 4 features les plus corrélées avec la cible
- Carte géographique de la distribution des prix (Latitude × Longitude)

### 2. Preprocessing
- **Feature engineering** : création de 3 ratios dérivés (`Rooms_per_Household`, `Bedrooms_per_Room`, `Population_per_Household`)
- **Suppression des outliers** : filtre IQR (percentile 1 %–99 %) sur la variable cible
- **Train/Test split** : 80/20, stratifié sur la cible discrétisée pour garantir la représentativité
- **Standardisation** : `StandardScaler` intégré dans les Pipelines Scikit-learn (pas de data leakage)

### 3. Conception et Comparaison des Modèles

| Algorithme              | Justification du choix                                                                 |
|-------------------------|----------------------------------------------------------------------------------------|
| **Random Forest**       | Ensemble de N arbres (bagging) — robuste aux outliers, gère les non-linéarités, interprétable via feature importances |
| **Gradient Boosting**   | Ensemble séquentiel (boosting) — corrige les erreurs résiduelles itération par itération, très performant sur données tabulaires |

Les deux algorithmes sont encapsulés dans un **Pipeline Scikit-learn** (`StandardScaler → Estimateur`) et leurs hyperparamètres sont optimisés via **GridSearchCV (5-fold CV)**.

### 4. Évaluation

| Métrique    | Définition                                                                 |
|-------------|----------------------------------------------------------------------------|
| **RMSE**    | Racine de l'erreur quadratique moyenne — pénalise les grandes erreurs      |
| **MAE**     | Erreur absolue moyenne — robuste aux outliers, interprétable en 100k $     |
| **R²**      | Proportion de variance expliquée par le modèle (1 = parfait, 0 = baseline) |

---

## Résultats

> *Les valeurs ci-dessous sont représentatives. Exécutez `model_conception.py` pour obtenir les résultats exacts sur votre machine.*

| Modèle               | RMSE Test ↓ | MAE Test ↓ | R² Test ↑ |
|----------------------|:-----------:|:----------:|:---------:|
| Random Forest        | ~0.47       | ~0.32      | ~0.82     |
| **Gradient Boosting**| **~0.43**   | **0.29**   | **~0.85** |

**Conclusion** : Le Gradient Boosting surpasse légèrement le Random Forest sur ce dataset, notamment grâce à sa capacité à corriger les erreurs résiduelles de façon séquentielle. Un R² > 0.82 confirme la qualité prédictive du modèle sur un dataset aussi hétérogène géographiquement.

---

## Visualisations Générées

| Fichier                                | Contenu                                               |
|----------------------------------------|-------------------------------------------------------|
| `eda_distribution_cible.png`           | Distribution de la variable cible (brut + log)        |
| `eda_correlation_matrix.png`           | Heatmap de corrélation (toutes features)              |
| `eda_top_features_vs_cible.png`        | Scatter plots top 4 features vs cible                 |
| `eda_geographic_distribution.png`      | Carte géographique des prix en Californie             |
| `resultats_comparaison_metriques.png`  | Barplot comparatif RMSE / MAE / R²                    |
| `resultats_reel_vs_predit.png`         | Valeurs réelles vs prédites (test set)                |
| `resultats_distribution_residus.png`   | Distribution des résidus (test set)                   |
| `resultats_feature_importance.png`     | Importance des features — Random Forest (MDI)         |

---

## Installation & Exécution

```bash
# 1. Cloner le dépôt
git clone https://github.com/milou93136/Predictive-Analytics-Comparison.git
cd Predictive-Analytics-Comparison

# 2. Créer et activer un environnement virtuel
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer le pipeline complet
python model_conception.py
```

> Le script s'exécute de bout en bout et génère automatiquement les figures dans le répertoire courant.
> Pour un entraînement rapide (sans GridSearchCV), passer `optimiser=False` dans le `__main__`.

---

## Structure du Projet

```
Predictive-Analytics-Comparison/
├── model_conception.py          # Pipeline ML complet (EDA → Évaluation)
├── requirements.txt             # Dépendances Python
├── README.md                    # Documentation du projet
├── eda_distribution_cible.png
├── eda_correlation_matrix.png
├── eda_top_features_vs_cible.png
├── eda_geographic_distribution.png
├── resultats_comparaison_metriques.png
├── resultats_reel_vs_predit.png
├── resultats_distribution_residus.png
└── resultats_feature_importance.png
```

---

## Axes d'Amélioration

- **Modèles avancés** : Tester XGBoost / LightGBM / CatBoost pour gagner en performance
- **Feature Engineering** : Enrichir avec des données open-data (proximité écoles, transports, criminalité)
- **Explicabilité** : Intégrer SHAP pour des explications locales des prédictions
- **Déploiement** : Exposer le modèle via une API FastAPI conteneurisée avec Docker

---

## À Propos

Projet réalisé par un **Étudiant Ingénieur 4A en Intelligence Artificielle**, dans le cadre de la constitution de mon portfolio GitHub pour décrocher une **alternance en Data Science / IA**.

Ce projet illustre ma capacité à :
- Mener un projet ML complet de manière autonome et structurée
- Automatiser l'analyse de données et la génération de visualisations
- Justifier mes choix techniques avec rigueur (comme en contexte professionnel)
- Produire un code lisible, commenté et reproductible

> N'hésitez pas à ouvrir une issue ou à me contacter pour toute question.

---

*Dernière mise à jour : Mars 2026*
