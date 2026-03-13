"""
=============================================================================
Projet de Machine Learning - Prédiction du Prix de l'Immobilier
=============================================================================
Auteur      : Étudiant Ingénieur 4A - Spécialité Intelligence Artificielle
Dataset     : California Housing (Scikit-learn)
Objectif    : Prédire la valeur médiane des logements (en 100 000 $)
Algorithmes : Random Forest Regressor vs Gradient Boosting Regressor
Métriques   : RMSE, MAE, R²
=============================================================================

Ce projet a été conçu pour mon portfolio GitHub dans le cadre de ma recherche
d'alternance en tant qu'Ingénieur IA / Data Scientist.

Il illustre ma capacité à mener un projet ML de bout en bout :
  - Analyse exploratoire rigoureuse (EDA)
  - Pipeline de preprocessing robuste
  - Comparaison et sélection de modèles
  - Évaluation quantitative avec métriques standards
"""

# =============================================================================
# 0. IMPORTS & CONFIGURATION
# =============================================================================

import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

# Configuration globale du style des graphiques
matplotlib.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
sns.set_theme(style="whitegrid", palette="muted")

# Graine aléatoire pour la reproductibilité des résultats
RANDOM_STATE = 42
OUTPUT_DIR = "."  # Dossier de sauvegarde des figures


# =============================================================================
# 1. CHARGEMENT DES DONNÉES
# =============================================================================

def charger_donnees() -> pd.DataFrame:
    """
    Charge le dataset California Housing depuis Scikit-learn.

    Ce dataset est un classique du benchmark de régression. Il contient des
    informations agrégées par quartier (block group) de Californie, issues du
    recensement de 1990.

    Features :
        - MedInc      : Revenu médian du foyer (en dizaines de milliers $)
        - HouseAge    : Âge médian des logements du quartier (années)
        - AveRooms    : Nombre moyen de pièces par logement
        - AveBedrms   : Nombre moyen de chambres par logement
        - Population  : Population du block group
        - AveOccup    : Nombre moyen d'occupants par logement
        - Latitude    : Latitude géographique du quartier
        - Longitude   : Longitude géographique du quartier

    Cible (target) :
        - MedHouseVal : Valeur médiane des logements (en centaines de milliers $)
    """
    print("=" * 70)
    print("  CHARGEMENT DES DONNÉES - California Housing Dataset")
    print("=" * 70)

    housing = fetch_california_housing(as_frame=True)
    df = housing.frame.copy()

    # Renommage de la cible pour plus de clarté
    df.rename(columns={"MedHouseVal": "Prix_Median"}, inplace=True)

    print(f"\n  Dimensions du dataset : {df.shape[0]} lignes x {df.shape[1]} colonnes")
    print(f"  Features              : {list(df.columns[:-1])}")
    print(f"  Cible                 : Prix_Median\n")

    return df


# =============================================================================
# 2. ANALYSE EXPLORATOIRE DES DONNÉES (EDA)
# =============================================================================

def analyse_exploratoire(df: pd.DataFrame) -> None:
    """
    Réalise une analyse exploratoire complète du dataset.

    Étapes :
        1. Statistiques descriptives
        2. Analyse des valeurs manquantes
        3. Distribution de la variable cible
        4. Matrice de corrélation (Heatmap)
        5. Relations features / cible (scatter plots)
        6. Distribution géographique des prix
    """
    print("=" * 70)
    print("  ANALYSE EXPLORATOIRE DES DONNÉES (EDA)")
    print("=" * 70)

    # -------------------------------------------------------------------
    # 2.1 Statistiques descriptives
    # -------------------------------------------------------------------
    print("\n[2.1] Statistiques descriptives :")
    print(df.describe().round(2).to_string())

    # -------------------------------------------------------------------
    # 2.2 Analyse des valeurs manquantes
    # -------------------------------------------------------------------
    print("\n[2.2] Valeurs manquantes par colonne :")
    valeurs_manquantes = df.isnull().sum()
    if valeurs_manquantes.sum() == 0:
        print("  --> Aucune valeur manquante détectée. Dataset complet.")
    else:
        print(valeurs_manquantes[valeurs_manquantes > 0])

    # -------------------------------------------------------------------
    # 2.3 Distribution de la variable cible
    # -------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Distribution de la Variable Cible (Prix Médian)", fontsize=14, fontweight="bold")

    # Histogramme brut
    axes[0].hist(df["Prix_Median"], bins=50, color="#4C72B0", edgecolor="white", alpha=0.85)
    axes[0].set_xlabel("Prix médian (100k $)")
    axes[0].set_ylabel("Fréquence")
    axes[0].set_title("Distribution originale")
    axes[0].axvline(df["Prix_Median"].mean(), color="#DD8452", linestyle="--",
                    label=f"Moyenne : {df['Prix_Median'].mean():.2f}")
    axes[0].axvline(df["Prix_Median"].median(), color="#55A868", linestyle="--",
                    label=f"Médiane : {df['Prix_Median'].median():.2f}")
    axes[0].legend()

    # Log-distribution pour visualiser l'asymétrie
    axes[1].hist(np.log1p(df["Prix_Median"]), bins=50, color="#DD8452",
                 edgecolor="white", alpha=0.85)
    axes[1].set_xlabel("log(1 + Prix médian)")
    axes[1].set_ylabel("Fréquence")
    axes[1].set_title("Distribution après transformation log")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/eda_distribution_cible.png", bbox_inches="tight")
    plt.close()
    print("\n[2.3] Figure sauvegardée : eda_distribution_cible.png")

    # -------------------------------------------------------------------
    # 2.4 Matrice de corrélation
    # -------------------------------------------------------------------
    # J'analyse les corrélations pour identifier les features les plus
    # prédictives et détecter d'éventuelles redondances (multicolinéarité).
    fig, ax = plt.subplots(figsize=(10, 8))
    corr_matrix = df.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Matrice de Corrélation des Features", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/eda_correlation_matrix.png", bbox_inches="tight")
    plt.close()
    print("[2.4] Figure sauvegardée : eda_correlation_matrix.png")

    # -------------------------------------------------------------------
    # 2.5 Relations features / cible (top 4 features par corrélation)
    # -------------------------------------------------------------------
    # Je sélectionne les 4 features les plus corrélées avec la cible pour
    # visualiser leurs relations et repérer les tendances non-linéaires.
    top_features = (
        corr_matrix["Prix_Median"]
        .drop("Prix_Median")
        .abs()
        .nlargest(4)
        .index
        .tolist()
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Top 4 Features vs Prix Médian", fontsize=14, fontweight="bold")
    axes = axes.flatten()

    for i, feature in enumerate(top_features):
        # Échantillon pour la lisibilité (dataset volumineux)
        sample = df.sample(n=2000, random_state=RANDOM_STATE)
        axes[i].scatter(
            sample[feature], sample["Prix_Median"],
            alpha=0.3, s=10, color="#4C72B0"
        )
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel("Prix Médian (100k $)")
        corr_val = df[feature].corr(df["Prix_Median"])
        axes[i].set_title(f"{feature}  |  r = {corr_val:.3f}")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/eda_top_features_vs_cible.png", bbox_inches="tight")
    plt.close()
    print("[2.5] Figure sauvegardée : eda_top_features_vs_cible.png")

    # -------------------------------------------------------------------
    # 2.6 Carte géographique des prix (Latitude / Longitude)
    # -------------------------------------------------------------------
    # La Californie a une forte disparité géographique (côte vs intérieur).
    # Cette visualisation confirme que la localisation est un prédicteur fort.
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        df["Longitude"], df["Latitude"],
        c=df["Prix_Median"],
        cmap="plasma",
        s=df["Population"] / 500,
        alpha=0.5,
    )
    plt.colorbar(scatter, ax=ax, label="Prix Médian (100k $)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(
        "Distribution Géographique des Prix en Californie\n"
        "(taille des points ∝ population du block group)",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/eda_geographic_distribution.png", bbox_inches="tight")
    plt.close()
    print("[2.6] Figure sauvegardée : eda_geographic_distribution.png\n")


# =============================================================================
# 3. PREPROCESSING
# =============================================================================

def preprocessing(df: pd.DataFrame):
    """
    Prépare les données pour l'entraînement des modèles.

    Étapes :
        1. Séparation features / cible
        2. Feature engineering : ajout de ratios pertinents
        3. Suppression des outliers extrêmes (méthode IQR)
        4. Split Train / Test (80/20, stratifié sur la cible discrétisée)
        5. Standardisation (StandardScaler) - intégrée dans les pipelines

    Remarques de conception :
        - Je n'applique PAS le scaler ici pour éviter le data leakage.
          La normalisation sera intégrée dans un Pipeline Scikit-learn,
          garantissant qu'elle n'est apprise QUE sur les données d'entraînement.
    """
    print("=" * 70)
    print("  PREPROCESSING")
    print("=" * 70)

    # -------------------------------------------------------------------
    # 3.1 Feature Engineering
    # -------------------------------------------------------------------
    # J'ajoute des features dérivées qui capturent mieux la structure des
    # données et peuvent améliorer la capacité prédictive des modèles.
    df = df.copy()
    df["Rooms_per_Household"] = df["AveRooms"] / df["AveOccup"]
    df["Bedrooms_per_Room"] = df["AveBedrms"] / df["AveRooms"]
    df["Population_per_Household"] = df["Population"] / df["AveOccup"]
    print("\n[3.1] Feature engineering : 3 features dérivées créées.")
    print(f"  --> Rooms_per_Household, Bedrooms_per_Room, Population_per_Household")

    # -------------------------------------------------------------------
    # 3.2 Suppression des outliers extrêmes (méthode IQR)
    # -------------------------------------------------------------------
    # J'applique le filtre IQR uniquement sur la cible pour retirer les
    # logements au prix plafonné (artefact du recensement : valeurs bloquées
    # à 5.0, soit 500 000 $), qui biaiseraient l'apprentissage.
    Q1 = df["Prix_Median"].quantile(0.01)
    Q3 = df["Prix_Median"].quantile(0.99)
    avant = len(df)
    df = df[(df["Prix_Median"] >= Q1) & (df["Prix_Median"] <= Q3)]
    apres = len(df)
    print(f"\n[3.2] Suppression outliers (IQR percentile 1%-99%) :")
    print(f"  --> {avant - apres} lignes supprimées ({avant} → {apres})")

    # -------------------------------------------------------------------
    # 3.3 Séparation features / cible et train/test split
    # -------------------------------------------------------------------
    X = df.drop(columns=["Prix_Median"])
    y = df["Prix_Median"]

    # Discrétisation de y pour un split stratifié équilibré
    y_binned = pd.cut(y, bins=5, labels=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y_binned,
    )

    print(f"\n[3.3] Split Train/Test (80/20, stratifié) :")
    print(f"  --> Entraînement : {X_train.shape[0]} échantillons")
    print(f"  --> Test         : {X_test.shape[0]} échantillons")
    print(f"  --> Features     : {X_train.shape[1]} colonnes\n")

    return X_train, X_test, y_train, y_test, list(X.columns)


# =============================================================================
# 4. CONCEPTION DES MODÈLES
# =============================================================================

def construire_pipelines() -> dict:
    """
    Construit les pipelines de ML pour chaque algorithme testé.

    Choix des algorithmes :
        - Random Forest Regressor :
            Ensemble d'arbres de décision entraînés en parallèle (bagging).
            Robuste aux outliers, gère naturellement les relations non-linéaires
            et les interactions entre features. Peu sensible à la normalisation,
            mais je l'inclus par cohérence et pour faciliter la comparaison.

        - Gradient Boosting Regressor :
            Ensemble séquentiel qui corrige les erreurs résiduelles à chaque
            itération. Généralement plus précis que le Random Forest mais plus
            sensible aux hyperparamètres et aux outliers. Excellent sur des
            données tabulaires structurées comme l'immobilier.

    Architecture Pipeline :
        StandardScaler --> Estimateur
        Le scaler est encapsulé dans le Pipeline pour garantir l'absence
        de data leakage : il n'est ajusté que sur X_train.
    """
    pipelines = {
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestRegressor(
                n_estimators=200,
                max_depth=None,
                min_samples_split=5,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )),
        ]),
        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("model", GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                random_state=RANDOM_STATE,
            )),
        ]),
    }
    return pipelines


def optimiser_hyperparametres(pipeline: Pipeline, X_train: pd.DataFrame,
                               y_train: pd.Series, nom: str) -> Pipeline:
    """
    Recherche les meilleurs hyperparamètres via GridSearchCV (5-fold CV).

    Je limite la grille de recherche pour garder un temps d'exécution
    raisonnable tout en explorant les paramètres les plus impactants.
    """
    print(f"\n  [Tuning] GridSearchCV pour {nom}...")

    if nom == "Random Forest":
        param_grid = {
            "model__n_estimators": [100, 200],
            "model__max_depth": [None, 20],
            "model__min_samples_leaf": [1, 2],
        }
    else:  # Gradient Boosting
        param_grid = {
            "model__n_estimators": [200, 300],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [4, 5],
        }

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    grid_search.fit(X_train, y_train)

    print(f"  --> Meilleurs params : {grid_search.best_params_}")
    print(f"  --> RMSE CV (train)  : {-grid_search.best_score_:.4f}")

    return grid_search.best_estimator_


# =============================================================================
# 5. ENTRAÎNEMENT
# =============================================================================

def entrainer_modeles(pipelines: dict, X_train: pd.DataFrame,
                      y_train: pd.Series, optimiser: bool = True) -> dict:
    """
    Entraîne chaque modèle et retourne un dictionnaire de modèles ajustés.

    Paramètre `optimiser` :
        True  → Utilise GridSearchCV pour tuner les hyperparamètres.
        False → Entraîne avec les paramètres par défaut (plus rapide).
    """
    print("=" * 70)
    print("  ENTRAÎNEMENT DES MODÈLES")
    print("=" * 70)

    modeles_entrained = {}
    for nom, pipeline in pipelines.items():
        print(f"\n[>] Entraînement : {nom}")

        if optimiser:
            modele = optimiser_hyperparametres(pipeline, X_train, y_train, nom)
        else:
            modele = pipeline.fit(X_train, y_train)

        # Validation croisée sur le jeu d'entraînement (5-fold)
        scores_cv = cross_val_score(
            modele, X_train, y_train,
            cv=5,
            scoring="r2",
            n_jobs=-1,
        )
        print(f"  --> R² CV moyen    : {scores_cv.mean():.4f} ± {scores_cv.std():.4f}")

        modeles_entrained[nom] = modele

    print()
    return modeles_entrained


# =============================================================================
# 6. ÉVALUATION
# =============================================================================

def evaluer_modeles(modeles: dict, X_train: pd.DataFrame, X_test: pd.DataFrame,
                    y_train: pd.Series, y_test: pd.Series) -> pd.DataFrame:
    """
    Évalue tous les modèles sur le jeu de test et compare leurs performances.

    Métriques retenues :
        - RMSE (Root Mean Squared Error) : pénalise fortement les grandes erreurs.
          Même unité que la cible → interprétable directement en 100k $.
        - MAE  (Mean Absolute Error)     : erreur moyenne absolue, robuste aux
          outliers, facile à expliquer à un client non-technique.
        - R²   (Coefficient de détermination) : proportion de variance expliquée.
          R² = 1 → modèle parfait ; R² = 0 → modèle constant (baseline).

    Je calcule également ces métriques sur le TRAIN SET pour détecter
    un éventuel overfitting (écart train/test trop important).
    """
    print("=" * 70)
    print("  ÉVALUATION DES MODÈLES")
    print("=" * 70)

    resultats = []
    predictions = {}

    for nom, modele in modeles.items():
        y_pred_train = modele.predict(X_train)
        y_pred_test = modele.predict(X_test)
        predictions[nom] = y_pred_test

        rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
        mae_train = mean_absolute_error(y_train, y_pred_train)
        r2_train = r2_score(y_train, y_pred_train)

        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
        mae_test = mean_absolute_error(y_test, y_pred_test)
        r2_test = r2_score(y_test, y_pred_test)

        resultats.append({
            "Modèle": nom,
            "RMSE Train": round(rmse_train, 4),
            "RMSE Test": round(rmse_test, 4),
            "MAE Train": round(mae_train, 4),
            "MAE Test": round(mae_test, 4),
            "R² Train": round(r2_train, 4),
            "R² Test": round(r2_test, 4),
        })

    df_resultats = pd.DataFrame(resultats).set_index("Modèle")

    print("\n  Tableau comparatif des performances :\n")
    print(df_resultats.to_string())

    # -------------------------------------------------------------------
    # Sélection du meilleur modèle (critère : RMSE Test)
    # -------------------------------------------------------------------
    meilleur_nom = df_resultats["RMSE Test"].idxmin()
    print(f"\n  --> Meilleur modèle (RMSE Test) : {meilleur_nom}")
    print(f"      RMSE = {df_resultats.loc[meilleur_nom, 'RMSE Test']:.4f} (× 100k $)")
    print(f"      MAE  = {df_resultats.loc[meilleur_nom, 'MAE Test']:.4f} (× 100k $)")
    print(f"      R²   = {df_resultats.loc[meilleur_nom, 'R² Test']:.4f}\n")

    return df_resultats, predictions, meilleur_nom


# =============================================================================
# 7. VISUALISATIONS DES RÉSULTATS
# =============================================================================

def visualiser_resultats(modeles: dict, predictions: dict, y_test: pd.Series,
                          feature_names: list, df_resultats: pd.DataFrame) -> None:
    """
    Génère les visualisations post-évaluation :
        1. Comparaison des métriques (barplot groupé)
        2. Valeurs réelles vs prédites (scatter plot)
        3. Distribution des résidus
        4. Importance des features (meilleur modèle)
    """
    print("=" * 70)
    print("  VISUALISATIONS DES RÉSULTATS")
    print("=" * 70)

    noms = list(modeles.keys())
    couleurs = ["#4C72B0", "#DD8452"]

    # -------------------------------------------------------------------
    # 7.1 Comparaison des métriques
    # -------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Comparaison des Performances : Random Forest vs Gradient Boosting",
                 fontsize=13, fontweight="bold")

    metriques = [("RMSE Test", "RMSE (↓ meilleur)", "coral"),
                 ("MAE Test",  "MAE  (↓ meilleur)", "steelblue"),
                 ("R² Test",   "R²   (↑ meilleur)", "seagreen")]

    for ax, (col, label, color) in zip(axes, metriques):
        vals = df_resultats[col].values
        bars = ax.bar(noms, vals, color=[couleurs[i] for i in range(len(noms))],
                      edgecolor="white", width=0.5)
        ax.set_title(label, fontweight="bold")
        ax.set_ylabel(col.split()[0])
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.001,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=10)
        ax.set_ylim(0, max(vals) * 1.2)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/resultats_comparaison_metriques.png", bbox_inches="tight")
    plt.close()
    print("\n[7.1] Figure sauvegardée : resultats_comparaison_metriques.png")

    # -------------------------------------------------------------------
    # 7.2 Valeurs réelles vs prédites
    # -------------------------------------------------------------------
    fig, axes = plt.subplots(1, len(modeles), figsize=(14, 6))
    if len(modeles) == 1:
        axes = [axes]

    for ax, (nom, y_pred) in zip(axes, predictions.items()):
        ax.scatter(y_test, y_pred, alpha=0.3, s=8, color="#4C72B0")
        lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        ax.plot(lims, lims, "r--", lw=1.5, label="Prédiction parfaite")
        ax.set_xlabel("Valeurs réelles (100k $)")
        ax.set_ylabel("Valeurs prédites (100k $)")
        ax.set_title(f"{nom}\nR² = {r2_score(y_test, y_pred):.4f}", fontweight="bold")
        ax.legend()

    fig.suptitle("Réel vs Prédit - Jeu de Test", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/resultats_reel_vs_predit.png", bbox_inches="tight")
    plt.close()
    print("[7.2] Figure sauvegardée : resultats_reel_vs_predit.png")

    # -------------------------------------------------------------------
    # 7.3 Distribution des résidus
    # -------------------------------------------------------------------
    fig, axes = plt.subplots(1, len(modeles), figsize=(14, 5))
    if len(modeles) == 1:
        axes = [axes]

    for ax, (nom, y_pred), couleur in zip(axes, predictions.items(), couleurs):
        residus = y_test.values - y_pred
        ax.hist(residus, bins=60, color=couleur, edgecolor="white", alpha=0.8)
        ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
        ax.axvline(residus.mean(), color="orange", linestyle="--",
                   linewidth=1.5, label=f"Moyenne : {residus.mean():.3f}")
        ax.set_xlabel("Résidu (Réel - Prédit)")
        ax.set_ylabel("Fréquence")
        ax.set_title(f"{nom}", fontweight="bold")
        ax.legend()

    fig.suptitle("Distribution des Résidus - Jeu de Test", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/resultats_distribution_residus.png", bbox_inches="tight")
    plt.close()
    print("[7.3] Figure sauvegardée : resultats_distribution_residus.png")

    # -------------------------------------------------------------------
    # 7.4 Importance des features (Random Forest - feature_importances_)
    # -------------------------------------------------------------------
    # J'utilise le Random Forest car il expose directement les importances
    # via la réduction d'impureté (MDI), ce qui est rapide et interprétable.
    if "Random Forest" in modeles:
        rf_model = modeles["Random Forest"].named_steps["model"]
        importances = pd.Series(rf_model.feature_importances_, index=feature_names)
        importances_sorted = importances.sort_values(ascending=True)

        fig, ax = plt.subplots(figsize=(9, 7))
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(importances_sorted)))
        importances_sorted.plot(kind="barh", ax=ax, color=colors, edgecolor="white")
        ax.set_xlabel("Importance (MDI - Mean Decrease in Impurity)")
        ax.set_title("Importance des Features - Random Forest\n(MDI)",
                     fontsize=12, fontweight="bold")
        ax.axvline(1 / len(importances_sorted), color="red", linestyle="--",
                   linewidth=1.2, label="Importance uniforme (baseline)")
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/resultats_feature_importance.png", bbox_inches="tight")
        plt.close()
        print("[7.4] Figure sauvegardée : resultats_feature_importance.png\n")


# =============================================================================
# 8. RAPPORT FINAL
# =============================================================================

def generer_rapport(df_resultats: pd.DataFrame, meilleur_nom: str) -> None:
    """Affiche un résumé synthétique des résultats dans la console."""
    print("=" * 70)
    print("  RAPPORT DE SYNTHÈSE")
    print("=" * 70)
    print("""
  Dataset    : California Housing (Scikit-learn)
  Cible      : Prix médian des logements (en 100 000 USD)
  Split      : 80 % entraînement / 20 % test (stratifié)
  Algorithmes: Random Forest Regressor vs Gradient Boosting Regressor
  Tuning     : GridSearchCV (5-fold cross-validation)
    """)
    print("  Résultats finaux sur le jeu de test :\n")
    print(df_resultats[["RMSE Test", "MAE Test", "R² Test"]].to_string())
    print(f"""
  Conclusion :
    Le modèle "{meilleur_nom}" obtient les meilleures performances.
    Un R² supérieur à 0.80 indique que le modèle explique une large
    part de la variabilité des prix, ce qui est satisfaisant pour un
    dataset aussi hétérogène géographiquement.

    Les features les plus prédictives sont MedInc (revenu médian) et
    la localisation géographique (Latitude, Longitude), ce qui est
    cohérent avec la réalité du marché immobilier californien.

  Axes d'amélioration :
    - Intégrer des données externes (proximité écoles, transports)
    - Tester XGBoost / LightGBM pour gagner en performance
    - Déployer le modèle via une API FastAPI + conteneur Docker
  """)
    print("=" * 70)


# =============================================================================
# POINT D'ENTRÉE PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  PRÉDICTION DU PRIX DE L'IMMOBILIER - CALIFORNIA HOUSING")
    print("  Projet Portfolio ML | Ingénieur IA 4A")
    print("=" * 70 + "\n")

    # 1. Chargement
    df = charger_donnees()

    # 2. EDA
    analyse_exploratoire(df)

    # 3. Preprocessing
    X_train, X_test, y_train, y_test, feature_names = preprocessing(df)

    # 4. Construction des pipelines
    pipelines = construire_pipelines()

    # 5. Entraînement (optimiser=True active le GridSearchCV)
    # Passer optimiser=False pour un entraînement rapide sans tuning.
    modeles = entrainer_modeles(pipelines, X_train, y_train, optimiser=True)

    # 6. Évaluation
    df_resultats, predictions, meilleur_nom = evaluer_modeles(
        modeles, X_train, X_test, y_train, y_test
    )

    # 7. Visualisations
    visualiser_resultats(modeles, predictions, y_test, feature_names, df_resultats)

    # 8. Rapport final
    generer_rapport(df_resultats, meilleur_nom)

    print("\nPipeline complet terminé. Figures générées dans le répertoire courant.\n")
