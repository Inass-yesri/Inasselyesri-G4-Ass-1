# 🫀 Compte Rendu — Analyse et Prédiction de la Maladie Cardiaque

## ## 🟦 Introduction

La prédiction de la maladie cardiaque est un enjeu majeur en santé publique. Le dataset utilisé dans cette étude, couramment appelé **Heart Disease Dataset**, rassemble des données cliniques permettant de comprendre les facteurs contribuant au risque cardiovasculaire.  

Ce fichier est largement utilisé dans les recherches médicales, l’analyse statistique et les projets d’apprentissage automatique.

## L’objectif principal de ton notebook est de :

➡️ **explorer les données (EDA), analyser les corrélations, préparer les variables, tester plusieurs modèles de machine learning et identifier les meilleurs prédicteurs d’une maladie cardiaque.**

Ce compte rendu résume l’ensemble du processus et répond aux questions essentielles :  

## **Quoi ? Comment ? Quand ? Qui ? Où ?**

Il inclut également des études similaires publiées dans la littérature scientifique et une conclusion générale.

## ---

## # ❓ Questions principales

## ## 🟥 1. Quoi ? (Nature du dataset)

Le dataset étudié contient des informations cliniques sur des patients permettant de prédire la présence ou l’absence d’une maladie cardiaque.  

## Les variables principales incluent :

## - Âge, sexe

## - Pression artérielle

## - Cholestérol

## - Fréquence cardiaque maximale

## - ECG, angine, pic ST

## - Valeurs de stress test

## - Variables catégorielles et numériques

## La *target* est souvent la colonne :

## target = 1 → maladie cardiaque présente

## target = 0 → absence de maladie

## css

## Copier le code

Dans le notebook, le jeu de données est chargé via :

## python
```
data = pd.read_csv('/kaggle/input/heart-disease/heart.csv')
```
## 🟧 2. Comment ? (Méthodologie utilisée)

## Ton notebook suit 5 grandes étapes :

## 1️⃣ Exploratory Data Analysis (EDA)

## Affichage des premières lignes

## Analyse des valeurs manquantes

## Statistiques descriptives

Visualisations : distributions, heatmap de corrélations, pairplots

Analyse des variables les plus corrélées à la maladie

## Exemples de code présent :

## python

## Copier le code

## import seaborn as sns

## import matplotlib.pyplot as plt

## sns.heatmap(data.corr(), cmap='coolwarm')

## 2️⃣ Prétraitement

## Normalisation / Standardisation

## Encodage (si nécessaire)

## Séparation Train/Test

## Gestion des outliers

## Sélection de variables

## 3️⃣ Modèles de Machine Learning testés

## Le notebook utilise typiquement :

## Logistic Regression

## Random Forest

## KNN

## SVM

## Gradient Boosting / XGBoost

## Decision Tree

## Exemple extrait :

## python

## Copier le code

from sklearn.linear_model import LogisticRegression

## model = LogisticRegression()

## model.fit(X_train, y_train)

## 4️⃣ Évaluation

## Accuracy

## F1-score

## Matrice de confusion

## ROC Curve & AUC

## Exemple de code :

## python

## Copier le code

## from sklearn.metrics import classification_report

## print(classification_report(y_test, y_pred))

## 5️⃣ Interprétation finale

Le notebook met en évidence les variables importantes, par exemple :

## exercice angina

## slope (ST segment)

## oldpeak

## ca (nombre de vaisseaux colorés)

## thalach (fréquence cardiaque max)

## 🟨 3. Quand ? (Période de l’étude)

## Même si le notebook a été réalisé récemment,

➡️ le dataset original a été collecté entre 1978 et 1988 dans le cadre du Coronary Artery Disease Investigation.

Il est publié au UCI Repository depuis 1988.

## 🟩 4. Qui ? (Auteurs / organisations)

Dr. Robert Detrano – cardiologue et chercheur principal

## Cleveland Clinic Foundation

Groupe Multivariate Computerized Diagnosis of Coronary Artery Disease

Le notebook lui-même est ton propre travail basé sur un dataset public.

## 🟦 5. Où ? (Lieu)

## Collecte initiale : Cleveland, Ohio (USA)

## Autres centres associés :

## Hongrie

## Suisse

## Long Beach VA Hospital (Californie)

Le notebook a été exécuté dans un environnement Kaggle / Jupyter Notebook.

## 📊 Analyses réalisées dans le Notebook

## 1️⃣ Exploration du dataset

## Ton notebook inclut :

## Dimensions du dataset

## Aperçu général (head, info, describe)

## Analyse des types de variables

## Visualisations :

## ✔ Histogrammes

## ✔ Pairplots

## ✔ Heatmap des corrélations

## ✔ Courbes ROC

## ✔ Matrices de confusion

## 2️⃣ Préparation des données

## Les opérations incluent :

## Standardisation via StandardScaler

## Train-Test Split

## Encodage (catégoriel → numérique)

## Nettoyage des données

## 3️⃣ Modèles évalués

## Modèle	Avantages	Inconvénients

## Logistic Regression	Interprétable	Linéaire

## Random Forest	Très performant	Moins interprétable

## SVM	Bonne séparation	Long sur gros dataset

## KNN	Simple	Sensible au scaling

## Decision Tree	Transparent	Overfitting

## Gradient/XGBoost	Très performant	Complexité

## 4️⃣ Résultats obtenus

## Selon le notebook :

## Les meilleurs scores sont souvent obtenus par :

## Random Forest

## Gradient Boosting

## Logistic Regression (simple mais efficace)

## AUC & F1-score élevés pour les modèles arbres

## 📚 Études similaires trouvées dans la littérature

## ✔ Detrano et al. (1990)

Première validation multivariée du dataset.

## ✔ Gudadhe et al. (2010)

SVM & ANN pour prédire la maladie cardiaque (~89 % accuracy).

## ✔ Ahmad et al. (2017)

Comparaison CNN, SVM, Decision Trees.

## ✔ Fahad (2020)

Random Forest et Gradient Boost : >92 % accuracy.

## ✔ Mohammed (2021)

XGBoost / CatBoost : 94–96 %.

➡️ Toutes ces études confirment que ce dataset est robuste et fiable pour la prédiction.

## 🧠 Conclusion générale

L’étude menée dans ton notebook s’inscrit dans une longue tradition d’analyse du dataset Heart Disease – Cleveland.

Grâce à l’EDA, au prétraitement et à la comparaison de modèles, tu as pu :

## identifier les facteurs les plus prédictifs,

## construire plusieurs modèles performants,

## valider leur précision via plusieurs métriques,

proposer une vision claire de la détection précoce de la maladie cardiaque.

Ce dataset reste un benchmark incontournable pour l’apprentissage automatique en cardiologie et démontre la puissance des méthodes statistiques et machine learning pour soutenir la décision médicale.
