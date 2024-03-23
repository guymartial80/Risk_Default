## 1. DESCRIPTION

Cette étude se propose de déterminer les risques de défaut des clients d'une banque Taïwanaise. Nous allons utiliser le modèle de classification basé sur les forêts aléatoires pour y arriver. De plus, nous rechercherons les meilleurs hyperparamètres qui optimiseront les performances de notre modèle.

## 2. ETAPES DE REALISATION

### 2.1 Importation des librairies
Les packages de base ont été importés dans un premier temps, notamment Pandas, numpy, Matplolib, Seaborn. Les bibliothèques utilisées sont listées dans le fichier "requirement.txt".

### 2.2 Structure globale du travail
Pour une meilleure lisibilité du code principal, le fichier "risk_utils.py" a été créé pour héberger toutes les fonctions nécessaires pour ce projet. Ces dernières sont y sont d'ailleurs décrites. Ainsi, lesdites fonctions sont appelées dans le fichier principal "Notebook_risk.ipynb" pour exécution.

### 2.3 Importation et structure du dataset
#### 2.3.1 Importation et aperçu
Le jeu de données est importé à travers la fonction implémentée à cet effet; l'importation est faite pour s'assurer de la réussite de l'opération.

#### 2.3.2 Structure du dataset
Cette partie nous permet d'en savoir un peu plus sur le contenu de la base de données, tout en sachant que la variable cible (default payment next month) est connue.

#### 2.3.3 Résumé statistique
Calculs statistiques de base sur toutes les variables de type numérique du jeu de données.

### 2.4 EDA
A cette étape, plusieurs démarches ont été sollicitées pour faciliter l'analyse :
- Les statistiques univariées: Recueillir le nombre d'occurrences par variables de type catégoriel;
- Les statistiques bivariées : Nombre d'occurrences de chaque variable catégorielle en fonction des modalités de la variable cible;
- Les statistiques bivariées pour les variables  de type numérique : cela s'est fait par la matrice des corrélations et les boites à moustache.

### 2.5 Preprocessing
- Séparation des variables : type catégoriel nominal, type catégoriel ordinal et type numérique ;
- Création d'un pipeline pour l'encodage des données ainsi catégorisées ;
- Instanciation des estimateurs RandomForestClassifier et LogisticRegression dans le pipeline.

### 2.6 Machine Learning
- Séparation des variables (explicatives et expliquée);
- Séparation en base d'entrainement, validation et test;
- Entrainement des modèles du pipeline;
- Modélisation, évaluation des modèles entrainés et recherche des meilleurs hyperparamètres avec GridSearchCV;
- Modélisation du meilleur modèle avec les meilleurs hyperparamètres dans la base de test

### 2.7 Résultats obtenus
- Le processus de modélisation a revélé, après détermination du meilleur modèle de Machine Learning et recherche des meilleurs hyperparamètres à adopter pour ledit modèle, que le RandomForest est meilleur que la Régression Logistique dans ce contexte et les hyperparamètres les mieux adaptés pour l'obtention d'un meilleur résultat sont:
  - {'model__criterion': 'entropy';
  - 'model__max_depth': 7;
  - 'model__n_estimators': 100})
La courbe ROC du meilleur modèle est ci-dessous illustrée :

![Results](https://github.com/guymartial80/Risk_Default/blob/main/best_output.png)

## 3. LIBRAIRIES UTILISEES
![Static Badge](https://img.shields.io/badge/Pandas-black?style=for-the-badge&logo=Pandas) ![Static Badge](https://img.shields.io/badge/Scikit-learn-black?style=for-the-badge&logo=Scikit-learn) ![Static Badge](https://img.shields.io/badge/Numpy-black?style=for-the-badge&logo=Numpy) ![Static Badge](https://img.shields.io/badge/Matplotlib-black?style=for-the-badge&logo=Matplotlib) ![Static Badge](https://img.shields.io/badge/Seaborn-black?style=for-the-badge&logo=Seaborn)





