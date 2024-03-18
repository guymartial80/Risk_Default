import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
import time
seed = 42


def importer_affichage_dataset(chemin_fichier):
    """
    Cette fonction importe un fichier Excel ou CSV en indiquant seulement en paramètre le nom du fichier et son extension,
    à condition que ce dernier soit dans le même repertoire que le présent fichier.
    
    Args:
    - chemin_fichier : Nom du fichier et son extension ou Chemin vers le fichier à importer (Excel ou CSV).
    
    Returns:
    - df : DataFrame contenant les données du fichier.
    """
    # Vérifier l'extension du fichier pour déterminer le type de fichier
    if chemin_fichier.endswith('.xlsx'):
        # Importer un fichier Excel
        df = pd.read_excel(chemin_fichier, header = 1)
    elif chemin_fichier.endswith('.xls'):
        # Importer un fichier xls (ancienne version)
        df = pd.read_excel(chemin_fichier, header = 1)
    elif chemin_fichier.endswith('.csv'):
        # Importer un fichier CSV
        df = pd.read_csv(chemin_fichier)
    else:
        raise ValueError("Le fichier doit être au format Excel (.xlsx ou .xls) ou CSV (.csv)")
    
    return df



def transform_dataset(df):
    # Mettre tous les libellés de l'entête en minuscule et remplacer les espaces par des underscores
    df.columns = df.columns.str.lower().str.replace(' ', '_')

    # Liste des mois et des types de statut
    # months = ["sept", "aug", "july", "june", "may", "april"]
    
    # Dictionnaire des mois
    months_pay = {'sept': 0, 'aug': 2, 'july': 3, 'june': 4, 'may': 5, 'april': 6}
    months = {'sept': 1, 'aug': 2, 'july': 3, 'june': 4, 'may': 5, 'april': 6}
    status = ["payment_status", "bill_status", "previous_payment"]

    # Créer un dictionnaire de correspondance
    column_mapping = {}

    # Pour les colonnes commençant par 'pay_'
    for month, index in months_pay.items():
        column_mapping[f'pay_{index}'] = f'{status[0]}_{month}'

    # Pour les colonnes commençant par 'bill_amt'
    for month, index in months.items():
        column_mapping[f'bill_amt{index}'] = f'{status[1]}_{month}'

    # Pour les colonnes commençant par 'pay_amt'
    for month, index in months.items():
        column_mapping[f'pay_amt{index}'] = f'{status[2]}_{month}'

    # Renommer les colonnes selon le dictionnaire de correspondance
    df.rename(columns=column_mapping, inplace=True)

    # Créer un nouveau dictionnaire contenant uniquement les valeurs avec le préfixe "payment_status"
    payment_status_mapping = {key: value for key, value in column_mapping.items() if value.startswith('payment_status')}

    # Création des ditionnaires 
    gender_mapping = {1: 'Male', 
                      2: 'Female'}

    education_mapping = {0: 'Others',
                        1: 'Graduate school',
                        2: 'University',
                        3: 'High school',
                        4: 'Others',
                        5: 'Others',
                        6: 'Others'}

    marriage_mapping = {0: 'Others',
                        1: 'Married',
                        2: 'Single',
                        3: 'Others'}

    
    default_payment_mapping = {0: 'No',
                               1: 'Yes'}
    
    payment_status_description_mapping = {-2: 'No consumption',
                                          -1: 'Payed duly',
                                           0: 'Use of revolving credit',
                                           1: 'delayed 1 month',
                                           2: 'delayed 2 month',
                                           3: 'delayed 3 month',
                                           4: 'delayed 4 month',
                                           5: 'delayed 5 month',
                                           6: 'delayed 6 month',
                                           7: 'delayed 7 month',
                                           8: 'delayed 8 month',
                                           9: 'delayed >=9 month'}
    

    # Supprimer la colonne 'id'
    df.drop(columns=['id'], inplace=True)

    # Conversion des variables en type catégoriel
    df['sex'] = df['sex'].map(gender_mapping).astype('category')
    df['education'] = df['education'].map(education_mapping).astype('category')
    df['marriage'] = df['marriage'].map(marriage_mapping).astype('category')
    df['default_payment_next_month'] = df['default_payment_next_month'].map(default_payment_mapping).astype('category')

    for col in payment_status_mapping.values():
        df[col] = df[col].map(payment_status_description_mapping).astype('category')

    # Exportation du DataFrame en fichier CSV
    df.to_csv('default_of_credit_card_clients.csv', index=False)
    
    return df


def re_transform_dataset(df):
    # Recupérer toutes les colonnes relatives à l'état de remboursement mensuel
    payment_status_cols = ['payment_status_sept','payment_status_aug','payment_status_july','payment_status_june','payment_status_may','payment_status_april',]
    
    gender_mapping = {'Male':1, 
                      'Female':2}
    
    
    education_mapping = {'Others':0,
                         'Graduate school':1,
                         'University':2,
                         'High school':3,
                         'Others':4,
                         'Others':5,
                         'Others':6}
    
    
    marriage_mapping = {'Others':0,
                        'Married':1,
                        'Single':2,
                        'Others':3}
    
    payment_status_description_mapping = {'No consumption':-2,
                                          'Payed duly':-1,
                                          'Use of revolving credit':0,
                                          'delayed 1 month':1,
                                          'delayed 2 month':2,
                                          'delayed 3 month':3,
                                          'delayed 4 month':4,
                                          'delayed 5 month':5,
                                          'delayed 6 month':6,
                                          'delayed 7 month':7,
                                          'delayed 8 month':8,
                                          'delayed >=9 month':9}
    

    # Conversion des variables en type catégoriel
    for col in payment_status_cols:
        df[col] = df[col].map(payment_status_description_mapping).astype('int64')
        
    
    df['sex'] = df['sex'].map(gender_mapping).astype('object')
    df['education'] = df['education'].map(education_mapping).astype('object')
    df['marriage'] = df['marriage'].map(marriage_mapping).astype('object')

    # Exportation du DataFrame en fichier CSV
    df.to_csv('default_of_credit_card_clients_for_model.csv', index=False)
    
    return df


def plot_categorical_column(df, column):
    """
    Fonction interne pour tracer le graphique de la variable catégorielle.
    """
    plt.figure(figsize=(5, 3))
    # Créer un countplot avec la variable catégorielle en x et Attrition en hue
    ax = sns.countplot(y=column, hue=column, data=df)

    # Orienter les libellés de l'axe des abscisses de 45°
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    plt.xlabel('Effectif des clients')
    plt.ylabel(column)
    plt.title(f'Comptage des valeurs de {column}')
    plt.show()



# analyses univariés
def univariate_statistics(df):
    """
    Fonction pour afficher les comptages, pourcentages et graphiques pour chaque variable catégorielle d'un DataFrame.
    
    Args:
    - df: DataFrame pandas à analyser.
    """

    for column in df.columns:
        if df[column].dtype == object:
            # Compter les occurrences de chaque catégorie
            value_counts = df[column].value_counts()
            
            # Calculer les pourcentages
            percentages = (df[column].value_counts(normalize=True) * 100).round(2)
            
            # Créer un DataFrame à partir des comptages et pourcentages
            result = pd.concat([value_counts, percentages], axis=1)
            result.columns = ['Comptage', 'Pourcentage (%)']
            
            # Afficher les résultats
            print(f"Tableau de comptage des valeurs avec pourcentages pour la variable '{column}':\n")
            print(result)
            print("\n")
            
            # Tracer le graphique de la variable catégorielle
            plot_categorical_column(df, column)


def count_variable_types(df):
    # Compter les variables catégorielles et numériques
    categorical_variables = df.select_dtypes(include=['category', 'object']).columns
    numeric_variables = df.select_dtypes(include=['number', 'int64', 'float']).columns
    
    # Créer un DataFrame pour afficher les résultats
    result_df = pd.DataFrame({
        'Variable Type': ['Categorical', 'Numeric'],
        'Count': [len(categorical_variables), len(numeric_variables)]
    })
    
    return result_df


def bivariate_statistics(df):
    """
    Fonction pour afficher des countplots bivariés pour chaque colonne catégorielle en fonction de 'default_payment_next_month'.
    
    Args:
    - df: DataFrame pandas contenant les données.
    """
    # Récupérer les noms des colonnes catégorielles
    categorical_columns = df.select_dtypes(include=['object']).columns
    # Exclure la variable 'default_payment_next_month' de la liste des colonnes catégorielles
    categorical_columns_except_default_pay = categorical_columns.drop('default_payment_next_month') 
    
    # Pour chaque colonne catégorielle
    for column_name in categorical_columns_except_default_pay:
        # ============================
        # Compter les occurrences de chaque catégorie pour default_payment_next_month = "Yes"
        value_counts_yes = df[df['default_payment_next_month'] == 'Yes'][column_name].value_counts()
        # Calculer les pourcentages pour default_payment_next_month = "Yes"
        percentages_yes = (df[df['default_payment_next_month'] == 'Yes'][column_name].value_counts(normalize=True) * 100).round(2)
        
        # Compter les occurrences de chaque catégorie pour default_payment_next_month = "No"
        value_counts_no = df[df['default_payment_next_month'] == 'No'][column_name].value_counts()
        # Calculer les pourcentages pour Attrition = "No"
        percentages_no = (df[df['default_payment_next_month'] == 'No'][column_name].value_counts(normalize=True) * 100).round(2)
        
        # Créer un DataFrame à partir des comptages et pourcentages pour default_payment_next_month = "Yes"
        result_yes = pd.concat([value_counts_yes, percentages_yes], axis=1)
        result_yes.columns = ['Default_Yes', 'Ratio_Yes']
        
        # Créer un DataFrame à partir des comptages et pourcentages pour default_payment_next_month = "No"
        result_no = pd.concat([value_counts_no, percentages_no], axis=1)
        result_no.columns = ['Default_No', 'Ration_No']
        
        # Fusionner les deux DataFrames sur les index (valeurs de catégories)
        result = pd.concat([result_yes, result_no], axis=1, sort=False)
        
        # Afficher les résultats
        print(f"Tableau de comptage des valeurs avec pourcentages pour la variable '{column_name}':\n")
        print(result)
        print("\n")
        
        # Tracer le graphique de la variable catégorielle en fonction des defauts de paiement
        plt.figure(figsize=(6, 4))
        # Créer un countplot avec la variable catégorielle en y et default_payment_next_month en hue
        ax = sns.countplot(y=column_name, hue="default_payment_next_month", data=df)

        # Orienter les libellés de l'axe des abscisses de 45°
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

        # Afficher le titre et les étiquettes des axes
        plt.title('Histogramme des modalités de {}'.format(column_name))
        plt.xlabel('Effectif des clients')
        plt.ylabel(column_name)

        # Afficher la légende
        plt.legend(title='Defaut paiement', loc='lower right')

        # Afficher le plot
        plt.show()



def categorical_variables(df):
    """
    Fonction pour sélectionner les variables de type catégoriel dans un DataFrame.
    
    Args:
    - df: DataFrame pandas contenant les variables à afficher.

    Returns:
    - Liste des libellés des variables de type catégoriel.
    """
    # Sélectionner les colonnes de type 'object'
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    return categorical_cols



def numerical_variables(df):
    """
    Fonction pour sélectionner les variables de type catégoriel dans un DataFrame.
    
    Args:
    - df: DataFrame pandas contenant les variables à afficher.

    Returns:
    - Liste des libellés des variables de type numérique.
    """
    # Sélectionner les colonnes de type 'object'
    numerical_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
    return numerical_cols



def select_numeric_columns_corr(df):
    """
    Fonction pour sélectionner les variables de type numérique dans un DataFrame.
    
    Args:
    - df: DataFrame pandas contenant les variables à afficher.
    """    
    numeric_columns = df.select_dtypes(include=['number']).columns
    return df[numeric_columns]



def plot_correlation_matrix(df):
    """
    Fonction pour afficher la matrice des corrélations, dans le but de déceler les liens entre les variables.
    
    Args:
    - df: DataFrame pandas contenant les variables à afficher.
    """
    plt.figure(figsize=(20,20))
    mask = np.triu(np.ones_like(df.corr(), dtype=bool))
    sns.heatmap(df.corr(), mask=mask, center=0, cmap='RdBu', annot=True, fmt=".2f", vmin=-1, vmax=1)
    plt.title('Matrice des corrélations', fontsize = 18, fontweight = 'bold')
    plt.show()



def boxplot_numeric_variables(df):
    """
    Fonction pour afficher les boxplots des variables de type int64 dans un DataFrame.
    
    Args:
    - df: DataFrame pandas contenant les variables à afficher.
    """
    # Récupérer les variables avec le dtype 'int64'
    int64_columns = df.select_dtypes(include=['int64']).columns
    
    # Diviser les variables en blocs de 3 sur une ligne
    num_plots = len(int64_columns)
    num_rows = (num_plots + 2) // 3
    num_cols = min(3, num_plots)
    
    # Créer une nouvelle figure
    plt.figure(figsize=(15, 5*num_rows))
    
    # Afficher les boxplots pour chaque variable
    for i, column in enumerate(int64_columns):
        plt.subplot(num_rows, num_cols, i + 1)
        sns.boxplot(data=df, x="default_payment_next_month", y=df[column], hue='default_payment_next_month')
        plt.title(column)
    
    # Ajuster l'espacement entre les sous-graphiques
    plt.tight_layout()
    
    # Afficher les graphiques
    plt.show()



def remove_outliers(df):
    """
    Fonction pour éliminer les outliers d'un DataFrame en les remplaçant par les limites inférieures (lower)
    et supérieures (upper) définies par la méthode de la zone interquartile (IQR).
    
    Args:
    - df: DataFrame pandas contenant les données.
    
    Returns:
    - DataFrame pandas avec les outliers remplacés par les limites inférieures et supérieures.
    """
    int64_columns = df.select_dtypes(include=['int64']).columns  # Sélectionner les colonnes de type int64
    
    # Pour chaque variable numérique
    for var in int64_columns:
        IQR = df[var].quantile(0.75) - df[var].quantile(0.25)
        lower = df[var].quantile(0.25) - (1.5 * IQR)
        upper = df[var].quantile(0.75) + (1.5 * IQR)
        
        # Remplacer les valeurs atypiques par les limites inférieures et supérieures
        df[var] = df[var].apply(lambda x: min(upper, max(x, lower)))
    
    boxplot_numeric_variables(df)



def metrics_best_model(pipeline, X_test, y_test):
    # Prédictions sur l'ensemble de validation
    y_pred = pipeline.predict(X_test)
    
    # Convertir les valeurs de y_val en valeurs numériques
    y_test_numeric = y_test.replace({'Yes': 1, 'No': 0})
    
    # Convertir les valeurs de y_pred en valeurs numériques
    y_pred_numeric = pd.Series(y_pred).replace({'Yes': 1, 'No': 0})
    
    # Calcul du rapport de classification
    cr = classification_report(y_test_numeric, y_pred_numeric)
    print("Classification Report:")
    print(cr)

    # Calcul de la matrice de confusion
    cm = confusion_matrix(y_test_numeric, y_pred_numeric)

    # Affichage du graphique de la matrice de confusion et de la courbe ROC AUC
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Matrice de confusion
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[0])
    axes[0].set_xlabel('Predicted labels')
    axes[0].set_ylabel('True labels')
    axes[0].set_title('Confusion Matrix')

    # Courbe ROC AUC
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test_numeric, y_pred_proba)
    auc = roc_auc_score(y_test_numeric, y_pred_proba)
    axes[1].plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc)
    axes[1].plot([0, 1], [0, 1], 'k--')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('Receiver Operating Characteristic (ROC)')
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    plt.show()


def evaluate_and_find_best_model(pipelines, X_val, y_val, X_train, y_train, X_test, y_test):
    best_model = None
    best_score = 0
    
    for model_name, pipeline in pipelines.items():
        print(f"Evaluation du modèle {model_name}:")
        # Prédictions sur l'ensemble de validation
        y_pred = pipeline.predict(X_val)

        # Convertir les valeurs de y_val en valeurs numériques
        y_val_numeric = y_val.replace({'Yes': 1, 'No': 0})

        # Convertir les valeurs de y_pred en valeurs numériques
        y_pred_numeric = pd.Series(y_pred).replace({'Yes': 1, 'No': 0})

        # Calcul du rapport de classification
        cr = classification_report(y_val_numeric, y_pred_numeric)
        print("Classification Report:")
        print(cr)

        # Calcul de la matrice de confusion
        cm = confusion_matrix(y_val_numeric, y_pred_numeric)

        # Affichage du graphique de la matrice de confusion et de la courbe ROC sur la même ligne
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Matrice de confusion
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[0])
        axes[0].set_xlabel('Predicted labels')
        axes[0].set_ylabel('True labels')
        axes[0].set_title('Confusion Matrix')

        # Courbe ROC AUC
        y_pred_proba = pipeline.predict_proba(X_val)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_val_numeric, y_pred_proba)
        auc = roc_auc_score(y_val_numeric, y_pred_proba)
        axes[1].plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc)
        axes[1].plot([0, 1], [0, 1], 'k--')
        axes[1].set_xlim([0.0, 1.0])
        axes[1].set_ylim([0.0, 1.05])
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title('Receiver Operating Characteristic (ROC)')
        axes[1].legend(loc="lower right")

        plt.tight_layout()
        plt.show()
        
        # Comparer avec le meilleur score actuel
        if auc > best_score:
            best_model = pipeline
            best_score = auc
    
    print("Meilleur modèle sélectionné:")
    print(best_model)

    # Identifier le meilleur modèle en fonction des performances sur les données de validation
    best_model_name = max(pipelines.keys(), key=lambda k: roc_auc_score(y_val_numeric, pipeline.predict_proba(X_val)[:, 1]))

    # Définir la grille de recherche des hyperparamètres pour le meilleur modèle
    if best_model_name == 'RandomForestClassifier':
        # Définir la grille de recherche des hyperparamètres pour le RandomForestClassifier
        param_grid = {
                "model__criterion": ["entropy", "gini"],
                "model__max_depth": range(7, 11),
                "model__n_estimators": [100, 150, 200]
        }  
        best_model_pipeline = pipelines[best_model_name]

    elif best_model_name == 'LogisticRegression':
        # Définir la grille de recherche des hyperparamètres pour la LogisticRegression
        param_grid = [{
            'penalty':['l1','l2'],
            'C':[0.001,0.01,0.05,0.1,0.5,1.0,10.0,100.0]
        }]

        best_model_pipeline = pipelines[best_model_name]

    # Exécuter la GridSearchCV sur le meilleur modèle avec la grille de recherche des hyperparamètres
    grid_search = GridSearchCV(best_model_pipeline, param_grid, cv=5, scoring='roc_auc')
    grid_search.fit(X_train, y_train)

    # Identifier les meilleurs hyperparamètres
    best_params = grid_search.best_params_

    # Entraîner le modèle avec les meilleurs hyperparamètres sur l'ensemble de données complet
    best_model_pipeline.set_params(**best_params)
    best_model_pipeline.fit(X_test, y_test)
    
    # Afficher le rapport de classification, la matrice de confusion et la courbe ROC pour le meilleur modèle
    metrics_best_model(best_model_pipeline, X_test, y_test)
    
    return best_model_pipeline, best_params


