#!/usr/bin/env python
# coding: utf-8
# Tratamiento de datos
# ==============================================================================
import pandas as pd
import numpy as np
from collections import Counter

# Gráficos
# ==============================================================================
import seaborn as sns
from matplotlib import pyplot as plt


# Procesamiento de datos
# ==============================================================================
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

# Modelos de clasificación
# ==============================================================================
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, 
    ExtraTreesClassifier
)
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Validación y partición
# ==============================================================================
from sklearn.model_selection import (
    train_test_split, cross_validate, KFold
)

# Métricas de evaluación
# ==============================================================================
from sklearn import metrics
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, fbeta_score, f1_score, 
    precision_score, recall_score, confusion_matrix, roc_curve, auc, 
    roc_auc_score, precision_recall_curve, ConfusionMatrixDisplay, 
    classification_report
)
from sklearn.model_selection import RandomizedSearchCV

# Técnicas de balanceo de datos
# ==============================================================================
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN





# Funciones necesarias para este proyecto:
# ==============================================================================

def duplicate_columns(frame):
    '''
    ----------------------------------------------------------------------------------------------------------
    Función duplicate_columns:
    ----------------------------------------------------------------------------------------------------------
    - Descripción:
        Esta función recibe un DataFrame y busca columnas duplicadas basándose en su contenido. Si dos 
        columnas tienen exactamente los mismos valores, se considera que una de ellas es duplicada.
        
    - Inputs: 
        - frame (DataFrame): DataFrame que contiene las columnas a evaluar.
        
    - Return:
        - dups (list): Lista con los nombres de las columnas duplicadas.
    '''
    
    # Agrupamos las columnas por su tipo de datos
    groups = frame.columns.to_series().groupby(frame.dtypes).groups
    dups = []
    
    # Recorremos cada grupo de columnas con el mismo tipo de dato
    for t, v in groups.items():

        cs = frame[v].columns
        vs = frame[v]
        lcs = len(cs)
        
        # Comparamos cada columna con las demás dentro del grupo
        for i in range(lcs):
            ia = vs.iloc[:,i].values
            for j in range(i+1, lcs):
                ja = vs.iloc[:,j].values
                if np.array_equal(ia, ja):
                    dups.append(cs[i])
                    break
    return dups



def dame_variables_categoricas(dataset=None):
    '''
    ----------------------------------------------------------------------------------------------------------
    Función dame_variables_categoricas:
    ----------------------------------------------------------------------------------------------------------
    - Descripción:
        Esta función recibe un DataFrame y devuelve una lista de las variables 
        categóricas (con pocos valores únicos).
        
    - Inputs: 
        - dataset (DataFrame): DataFrame que contiene los datos de entrada.
        
    - Return:
        - lista_variables_categoricas (list): Lista con los nombres de las variables 
          categóricas en el DataFrame.
        - other (list): Lista con los nombres de las variables que no cumplen los criterios 
          para ser categóricas.
        - 1 (int): Indica que la ejecución es incorrecta debido a la falta del 
          argumento 'dataset'.
    '''
    # Verificar que el DataFrame de entrada no sea nulo
    if dataset is None:
        print(u'\nError: Falta el argumento dataset en la función')
        return 1 
    
    lista_variables_categoricas = []  
    other = []  

    # Recorrer las columnas del DataFrame
    for i in dataset.columns:
        
        # Si la columna es de tipo objeto (posiblemente categórica)
        if dataset[i].dtype == object:
            unicos = int(len(np.unique(dataset[i].dropna(axis=0, how='all'))))
            if unicos < 100:
                lista_variables_categoricas.append(i)  
            else:
                other.append(i)  
                
        # Si la columna es de tipo entero                
        if dataset[i].dtype == int:
            unicos = int(len(np.unique(dataset[i].dropna(axis=0, how='all'))))
            if unicos < 20:
                lista_variables_categoricas.append(i)  
            else:
                other.append(i) 

    return lista_variables_categoricas, other



def plot_feature(df, col_name, isContinuous):
    '''
    ----------------------------------------------------------------------------------------------------------
    Función plot_feature:
    ----------------------------------------------------------------------------------------------------------
    - Descripción:
        Esta función visualiza una variable, mostrando su distribución general y su
        relación con el estado la variable objetivo (TARGET). Para variables continuas,
        se usa un histograma y un boxplot; para variables categóricas, se usa un gráfico 
        de barras y uno de barras apiladas.
        
    - Inputs: 
        - df (DataFrame): DataFrame que contiene los datos de entrada.
        - col_name (str): Nombre de la variable a visualizar.
        - isContinuous (bool): Indica si la variable es continua (True) 
          o categórica (False).
        
    - Return:
         - None: Muestra los gráficos sin devolver valores.
    '''
    f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12,3), dpi=90)
    count_null = df[col_name].isnull().sum()
    
    # Gráfico sin considerar la variable objetivo
    if isContinuous:
        sns.distplot(df.loc[df[col_name].notnull(), col_name], kde=False, color='#5975A4', ax=ax1)
    else:
        order = df[col_name].dropna().value_counts().index
        sns.countplot(df[col_name].dropna(), order=order, color='#5975A4', saturation=1, ax=ax1)
    ax1.set_xlabel(col_name)
    ax1.set_ylabel('Count')
    ax1.set_title(col_name)
    ax1.set_title(col_name+ ' Numero de nulos: '+str(count_null))
    plt.xticks(rotation = 90)

    # Gráfico considerando la variable objetivo
    if isContinuous:
        sns.boxplot(x=col_name, y='TARGET', data=df, ax=ax2, palette='Set2') 
        ax2.set_ylabel('')
        ax2.set_title(col_name + ' by Target')
    else:
        data = df.groupby(col_name)["TARGET"].value_counts(normalize=True).to_frame('proportion').reset_index() 
        data.columns = [col_name, "TARGET", 'proportion']
        #sns.barplot(x = col_name, y = 'proportion', hue= target, data = data, saturation=1, ax=ax2)
        sns.barplot(x = col_name, y = 'proportion', hue= "TARGET", order=order, data = data, saturation=1, ax=ax2, palette='Set2')
        ax2.set_ylabel("TARGET"+' fraction')
        ax2.set_title("TARGET")
        plt.xticks(rotation = 90)
    ax2.set_xlabel(col_name)
    
    plt.tight_layout()
    plt.show()



def get_deviation_of_mean_perc(pd_loan, list_var_continuous, target, multiplier):
    '''
    ----------------------------------------------------------------------------------------------------------
    Función get_deviation_of_mean_perc:
    ----------------------------------------------------------------------------------------------------------
    - Descripción:
        Esta función calcula el porcentaje de valores que se encuentran fuera de un 
        intervalo de confianza, determinado por la media y una desviación estándar 
        multiplicada por un factor (multiplier), para cada variable continua en el 
        DataFrame. Luego, analiza la relación entre estos valores atípicos y la variable 
        objetivo (TARGET), y devuelve un resumen con los porcentajes de valores atípicos 
        y su distribución en relación con la variable objetivo.
        
    - Inputs: 
        - pd_loan (DataFrame): DataFrame que contiene los datos de entrada.
        - list_var_continuous (list): Lista con los nombres de las variables continuas 
          a analizar.
        - target (str): Nombre de la variable objetivo en el DataFrame.
        - multiplier (float): Factor multiplicador para calcular el intervalo de confianza
          (desviación estándar).
        
    - Return:
        - pd_final (DataFrame): DataFrame que contiene el porcentaje de valores atípicos 
          por cada variable continua, su distribución con respecto a la variable objetivo 
          (TARGET), y otros detalles relevantes.
    '''
    
    pd_final = pd.DataFrame()
    
    for i in list_var_continuous:
        
        series_mean = pd_loan[i].mean()
        series_std = pd_loan[i].std()
        std_amp = multiplier * series_std
        left = series_mean - std_amp
        right = series_mean + std_amp
        size_s = pd_loan[i].size
        
        perc_goods = pd_loan[i][(pd_loan[i] >= left) & (pd_loan[i] <= right)].size/size_s
        perc_excess = pd_loan[i][(pd_loan[i] < left) | (pd_loan[i] > right)].size/size_s
        
        if perc_excess>0:    
            pd_concat_percent = pd.DataFrame(pd_loan[target][(pd_loan[i] < left) | (pd_loan[i] > right)]\
                                            .value_counts(normalize=True).reset_index()).T
            pd_concat_percent.columns = [pd_concat_percent.iloc[0,0], 
                                         pd_concat_percent.iloc[0,1]]

            pd_concat_percent['variable'] = i
            pd_concat_percent['sum_outlier_values'] = pd_loan[i][(pd_loan[i] < left) | (pd_loan[i] > right)].size
            pd_concat_percent['porcentaje_sum_null_values'] = perc_excess
            pd_final = pd.concat([pd_final, pd_concat_percent], axis=0).reset_index(drop=True)
            
    if pd_final.empty:
        print('No existen variables con valores nulos')
        
    return pd_final



def get_corr_matrix(dataset = None, metodo='pearson', size_figure=[10,8]):
    ''''
    ----------------------------------------------------------------------------------------------------------
    Función get_corr_matrix:
    ----------------------------------------------------------------------------------------------------------
    - Descripción:
        Esta función calcula y visualiza la matriz de correlación entre las variables 
        numéricas de un conjunto de datos. 

    - Inputs: 
        - dataset (DataFrame): Conjunto de datos con las variables numéricas a analizar.
        - metodo (str): Método de correlación a utilizar.
        - size_figure (list): Tamaño de la figura del gráfico.

    - Return:
        - None: Muestra un mapa de calor de la matriz de correlación.
    ----------------------------------------------------------------------------------------------------------
    '''
    
    # Comprobación de que se ha proporcionado el dataset
    if dataset is None:
        print(u'\nHace falta pasar argumentos a la función')
        return 1
    sns.set(style="white")
    
    # Calcular la matriz de correlación
    corr = dataset.corr(method=metodo) 
    
    # Establecer la autocorrelación a cero para evitar distracciones
    for i in range(corr.shape[0]):
        corr.iloc[i, i] = 0
    
    f, ax = plt.subplots(figsize=size_figure)
    
    # Dibujar el mapa de calor con la correlación
    sns.heatmap(corr, center=0,
                square=True, linewidths=.5,  cmap ='viridis' ) #cbar_kws={"shrink": .5}
    plt.show()
    
    return 0



def get_percent_null_values_target(pd_loan, list_var_continuous, target):
    '''
    ----------------------------------------------------------------------------------------------------------
    Función get_percent_null_values_target:
    ----------------------------------------------------------------------------------------------------------
    - Descripción:
        Esta función analiza la relación entre los valores nulos de variables continuas 
        y la variable objetivo. Identifica si los valores faltantes de cada variable se 
        distribuyen de forma uniforme respecto a las clases de la variable objetivo o si 
        están asociados de manera significativa a alguna de ellas.
        
    - Inputs: 
        - pd_loan (DataFrame): DataFrame que contiene los datos de entrada.
        - list_var_continuous (list): Lista de nombres de variables continuas a analizar.
        - target (str): Nombre de la variable objetivo.

    - Output:
        - pd_final (DataFrame): DataFrame que resumen del analisis de la relación entre 
          los valores nulos de variables continuas y la variable objetivo.
    ----------------------------------------------------------------------------------------------------------
    '''
    
    # DataFrame final donde se acumularán los resultados
    pd_final = pd.DataFrame()
    
    # Iterar sobre cada variable continua de la lista
    for i in list_var_continuous:
        if pd_loan[i].isnull().sum()>0:
            target_distribution = pd_loan[target][pd_loan[i].isnull()].value_counts(normalize=True)
            
            target_dict = target_distribution.to_dict()
            percent_0 = target_dict.get(0, 0)  
            percent_1 = target_dict.get(1, 0)  

            # Crear un DataFrame temporal con la estructura deseada
            temp_df = pd.DataFrame({
                '0': [percent_0],
                '1': [percent_1],
                'variable': [i],
                'sum_null_values': [pd_loan[i].isnull().sum()],
                'porcentaje_sum_null_values': [pd_loan[i].isnull().sum() / pd_loan.shape[0]]
            })
            pd_final = pd.concat([pd_final, temp_df], axis=0).reset_index(drop=True)
    
    # Si no se encuentran variables con valores nulos, mostrar un mensaje
    if pd_final.empty:
        print('No existen variables con valores nulos')
        
    return pd_final



def cramers_v(confusion_matrix):
    ''' 
    ----------------------------------------------------------------------------------------------------------
    Función cramers_v:
    ----------------------------------------------------------------------------------------------------------
    - Descripción:
        Esta función calcula el estadístico V de Cramér para medir la asociación entre dos 
        variables categóricas. Utiliza la corrección de Bergsma y Wicher (2013) para ajustar
        el valor del chi-cuadrado y calcular una medida que indique la fuerza de la relación 
        entre las variables. El valor de Cramér's V oscila entre 0 (sin asociación) y 1 
        (asociación perfecta).
        
    - Inputs: 
        - confusion_matrix (DataFrame): Tabla de contingencia que contiene las frecuencias 
        absolutas de las categorías de las dos variables a analizar.
        
    - Output:
        - float: Valor de la V de Cramér 
    ----------------------------------------------------------------------------------------------------------    
    '''
    
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))


def calculate_woe(df, target, feature):
    '''
    Función para calcular el WOE de una variable categórica en relación a un target binario.
    
    Parámetros:
    - df (DataFrame): DataFrame de pandas que contiene los datos.
    - target (str): Nombre de la variable objetivo.
    - feature (str): Nombre de la columna de la variable categórica.
    
    Retorna:
    - WOE (DataFrame): DataFrame con las categorías de la variable y su WOE correspondiente.
    '''
    
    # Crear tabla de contingencia entre feature y target
    cross_tab = pd.crosstab(df[feature], df[target])
    
    # Calcular el número total de eventos y no-eventos
    total_events = cross_tab.sum(axis=0)[1]  # Suma de eventos (1)
    total_non_events = cross_tab.sum(axis=0)[0]  # Suma de no-eventos (0)
    
    # Calcular las proporciones de eventos (1) y no-eventos (0) por categoría
    cross_tab['p_event'] = cross_tab[1] / total_events
    cross_tab['p_non_event'] = cross_tab[0] / total_non_events
    
    # Calcular WOE para cada categoría
    cross_tab['WOE'] = np.log(cross_tab['p_non_event'] / cross_tab['p_event'])
    
    # Filtrar solo las categorías y su WOE
    woe_values = cross_tab[['WOE']]
    
    return woe_values


def calculate_iv(df, target, feature):
    '''
    Función para calcular el Information Value (IV) de una variable categórica en relación al target binario.
    
    Parámetros:
    - df (DataFrame): DataFrame de pandas que contiene los datos.
    - target (str): Nombre de la columna objetivo binaria.
    - feature (str): Nombre de la columna de la variable categórica.
    
    Retorna:
    - IV (float): El valor del Information Value.
    '''
    
    # Crear tabla de contingencia entre feature y target
    cross_tab = pd.crosstab(df[feature], df[target])
    
    # Calcular el número total de eventos y no-eventos
    total_events = cross_tab.sum(axis=0)[1]  # Suma de eventos (1)
    total_non_events = cross_tab.sum(axis=0)[0]  # Suma de no-eventos (0)
    
    # Calcular las proporciones de eventos (1) y no-eventos (0) por categoría
    cross_tab['p_event'] = cross_tab[1] / total_events
    cross_tab['p_non_event'] = cross_tab[0] / total_non_events
    
    # Calcular WOE para cada categoría
    cross_tab['WOE'] = np.log(cross_tab['p_non_event'] / cross_tab['p_event'])
    
    # Calcular IV sumando el producto de la diferencia de proporciones y WOE
    cross_tab['IV'] = (cross_tab['p_non_event'] - cross_tab['p_event']) * cross_tab['WOE']
    
    # Calcular el IV total
    iv_value = cross_tab['IV'].sum()
    
    return iv_value



def separar_por_unicos(df, list_columns_cat):
    '''
    ----------------------------------------------------------------------------------------------------------
    Función separar_por_unicos:
    ----------------------------------------------------------------------------------------------------------
    - Descripción: 
        Función que recibe un DataFrame y una lista de columnas categóricas y separa las 
        columnas en dos listas según el número de valores únicos que tienen. 
        
    - Inputs:
        - df (DataFrame): Pandas DataFrame que contiene los datos.
        - list_columns_cat (list): Lista con los nombres de las columnas categóricas del dataset.
        
    - Return:
        - list_columns_more_three_cat: Lista con los nombres de las columnas 
          categóricas que tienen más de 3 valores únicos.
        - list_columns_less_three_cat: Lista con los nombres de las columnas 
          categóricas que tienen 3 o menos valores únicos.
    ----------------------------------------------------------------------------------------------------------
    '''
    
    list_columns_more_three_cat = []  
    list_columns_less_three_cat = []  
    
    for col in list_columns_cat:
        num_unicos = df[col].nunique()  # Cuenta el número de valores únicos en la columna
        
        if num_unicos > 3:
            list_columns_more_three_cat.append(col)  
        else:
            list_columns_less_three_cat.append(col)  
    
    return list_columns_more_three_cat, list_columns_less_three_cat



def plot_cmatrix(y_true, y_pred, title='Confusion Matrix', figsize=(20,6)):
    ''' 
    ----------------------------------------------------------------------------------------------------------
    Función plot_cmatrix:
    ----------------------------------------------------------------------------------------------------------
    - Descripción:
        Esta función grafica la matriz de confusión en base a datos reales y predicciones de la variable objetivo
        en valores absolutos y normalizada.
    - Imputs:
        - y_true (list): Lista con los valores reales de la variable objetivo.
        - y_pred (list): Lista con las probabilidades predecidas por el modelo.
        - title (str): Título del gráfico.
        - figsize: Tamaño deseado para los gráficos.
        
    - Return:
        - Matriz de confusión en base a datos reales y predicciones de la variable objetivo
          en valores absolutos y normalizada.
    ''' 
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=figsize)
    ConfusionMatrixDisplay(confusion_matrix).from_predictions(y_true,y_pred, cmap='Blues', values_format=',.0f', ax=ax1)
    ConfusionMatrixDisplay(confusion_matrix).from_predictions(y_true,y_pred, cmap='Blues', normalize='true', values_format='.2%', ax=ax2)
    ax1.set_title(f'{title}', fontdict={'fontsize':18})
    ax2.set_title(f'{title} - Normalized', fontdict={'fontsize':18})
    ax1.set_xlabel('Predicted Label',fontdict={'fontsize':15})
    ax2.set_xlabel('Predicted Label',fontdict={'fontsize':15})
    ax1.set_ylabel('True Label',fontdict={'fontsize':15})
    ax2.set_ylabel('True Label',fontdict={'fontsize':15})

    plt.show()



def evaluate_model(y_true, y_pred):
    '''
    ----------------------------------------------------------------------------------------------------------
    Función metrics_summ:
    ----------------------------------------------------------------------------------------------------------
    - Descripción:
        Esta función recibe dos lista y calcula una serie de métricas (Accuracy, Balanced Accuracy, F2 Score, 
        F1 Score, Precision, Recall y la Matriz de Confusión) y las muestra en pantalla. 

    - Inputs:
        - y_true (list): Lista con los valores reales de la variable target. 
        - y_pred (list): Lista con los valores predecidos por el modelo.
        
    - Return:
        - Pinta en pantalla las métricas calculadas. 
    ''' 
    print(f'''
    Accuracy: {accuracy_score(y_true,y_pred):.5f}
    Balanced Accuracy: {balanced_accuracy_score(y_true,y_pred):.5f}
    F2 score: {fbeta_score(y_true,y_pred, beta=2):.5f}
    F1 score: {f1_score(y_true,y_pred):.5f}
    Precision: {precision_score(y_true,y_pred):.5f}
    Recall: {recall_score(y_true,y_pred):.5f}
    ''')
    
    plot_cmatrix(y_true, y_pred)


def plot_roc_gini(y_true=None, y_pred=None, size_figure=[6, 6], title='Curva ROC'):
    ''' 
    ----------------------------------------------------------------------------------------------------------
    Función plot_roc_gini:
    ----------------------------------------------------------------------------------------------------------
    - Descripción:
        Esta función grafica la curva ROC en base a datos reales y predicciones de la variable objetivo.
        
    - Imputs:
        - y_true (list): Lista con los valores reales de la variable objetivo.
        - y_pred (list): Lista con las probabilidades predecidas por el modelo.
        - size_figure: Tamaño deseado para los gráficos.
        - title (str): Título del gráfico.
        
    - Return:
        - Curva ROC en base a datos reales y predicciones de la variable objetivo.
    ''' 
    if y_true is None or y_pred is None:
        print('\nFaltan parámetros necesarios para la función.')
        return 1

    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    gini = (2 * roc_auc) - 1
    
    gmeans = np.sqrt(tpr * (1-fpr))
    
    ix = np.argmax(gmeans)

    plt.figure(figsize=size_figure)
    plt.plot([0, 1], [0, 1], linestyle='--', color='navy', label='No Skill')
    plt.plot(fpr, tpr, marker='.', label=f'Logistic Regresion (AUC = {roc_auc:.2f})')
    plt.scatter(fpr[ix], tpr[ix], s=100, marker='o', color='black', label='Best Threshold')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)

    plt.legend()

    plt.show()

    print('\n======================================================================')
    print(f'El coeficiente de GINI es: {gini:.2f}')
    print(f'El área bajo la curva ROC (AUC) es: {roc_auc:.2f}')
    print('El mejor Threshold es %.3f, con G-Mean %.3f' % (thresholds[ix], gmeans[ix]))
    print('======================================================================')
    return 0



def plot_precision_recall(y_true=None, y_pred=None, size_figure=[8, 6], title='Curva Precision-Recall'):
    ''' 
    ----------------------------------------------------------------------------------------------------------
    Función plot_roc_gini:
    ----------------------------------------------------------------------------------------------------------
    - Descripción:
        Esta función grafica la curva precision-recall en base a datos reales y predicciones de la variable objetivo.
        
    - Imputs:
        - y_true (list): Lista con los valores reales de la variable objetivo.
        - y_pred (list): Lista con las probabilidades predecidas por el modelo.
        - size_figure: Tamaño deseado para los gráficos.
        - title (str): Título del gráfico.
        
    - Return:
        - Curva precision-recall en base a datos reales y predicciones de la variable objetivo.
    ''' 
    if y_true is None or y_pred is None:
        print('\nFaltan parámetros necesarios para la función.')
        return 1

    # Calcular la curva Precision-Recall
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)

    # Convertir a F-Score con una pequeña constante para evitar división por cero
    epsilon = 1e-8  # Pequeño valor para evitar división por cero
    fscore = (2 * precision * recall) / (precision + recall + epsilon)

    # Localizar el índice del F-Score más alto
    ix = np.argmax(fscore)
    best_threshold = thresholds[ix]

    # Calcular la línea de referencia para "No Skill"
    no_skill = len(y_true[y_true == 1]) / len(y_true)

    # Dibujar la curva Precision-Recall
    plt.figure(figsize=size_figure)

    # Dibujar la línea de No Skill
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', color='navy', label='No Skill')

    # Dibujar la curva del modelo
    plt.plot(recall, precision, marker='.', label='Logistic Regresion')

    # Resaltar el mejor umbral
    plt.scatter(recall[ix], precision[ix], s=100, marker='o', color='black', label=f'Best Threshold')

    # Configurar detalles del gráfico
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend()

    # Mostrar el gráfico
    plt.show()
      
    # Imprimir métricas adicionales
    print('\n======================================================================')
    print('El mejor Threshold es %.3f, con F-Score %.3f' % (best_threshold, fscore[ix]))
    print('======================================================================')

    return 0
