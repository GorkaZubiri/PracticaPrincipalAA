# Práctica principal de la asignatura

**Autor:** Gorka Zubiri Elso

**Correo electrónico:** gorka.zubiri@cunef.edu

**Directorio GitHub:** https://github.com/GorkaZubiri/PracticaPrincipalAA

Esta práctica tiene como objetivo ayudar a un banco a mejorar la aprobación de préstamos mediante el análisis de datos históricos de solicitudes. La idea es identificar patrones en los clientes que cumplen con sus pagos y aquellos que no lo hacen, para aplicar esta información en futuros modelos predictivos.

## Objetivos concretos del proyecto
Los objetivos concretos del proyecto han sido:  

1. **EDA**: Entender cómo están organizados los datos, ver cómo se distribuyen las variables, detectar si hay valores atípicos y faltantes,y detectar relaciones clave entre las variables predictoras y la variable objetivo.  

2. **Selección y Evaluación del Modelo**: Probar y comparar múltiples modelos predictivos, optimizar sus hiperparámetros y evaluar su rendimiento utilizando métricas clave para seleccionar el modelo final.  

3. **Explicabilidad del Modelo**: Aplicar técnicas de explicabilidad global y local para entender el impacto de cada variable en las predicciones, garantizando que el modelo sea transparente y fácil de interpretar.  


## Estructura del Directorio

La estructura del directorio de este proyecto está organizada de la siguiente manera:

- **`data/`**: Contiene los archivos de datos con los que vamos a trabajar.

  - **`raw/`**: Archivos de datos originales, tal como se obtuvieron.
  
  - **`processed/`**: Datos que ya han sido procesados y transformados para su uso.
  
  - **`interim/`**: Datos intermedios que han sido parcialmente procesados y aún no están listos para su uso final.
  
  
- **`env/`**: Archivos relacionados con el entorno de desarrollo, incluyendo un archivo `requirements.txt` con todas las librerías y dependencias utilizadas en el proyecto.


- **`notebook/`**: Contiene los notebooks en formato Jupyter (`.ipynb`) que documentan el análisis de datos y otros experimentos.


- **`html/`**: Carpeta donde se almacenan los notebooks convertidos en formato HTML para facilitar su visualización y compartición.


- **`src/`**: Directorio que guarda los archivos fuente de Python, tales como scripts, funciones o clases utilizadas en el procesamiento de datos o la creación de modelos.

- **`models/`**: Carpeta donde se almacenan los modelo en formato .pickle.

## Notebooks Desarrollados

He desarrollado cinco notebooks en esta práctica:

1. **01_Initial_Exploratory_Analysis**: Su objetivo es realizar un análisis exploratorio inicial para obtener una mejor comprensión del problema que estamos tratando de resolver.
2. **02_Data_Preprocessing_Analysis**:  Se enfoca en el preprocesamiento de variables numéricas y categóricas, ajustando tipos de datos, gestionando outliers, valores faltantes y analizando correlaciones.
3. **03_Feature_Selection_Engineering**: Preparación de los datos para los modelos, seleccionando las variables más relevantes y realizando la ingeniería necesaria para garantizar su buen funcionamiento.
4. **04_Model_Selection_and_Optimization**: En este notebook se define un modelo base, se compara con otros modelos, se selecciona el más adecuado, y se optimizan sus hiperparámetros para evaluar su rendimiento.
5. **05_Shap_Values_and_Explainability**: Se aplica el modelo SHAP para obtener los valores SHAP y entender cómo cada característica afecta las predicciones, con un análisis de explicabilidad global y local del modelo.