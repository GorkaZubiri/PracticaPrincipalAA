# Práctica principal de la asignatura

**Autor:** Gorka Zubiri Elso

**Correo electrónico:** gorka.zubiri@cunef.edu

**Directorio GitHub:** https://github.com/GorkaZubiri/practica1

Esta práctica tiene como objetivo ayudar a un banco a mejorar la aprobación de préstamos mediante el análisis de datos históricos de solicitudes. La idea es identificar patrones en los clientes que cumplen con sus pagos y aquellos que no lo hacen, para aplicar esta información en futuros modelos predictivos.


## Estructura del Directorio

La estructura del directorio de este proyecto está organizada de la siguiente manera:

- **`data/`**: Contiene los archivos de datos con los que vamos a trabajar.

  - **`raw/`**: Archivos de datos originales, tal como se obtuvieron.
  
  - **`processed/`**: Datos que ya han sido procesados y transformados para su uso.
  
  - **`interim/`**: Datos intermedios que han sido parcialmente procesados y aún no están listos para su uso final.
  
  
- **`env/`**: Archivos relacionados con el entorno de desarrollo, incluyendo un archivo `requirements.txt` con todas las librerías y dependencias utilizadas en el proyecto.


- **`notebooks/`**: Contiene los notebooks en formato Jupyter (`.ipynb`) que documentan el análisis de datos y otros experimentos.


- **`html/`**: Carpeta donde se almacenan los notebooks convertidos en formato HTML para facilitar su visualización y compartición.


- **`src/`**: Directorio que guarda los archivos fuente de Python, tales como scripts, funciones o clases utilizadas en el procesamiento de datos o la creación de modelos.

- **`models/`**: Carpeta donde se almacenan los .pickle.

## Notebooks Desarrollados

He desarrollado cinco notebooks en este trabajo práctica:

1. **01_Initial_Exploratory_Analysis**: Su objetivo es realizar un análisis exploratorio inicial para obtener una mejor comprensión del problema que estamos tratando de resolver.
2. **02_Data_Preprocessing_Analysis**:  Se enfoca en el preprocesamiento de variables numéricas y categóricas, ajustando tipos de datos, gestionando outliers, valores faltantes y analizando correlaciones.
3. **03_Feature_Selection_Engineering**: Preparación de los datos para los modelos, seleccionando las variables más relevantes y realizando la ingeniería necesaria para garantizar su buen funcionamiento.
4. **04_Model_Selection_and_Optimization**: En este notebook se define un modelo base, se compara con otros modelos, se selecciona el más adecuado, y se optimizan sus hiperparámetros para evaluar su rendimiento.
5. **05_Shap_Values_and_Explainability**: Se aplica el modelo SHAP para obtener los valores SHAP y entender cómo cada característica afecta las predicciones, con un análisis de explicabilidad global y local del modelo.