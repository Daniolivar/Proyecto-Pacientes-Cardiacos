<h1> Análisis y Predicción de Riesgo Cardiovascular </h1> 
<p>Este proyecto tiene como objetivo analizar variables clínicas y predecir el riesgo de que un paciente desarrolle enfermedades cardiovasculares utilizando técnicas de análisis exploratorio de datos (EDA) y algoritmos de Machine Learning.
</p>

<h1> Descripción del Proyecto</h1>
<p>Las enfermedades cardiovasculares son una de las principales causas de mortalidad a nivel mundial. En este proyecto, analizamos un conjunto de datos médicos para identificar patrones y factores de riesgo clave. Posteriormente, implementamos un modelo predictivo capaz de estimar la probabilidad de enfermedad y simulamos diferentes escenarios clínicos (riesgo actual vs. riesgo con intervención médica)</p>



<h2>Configuración e Importación de Librerías</h2>
Para iniciar el proyecto, el primer paso es preparar nuestro entorno de trabajo. A continuación se muestran las librerías principales que utilizamos y su propósito dentro del análisis

```python
import kagglehub
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```


### 🔍 Paso 2: Exploración Inicial de los Datos

Una vez cargado el dataset, realizamos una inspección preliminar para comprender su estructura, el tamaño de la muestra y los tipos de datos con los que vamos a trabajar:

```python
# Verificamos las dimensiones del dataset
df.shape
# Salida: (70000, 13)

# Obtenemos información detallada sobre las columnas y valores nulos
df.info()

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 70000 entries, 0 to 69999
Data columns (total 13 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   id           70000 non-null  int64  
 1   age          70000 non-null  int64  
 2   gender       70000 non-null  int64  
 3   height       70000 non-null  int64  
 4   weight       70000 non-null  float64
 5   ap_hi        70000 non-null  int64  
 6   ap_lo        70000 non-null  int64  
 7   cholesterol  70000 non-null  int64  
 8   gluc         70000 non-null  int64  
 9   smoke        70000 non-null  int64  
 10  alco         70000 non-null  int64  
 11  active       70000 non-null  int64  
 12  cardio       70000 non-null  int64  
dtypes: float64(1), int64(12)
memory usage: 6.9 MB

```

```python
df.shape
(70000, 13)

df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 70000 entries, 0 to 69999
Data columns (total 13 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   id           70000 non-null  int64  
 1   age          70000 non-null  int64  
 2   gender       70000 non-null  int64  
 3   height       70000 non-null  int64  
 4   weight       70000 non-null  float64
 5   ap_hi        70000 non-null  int64  
 6   ap_lo        70000 non-null  int64  
 7   cholesterol  70000 non-null  int64  
 8   gluc         70000 non-null  int64  
 9   smoke        70000 non-null  int64  
 10  alco         70000 non-null  int64  
 11  active       70000 non-null  int64  
 12  cardio       70000 non-null  int64  
dtypes: float64(1), int64(12)
memory usage: 6.9 MB

```

```python

print("Limpieza de Datos")

# Remover registros duplicados del dataset
df = df.drop_duplicates()

# Diccionario de traducción de nombres de columnas
columnas_traducidas = {
    'age': 'edad',
    'gender': 'genero',
    'height': 'altura',
    'weight': 'peso',
    'ap_hi': 'presion_sistolica',
    'ap_lo': 'presion_diastolica',
    'cholesterol': 'colesterol',
    'gluc': 'glucosa',
    'smoke': 'fumador',
    'alco': 'alcohol',
    'active': 'activo',
    'cardio': 'enfermedad_cardiaca',
}

# Aplicar renombramiento solo a columnas presentes
columnas_a_renombrar = {k: v for k, v in columnas_traducidas.items() if k in df.columns}
df = df.rename(columns=columnas_a_renombrar)

# Identificar y renombrar la variable dependiente
if 'enfermedad_cardiaca' not in df.columns:
    variables_objetivo = ['cardio', 'HeartDisease', 'target', 'disease']
    columna_encontrada = False

    for nombre_col in variables_objetivo:
        if nombre_col in df.columns:
            df = df.rename(columns={nombre_col: 'enfermedad_cardiaca'})
            print(f" Variable objetivo identificada: '{nombre_col}' renombrada exitosamente")
            columna_encontrada = True
            break

    if not columna_encontrada:
        raise KeyError("Error: No se pudo identificar la variable objetivo en el dataset.")

# Normalizar variable objetivo a formato binario
df['enfermedad_cardiaca'] = df['enfermedad_cardiaca'].apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)

# Transformar edad de días a años cuando sea necesario
if 'edad' in df.columns:
    valor_maximo_edad = df['edad'].max()
    if valor_maximo_edad > 150:
        df['edad'] = np.floor(df['edad'] / 365.25).astype(int)
        print("→ Transformación aplicada: edad convertida de días a años.")

# Análisis y tratamiento de datos faltantes
total_nulos = df.isnull().sum().sum()

if total_nulos > 0:
    print(f"→ Detectados {total_nulos} valores faltantes. Aplicando imputación...")

    # Imputación por mediana en variables numéricas
    columnas_numericas = df.select_dtypes(include=[np.number]).columns
    for columna in columnas_numericas:
        if df[columna].isnull().any():
            df[columna] = df[columna].fillna(df[columna].median())

    # Imputación por moda en variables categóricas
    columnas_categoricas = df.select_dtypes(exclude=[np.number]).columns
    for columna in columnas_categoricas:
        if df[columna].isnull().any():
            moda_calculada = df[columna].mode()
            valor_relleno = moda_calculada[0] if len(moda_calculada) > 0 else 'Desconocido'
            df[columna] = df[columna].fillna(valor_relleno)
else:
    print("→ Dataset completo: sin valores faltantes detectados.")

print("✅ Proceso de preprocesamiento finalizado exitosamente.\n")


# FILTRADO POR CRITERIOS MÉDICOS

print("Filtros de validez Clinica ")

# Establecer rangos aceptables para presión arterial
condiciones_presion = (
    (df['presion_sistolica'].between(90, 220)) &
    (df['presion_diastolica'].between(60, 140)) &
    (df['presion_diastolica'] < df['presion_sistolica'])
)
df = df[condiciones_presion]

# --- PEGA ESTO EN SU LUGAR ---

# 1. Filtro de Presión (Esto ya lo tenías bien, pero aseguramos la lógica)
condiciones_presion = (
    (df['presion_sistolica'].between(90, 240)) &
    (df['presion_diastolica'].between(60, 160)) &
    (df['presion_sistolica'] > df['presion_diastolica']) # La sistólica SIEMPRE debe ser mayor
)
df = df[condiciones_presion]

# 2. Filtro de IMC (La mejora clave)
# Calculamos IMC: Peso / (Altura en metros al cuadrado)
df['imc'] = df['peso'] / ((df['altura'] / 100) ** 2)

# Eliminamos IMCs imposibles (menores a 10 o mayores a 60 son errores casi seguro)
df = df[df['imc'].between(10, 60)]

registros_finales = len(df)
print(f"→ Registros válidos finales: {registros_finales}")


```
