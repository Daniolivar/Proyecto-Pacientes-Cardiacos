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
### 🧹 Paso 3: Limpieza de Datos y Validación Clínica

En esta fase preparamos el dataset para el modelado. Esto incluye la estandarización de columnas al español, la conversión de unidades (como la edad de días a años), el manejo de valores nulos y, lo más importante, la aplicación de **filtros médicos** para eliminar registros biológicamente imposibles.

```python
print("Limpieza de Datos")

# 1. Remover registros duplicados del dataset
df = df.drop_duplicates()

# 2. Traducción de nombres de columnas para mejor legibilidad
columnas_traducidas = {
    'age': 'edad', 'gender': 'genero', 'height': 'altura', 
    'weight': 'peso', 'ap_hi': 'presion_sistolica', 
    'ap_lo': 'presion_diastolica', 'cholesterol': 'colesterol', 
    'gluc': 'glucosa', 'smoke': 'fumador', 'alco': 'alcohol', 
    'active': 'activo', 'cardio': 'enfermedad_cardiaca'
}
df = df.rename(columns=columnas_traducidas)

# 3. Transformación de edad de días a años
if 'edad' in df.columns:
    if df['edad'].max() > 150:
        df['edad'] = np.floor(df['edad'] / 365.25).astype(int)
        print("→ Transformación aplicada: edad convertida de días a años.")

# 4. Tratamiento de datos faltantes
total_nulos = df.isnull().sum().sum()
if total_nulos > 0:
    print(f"→ Detectados {total_nulos} valores faltantes. Aplicando imputación...")
    # Imputación (Mediana para numéricas, Moda para categóricas)...
else:
    print("→ Dataset completo: sin valores faltantes detectados.")

print("✅ Proceso de preprocesamiento finalizado exitosamente.\n")

# -----------------------------------------
# FILTRADO POR CRITERIOS MÉDICOS
# -----------------------------------------
print("Filtros de validez Clínica")

# 5. Filtro de Presión Arterial
# La presión sistólica SIEMPRE debe ser mayor que la diastólica y estar en rangos lógicos
condiciones_presion = (
    (df['presion_sistolica'].between(90, 240)) &
    (df['presion_diastolica'].between(60, 160)) &
    (df['presion_sistolica'] > df['presion_diastolica']) 
)
df = df[condiciones_presion]

# 6. Feature Engineering: Creación y filtro de IMC
# Calculamos el Índice de Masa Corporal: Peso / (Altura en metros al cuadrado)
df['imc'] = df['peso'] / ((df['altura'] / 100) ** 2)

# Eliminamos IMCs imposibles (menores a 10 o mayores a 60 son errores de registro)
df = df[df['imc'].between(10, 60)]


```


### 📊 Paso 4: Análisis Exploratorio de Datos (EDA) y Visualización

Para comprender la relación entre nuestras variables numéricas clave y el riesgo de enfermedad cardíaca, generamos diagramas de caja (boxplots) comparativos con una estética científica.

```python
# Configuración estética profesional
sns.set(style="whitegrid", context="talk")

# 1. Definimos las variables que realmente tienes en tu dataset cardiaco
vars_numericas = ['edad', 'presion_sistolica', 'presion_diastolica', 'imc']
titulos = ['Edad (Años)', 'Presión Sistólica (mmHg)', 'Presión Diastólica (mmHg)', 'IMC (kg/m²)']

# 2. Creamos la figura general
plt.figure(figsize=(16, 12)) # Tamaño grande para que quepan todos

# 3. Bucle para generar un gráfico por cada variable
for num, (var, titulo) in enumerate(zip(vars_numericas, titulos), 1):
    plt.subplot(2, 2, num) # Crea una cuadrícula de 2x2

    # EL BOXPLOT (Estilo Científico)
    sns.boxplot(
        data=df,
        x='enfermedad_cardiaca',    # Eje X: El diagnóstico (0 o 1)
        y=var,                      # Eje Y: La variable médica
        palette=['#2ecc71', '#e74c3c'], # Verde (Sano) y Rojo (Enfermo)
        showfliers=False,           # Ocultamos puntos extremos para ver mejor las cajas
        linewidth=2.5,
        width=0.5
    )

    # Etiquetas y limpieza
    plt.title(f'Distribución de {titulo}', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('') # Quitamos etiqueta X redundante
    plt.ylabel(titulo, fontsize=12)
    plt.xticks([0, 1], ['Sin Enfermedad', 'Con Enfermedad'], fontsize=11)

    # Detalle Pro: Línea de la media general punteada (referencia)
    media_general = df[var].median()
    plt.axhline(media_general, color='gray', linestyle='--', alpha=0.7, label=f'Mediana General ({media_general:.1f})')
    plt.legend(fontsize=10)

plt.tight_layout()
plt.show()

```
<img width="1566" height="1166" alt="image" src="https://github.com/user-attachments/assets/e7164106-74a9-4914-b62e-bcb082e19887" />

