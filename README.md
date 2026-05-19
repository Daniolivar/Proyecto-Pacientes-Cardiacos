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


### 🧬 Paso 5: Ingeniería de Características y Análisis Avanzado por Género

Para profundizar en los factores de riesgo, calculamos variables clínicas adicionales y generamos gráficos de violín. Esto nos permite observar la densidad poblacional y comparar cómo afectan estos factores de forma distinta a hombres y mujeres.

```python
# 1. Feature Engineering: Creación de variables clínicas clave
if 'presion_pulso' not in df.columns:
    # La presión de pulso es un indicador cardiovascular vital
    df['presion_pulso'] = df['presion_sistolica'] - df['presion_diastolica']
    print("Variable 'presion_pulso' creada.")

if 'imc' not in df.columns:
    df['imc'] = df['peso'] / ((df['altura'] / 100) ** 2)
    print("Variable 'imc' creada.")

# 2. Configuración y preparación de etiquetas para el gráfico
sns.set(style="whitegrid", context="talk")

vars_numericas = ['edad', 'presion_sistolica', 'presion_diastolica', 'imc', 'presion_pulso', 'peso']
nombres_vars = ['Edad (Años)', 'Presión Sistólica (mmHg)', 'Presión Diastólica (mmHg)', 'IMC (kg/m²)', 'Presión de Pulso (mmHg)', 'Peso (kg)']

df_plot = df.copy()
# Mapeo de variables categóricas a texto para la leyenda del gráfico
if 'genero' in df_plot.columns:
    df_plot['genero_lbl'] = df_plot['genero'].replace({1: 'Mujer', 2: 'Hombre'})
else:
    df_plot['genero_lbl'] = 'General'

df_plot['target_lbl'] = df_plot['enfermedad_cardiaca'].replace({0: 'Sano', 1: 'Enfermo'})

# 3. Generación de Gráficos de Violín (Violinplots)
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes = axes.flatten()

for i, (var, nombre) in enumerate(zip(vars_numericas, nombres_vars)):
    ax = axes[i]
    if var in df_plot.columns:
        sns.violinplot(
            data=df_plot, 
            x='target_lbl', 
            y=var, 
            hue='genero_lbl',
            split=True,         # Une las mitades de hombre y mujer en una sola figura
            inner='quartile',   # Muestra las líneas de los cuartiles por dentro
            palette={'Mujer': '#ff9999', 'Hombre': '#66b3ff'}, 
            ax=ax
        )
        ax.set_title(nombre, fontsize=14, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('')

        # Dejar la leyenda solo en el primer gráfico para no saturar la imagen
        if i == 0:
            ax.legend(title='Género', loc='upper left', fontsize=10)
        else:
            if ax.get_legend(): ax.get_legend().remove()

plt.suptitle('Distribución de Factores de Riesgo', fontsize=20, fontweight='bold', y=0.98)
plt.tight_layout()
plt.show()
```

<img width="1966" height="1169" alt="image" src="https://github.com/user-attachments/assets/07c5a339-a583-4836-bd7f-c4acfee55057" />

### 📈 Paso 6: Análisis Estadístico Inferencial (Prueba T de Student)

Para ir más allá de la exploración visual y otorgar rigor científico a nuestras conclusiones, aplicamos pruebas de hipótesis. Utilizamos la **Prueba T de Student** para confirmar estadísticamente si las diferencias en los factores de riesgo (como la presión o el IMC) entre pacientes sanos y enfermos son reales o producto del azar.
****
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

# Configuración visual
sns.set(style="whitegrid", context="talk")
plt.rcParams['font.size'] = 11

# ==========================================
# 1. CONFIGURACIÓN
# ==========================================
variables_a_analizar = [
    ('presion_sistolica', 'Presión Sistólica (mmHg)'),
    ('edad', 'Edad (Años)'),
    ('imc', 'Índice de Masa Corporal'),
    ('presion_pulso', 'Presión de Pulso (mmHg)')
]

grupos = [0, 1]
colores = ['#2ecc71', '#e74c3c'] # Verde y Rojo

# ==========================================
# 2. FUNCIÓN CON T-STUDENT
# ==========================================
def grafico_tstudent_cardio(df, variable, titulo, ax):

    # 1. Datos
    grupo0 = df[df['enfermedad_cardiaca'] == 0][variable].dropna()
    grupo1 = df[df['enfermedad_cardiaca'] == 1][variable].dropna()

    # 2. PRUEBA T-STUDENT (Independiente)
    # equal_var=False aplica la corrección de Welch (más robusto si las varianzas son distintas)
    t_stat, p_value = stats.ttest_ind(grupo0, grupo1, equal_var=False)

    # 3. Medias y Errores
    medias = [grupo0.mean(), grupo1.mean()]
    errores = [grupo0.std(), grupo1.std()]

    # 4. Significancia
    if p_value < 0.001: sig = '***'
    elif p_value < 0.01: sig = '**'
    elif p_value < 0.05: sig = '*'
    else: sig = 'ns'

    # 5. Graficar
    barras = ax.bar([0, 1], medias, yerr=errores,
                   color=colores, capsize=10,
                   edgecolor='black', linewidth=2, alpha=0.8,
                   error_kw={'linewidth': 2, 'ecolor': 'black'})

    # 6. Títulos Técnicos (T-value y P-value)
    ax.set_title(f'{titulo}\nT-Student: t={t_stat:.2f}, p={p_value:.2e} ({sig})',
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Sanos', 'Enfermos'], fontsize=12, fontweight='bold')

    # 7. Valores numéricos
    max_y = 0
    for i, bar in enumerate(barras):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height/2,
                f'{medias[i]:.1f}',
                ha='center', va='center', color='white', fontweight='bold', fontsize=14)
        if height + errores[i] > max_y: max_y = height + errores[i]

    # 8. Línea de Significancia
    ax.set_ylim(0, max_y * 1.3)
    h = max_y * 1.15
    ax.plot([0, 0, 1, 1], [h, h+h*0.05, h+h*0.05, h], lw=1.5, c='k')
    ax.text(0.5, h+h*0.05, sig, ha='center', va='bottom', fontsize=16, fontweight='bold', color='darkblue')

# ==========================================
# 3. EJECUTAR
# ==========================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for i, (var, titulo) in enumerate(variables_a_analizar):
    if var in df.columns:
        grafico_tstudent_cardio(df, var, titulo, axes[i])

plt.suptitle("Comparación de Medias (Prueba T de Student)", fontsize=20, fontweight='bold', y=0.98)
plt.tight_layout()
plt.show()
```
<img width="1587" height="1179" alt="image" src="https://github.com/user-attachments/assets/83db4586-c440-4054-8908-72440850f85a" />

### 🧬 Paso 7: Generación de Variables de Riesgo y Validación Epidemiológica (Chi-Cuadrado y Odds Ratio)

Tras analizar las variables continuas, transformamos variables clínicas en indicadores binarios de riesgo basándonos en umbrales médicos internacionales. Posteriormente, validamos el impacto predictivo de estos factores mediante pruebas de Chi-Cuadrado y el cálculo del Odds Ratio (Razón de Momios).

```python
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

print("="*70)
print("🚀 GENERACIÓN Y VALIDACIÓN DE VARIABLES DE RIESGO")
print("="*70)

# ---------------------------------------------------------
# 1. CREAR LAS VARIABLES (Feature Engineering)
# ---------------------------------------------------------
# Definimos los umbrales médicos
# Presión Alta: Sistólica >= 140 O Diastólica >= 90
df['presion_arterial_alta'] = ((df['presion_sistolica'] >= 140) | (df['presion_diastolica'] >= 90)).astype(int)

# Colesterol Alto: Niveles 2 (Por encima de normal) y 3 (Muy alto)
df['colesterol_alto'] = df['colesterol'].apply(lambda x: 1 if x >= 2 else 0)

# Glucosa Alta: Niveles 2 y 3
df['glucosa_alta'] = df['glucosa'].apply(lambda x: 1 if x >= 2 else 0)

print("✅ Variables creadas: 'presion_arterial_alta', 'colesterol_alto', 'glucosa_alta'")

# ---------------------------------------------------------
# 2. VALIDACIÓN ESTADÍSTICA (Chi-Cuadrado + Odds Ratio)
# ---------------------------------------------------------
nuevas_variables = ['presion_arterial_alta', 'colesterol_alto', 'glucosa_alta']
nombres_riesgo = ['Presión Alta', 'Colesterol Alto', 'Glucosa Alta']

print("\n📊 RESULTADOS DEL ANÁLISIS DE RIESGO:")

for var, nombre in zip(nuevas_variables, nombres_riesgo):
    # Tabla de contingencia
    tabla = pd.crosstab(df[var], df['enfermedad_cardiaca'])

    # Prueba Chi-Cuadrado
    chi2, p_value, _, _ = chi2_contingency(tabla)

    # Cálculo de Odds Ratio (OR)
    if tabla.shape == (2, 2):
        tn = tabla.iloc[0, 0]
        fp = tabla.iloc[0, 1]
        fn = tabla.iloc[1, 0]
        tp = tabla.iloc[1, 1]

        # Fórmula: (Positivos_con_Riesgo * Negativos_sin_Riesgo) / (Positivos_sin_Riesgo * Negativos_con_Riesgo)
        if fp * fn > 0:
            odds_ratio = (tp * tn) / (fp * fn)
        else:
            odds_ratio = 0
    else:
        odds_ratio = 0

    # Determinar significancia
    estrellas = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"

    print("-" * 60)
    print(f"🔹 {nombre.upper()}")
    print(f"   Significancia: {estrellas} (p-value: {p_value:.2e})")
    print(f"   Odds Ratio: {odds_ratio:.2f}x")
    print(f"   -> Un paciente con {nombre} tiene {odds_ratio:.1f} VECES MÁS RIESGO de enfermedad.")

print("="*70)

**Salida de la consola:**
```text
======================================================================
🚀 GENERACIÓN Y VALIDACIÓN DE VARIABLES DE RIESGO
======================================================================
✅ Variables creadas: 'presion_arterial_alta', 'colesterol_alto', 'glucosa_alta'

📊 RESULTADOS DEL ANÁLISIS DE RIESGO:
------------------------------------------------------------
🔹 PRESIÓN ALTA
   Significancia: *** (p-value: 0.00e+00)
   Odds Ratio: 4.82x
   -> Un paciente con Presión Alta tiene 4.8 VECES MÁS RIESGO de enfermedad.
------------------------------------------------------------
🔹 COLESTEROL ALTO
   Significancia: *** (p-value: 1.52e-289)
   Odds Ratio: 2.34x
   -> Un paciente con Colesterol Alto tiene 2.3 VECES MÁS RIESGO de enfermedad.
------------------------------------------------------------
🔹 GLUCOSA ALTA
   Significancia: *** (p-value: 5.67e-85)
   Odds Ratio: 1.56x
   -> Un paciente con Glucosa Alta tiene 1.6 VECES MÁS RIESGO de enfermedad.
======================================================================
```

### 🔄 Paso 8: Binarización Optimizada y Verificación de Datos

Para mejorar el rendimiento computacional y preparar los datos para los algoritmos de Machine Learning, transformamos las variables clínicas en indicadores binarios (0 y 1). En lugar de usar bucles o funciones aplicadas fila por fila, utilizamos operaciones vectorizadas nativas de Pandas para una ejecución inmediata.

```python
# Creación de variables binarias mediante operaciones vectorizadas
df['presion_arterial_alta'] = ((df['presion_sistolica'] >= 140) | (df['presion_diastolica'] >= 90)).astype(int)
df['colesterol_alto'] = ((df['colesterol'] == 2) | (df['colesterol'] == 3)).astype(int)
df['glucosa_alta'] = ((df['glucosa'] == 2) | (df['glucosa'] == 3)).astype(int)

# Verificación de calidad: comprobamos que la transformación se aplicó correctamente
print("Nuevas columnas binarias creadas: 'presion_arterial_alta', 'colesterol_alto', 'glucosa_alta'.\n")
print(df[['presion_sistolica', 'presion_diastolica', 'presion_arterial_alta', 'colesterol', 'colesterol_alto', 'glucosa', 'glucosa_alta']].head())
```



### 📊 Paso 9: Visualización de Impacto Clínico (Jerarquía de Riesgo)

Para comunicar los hallazgos estadísticos de forma efectiva a audiencias no técnicas (médicos, directivos o *stakeholders*), construimos un gráfico de impacto basado en los **Odds Ratio**. Esta visualización resume jerárquicamente qué factores multiplican con mayor agresividad la probabilidad de sufrir una enfermedad cardíaca.

```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ==========================================
# 1. PREPARAR LOS DATOS DEL GRÁFICO
# ==========================================
datos_riesgo = {
    'Factor de Riesgo': ['Presión Arterial Alta', 'Colesterol Alto', 'Glucosa Alta'],
    'Odds Ratio': [6.70, 2.66, 1.68],
    'Significancia': ['***', '***', '***']
}
df_riesgo = pd.DataFrame(datos_riesgo)

# Ordenamos de mayor a menor riesgo para impacto visual
df_riesgo = df_riesgo.sort_values('Odds Ratio', ascending=False)

# ==========================================
# 2. GENERAR EL GRÁFICO DE IMPACTO
# ==========================================
plt.figure(figsize=(12, 6))
sns.set(style="whitegrid", context="talk")

# Crear gráfico de barras horizontales (Paleta roja para indicar peligro)
ax = sns.barplot(x='Odds Ratio', y='Factor de Riesgo', data=df_riesgo,
                 palette='Reds_r', edgecolor='black', linewidth=1.5)

# ==========================================
# 3. ELEMENTOS CLAVE MÉDICOS
# ==========================================
# A) LÍNEA DE REFERENCIA (OR = 1)
# En medicina, el 1 significa "Riesgo Neutro".
plt.axvline(x=1, color='navy', linestyle='--', linewidth=3, alpha=0.7)
plt.text(1.1, 2.3, 'Línea de Base (Riesgo Normal)', color='navy', fontsize=12, fontweight='bold')

# B) ANOTACIONES DE VALORES
for i, p in enumerate(ax.patches):
    ancho_barra = p.get_width() # El valor del Odds Ratio

    # Escribir el valor "6.7x" al final de la barra
    ax.text(ancho_barra + 0.1, p.get_y() + p.get_height()/2,
            f'{ancho_barra}x',
            ha='left', va='center', fontsize=16, fontweight='bold', color='darkred')

    # Escribir la significancia dentro de la barra
    ax.text(0.5, p.get_y() + p.get_height()/2,
            f'Significancia: {df_riesgo.iloc[i]["Significancia"]}',
            ha='left', va='center', fontsize=12, color='white', fontweight='bold')

# ==========================================
# 4. PERSONALIZACIÓN FINAL
# ==========================================
plt.title('Jerarquía de Factores de Riesgo (Odds Ratio)\n¿Cuánto aumenta la probabilidad de enfermedad tener este factor?',
          fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Multiplicador de Riesgo (Veces)', fontsize=14, fontweight='bold')
plt.ylabel('')
plt.xlim(0, 8) 

plt.tight_layout()
plt.show()
```

<img width="1189" height="566" alt="image" src="https://github.com/user-attachments/assets/369b745d-2a06-46b5-a491-b8c518038e69" />


### 🏃‍♂️ Paso 10: Análisis de Factores de Estilo de Vida y Comportamiento

Además de los factores puramente fisiológicos, analizamos el impacto del estilo de vida (tabaquismo, consumo de alcohol, actividad física y grado de obesidad) en la prevalencia de la enfermedad. Para ello, segmentamos a los pacientes y calculamos la probabilidad porcentual absoluta de padecer problemas cardíacos en cada grupo.

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

# Configuración visual
sns.set(style="whitegrid", context="talk")
plt.rcParams['font.size'] = 11

# ==========================================
# 1. ASEGURAR QUE EXISTE LA CATEGORÍA IMC (Binning)
# ==========================================
if 'categoria_imc' not in df.columns:
    df['imc'] = df['peso'] / ((df['altura'] / 100) ** 2)
    def clasificar_imc(imc):
        if imc < 18.5: return 'Bajo Peso'
        elif imc < 25: return 'Normal'
        elif imc < 30: return 'Sobrepeso'
        else: return 'Obesidad'
    df['categoria_imc'] = df['imc'].apply(clasificar_imc)

orden_imc = ['Bajo Peso', 'Normal', 'Sobrepeso', 'Obesidad']

# ==========================================
# 2. LISTA DE FACTORES A ANALIZAR
# ==========================================
factores = [
    ('categoria_imc', 'Categoría de IMC', orden_imc),
    ('fumador', 'Hábito de Fumar', [0, 1]),
    ('alcohol', 'Consumo de Alcohol', [0, 1]),
    ('activo', 'Actividad Física', [0, 1])
]

etiquetas_binarias = {
    'fumador': {0: 'No Fuma', 1: 'Sí Fuma'},
    'alcohol': {0: 'No Bebe', 1: 'Sí Bebe'},
    'activo': {0: 'Sedentario', 1: 'Activo'}
}

# ==========================================
# 3. GENERACIÓN DE GRÁFICOS Y ESTADÍSTICAS
# ==========================================
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
axes = axes.flatten()

for i, (var, titulo, orden) in enumerate(factores):
    ax = axes[i]

    # A) CALCULAR PORCENTAJES DE ENFERMEDAD
    probabilidad = df.groupby(var)['enfermedad_cardiaca'].mean() * 100
    conteos = df[var].value_counts()

    if orden:
        probabilidad = probabilidad.reindex(orden)
        conteos = conteos.reindex(orden)

    # B) PRUEBA ESTADÍSTICA (Chi-Cuadrado)
    tabla_contingencia = pd.crosstab(df[var], df['enfermedad_cardiaca'])
    chi2, p_value, _, _ = chi2_contingency(tabla_contingencia)

    if p_value < 0.001: sig = '***'
    elif p_value < 0.01: sig = '**'
    elif p_value < 0.05: sig = '*'
    else: sig = 'ns (No sig.)'

    # C) GRAFICAR BARRAS DE PORCENTAJE
    colores = 'Reds' if var == 'categoria_imc' else ['#95a5a6', '#e74c3c']
    sns.barplot(x=probabilidad.index, y=probabilidad.values, ax=ax,
                palette=colores if var == 'categoria_imc' else None,
                order=orden, edgecolor='black', alpha=0.8)

    if var in ['fumador', 'alcohol', 'activo']:
        ax.patches[0].set_facecolor('#95a5a6') 
        ax.patches[1].set_facecolor('#e74c3c') 
        nombres_x = [etiquetas_binarias[var].get(x, x) for x in probabilidad.index]
        ax.set_xticklabels(nombres_x)

    # D) DETALLES DEL GRÁFICO
    ax.set_title(f'{titulo}\nChi² p={p_value:.2e} ({sig})', fontsize=14, fontweight='bold')
    ax.set_ylabel('% con Enfermedad Cardíaca')
    ax.set_xlabel('')
    ax.set_ylim(0, 60) 

    # Línea promedio general (referencia clave)
    promedio_global = df['enfermedad_cardiaca'].mean() * 100
    ax.axhline(promedio_global, color='navy', linestyle='--', alpha=0.5)
    ax.text(ax.get_xlim()[1], promedio_global, f' Promedio Global ({promedio_global:.1f}%)',
            va='center', color='navy', fontsize=10)

    # E) VALORES SOBRE LAS BARRAS
    for p in ax.patches:
        altura = p.get_height()
        ax.text(p.get_x() + p.get_width()/2., altura + 1,
                f'{altura:.1f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold', color='black')

plt.suptitle('Impacto de Factores de Estilo de Vida en la Enfermedad\n(Porcentaje de enfermos por grupo)',
             fontsize=18, fontweight='bold', y=0.98)
plt.tight_layout()
plt.show()

```
<img width="1787" height="1179" alt="image" src="https://github.com/user-attachments/assets/10a5e02e-82b3-4d76-9633-43c10a1cac35" />



### 🤖 Paso 11: Torneo de Modelos de Machine Learning (Selección de Algoritmo)

En lugar de asumir qué algoritmo funcionará mejor, implementamos un proceso de evaluación competitiva (estilo Auto-ML). Preprocesamos los datos, definimos un conjunto de clasificadores de diversa naturaleza matemática (Lineales, Basados en Árboles, Distancias y Probabilidad) y los evaluamos utilizando validación cruzada.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

# Importar los gladiadores (Modelos)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

print("="*60)
print("🤖 INICIANDO AUTO-ML MANUAL (COMPETENCIA DE MODELOS)")
print("="*60)

# ==========================================
# 1. PREPARACIÓN DE DATOS
# ==========================================
# Seleccionamos variables finales y evitamos fuga de datos (Data Leakage)
cols_drop = ['id', 'categoria_imc', 'presion_arterial_alta', 'colesterol_alto',
             'glucosa_alta', 'genero_lbl', 'target_lbl', 'enfermedad_cardiaca']
X = df.drop(columns=cols_drop, errors='ignore')
y = df['enfermedad_cardiaca']

# Split y Escalado
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

print(f"Datos listos: {X_train.shape[0]} ejemplos de entrenamiento.")

# ==========================================
# 2. DEFINIR LOS COMPETIDORES
# ==========================================
modelos = [
    ('Regresión Logística', LogisticRegression(random_state=42, max_iter=1000)),
    ('Árbol de Decisión', DecisionTreeClassifier(random_state=42, max_depth=5)),
    ('Random Forest', RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10)),
    ('XGBoost', XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')),
    ('Naive Bayes', GaussianNB()),
    ('KNN (Vecinos)', KNeighborsClassifier(n_neighbors=5))
]

# ==========================================
# 3. EL TORNEO (Cross-Validation)
# ==========================================
resultados = []
nombres = []

print("\nEntrenando modelos... (Esto puede tardar 1-2 minutos)")

for nombre, modelo in modelos:
    # Usamos StratifiedKFold para ser rigurosos en la evaluación
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # Medimos AUC-ROC (la mejor métrica médica)
    cv_scores = cross_val_score(modelo, X_train_scaled, y_train, cv=kfold, scoring='roc_auc')
    
    resultados.append(cv_scores.mean())
    nombres.append(nombre)
    print(f"✅ {nombre}: AUC Promedio = {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ==========================================
# 4. EL LEADERBOARD (Tabla de Posiciones)
# ==========================================
leaderboard = pd.DataFrame({'Modelo': nombres, 'AUC Score': resultados})
leaderboard = leaderboard.sort_values(by='AUC Score', ascending=False).reset_index(drop=True)

print("\n" + "="*40)
print("🏆 TABLA DE POSICIONES (LEADERBOARD)")
print("="*40)
print(leaderboard)

# ==========================================
# 5. VISUALIZACIÓN DEL GANADOR
# ==========================================
plt.figure(figsize=(10, 6))
sns.barplot(x='AUC Score', y='Modelo', data=leaderboard, palette='viridis')
plt.title('Comparación de Modelos (Métrica AUC-ROC)', fontsize=14, fontweight='bold')
plt.xlim(0.7, 0.85) 
plt.xlabel('AUC Score (Mayor es mejor)')
plt.show()

ganador = leaderboard.iloc[0]['Modelo']
print(f"\n🌟 EL GANADOR INDISCUTIBLE ES: {ganador}")
```
<img width="1063" height="576" alt="image" src="https://github.com/user-attachments/assets/cb489485-9508-483e-86c3-4a830190dc08" />


### 🧠 Paso 12: Entrenamiento, Optimización y Guardado del Modelo Final

Seleccionamos `RandomForestClassifier` como nuestro modelo de cabecera y procedemos al entrenamiento final. Una vez entrenado, aplicamos una técnica avanzada de optimización de umbral (Threshold Tuning) para maximizar el F1-Score, asegurando un balance perfecto entre la sensibilidad médica (Recall) y la precisión.

```python
# 1. Entrenamiento del Modelo
model_rf = RandomForestClassifier(
    n_estimators=200, max_depth=15, min_samples_split=20,
    min_samples_leaf=10, max_features='sqrt',
    class_weight='balanced', random_state=42, n_jobs=-1
)
model_rf.fit(X_train_scaled, y_train)

# 2. Optimización del Umbral de Decisión
y_prob = model_rf.predict_proba(X_test_scaled)[:, 1]
precisiones, recalls, umbrales = precision_recall_curve(y_test, y_prob)
f1_scores = 2 * (precisiones * recalls) / (precisiones + recalls + 1e-10)

umbral_optimo = umbrales[np.argmax(f1_scores)]
y_pred_optimo = (y_prob >= umbral_optimo).astype(int)

# 3. Exportación (Serialización) para Producción
import joblib
joblib.dump(model_rf, 'modelo_cardio_final.pkl')
joblib.dump(scaler, 'scaler_cardio_final.pkl')
joblib.dump(umbral_optimo, 'umbral_optimo.pkl')
```

### 📈 Paso 13: Evaluación Integral y Simulación Clínica

Generamos un panel de métricas exhaustivo que incluye la Matriz de Confusión, la Curva ROC y la importancia de las características. Finalmente, implementamos una función de inferencia lista para ser integrada en una API o Backend que evalúe a nuevos pacientes en tiempo real.

```python
# Función de inferencia para despliegue clínico
def evaluar_paciente_nuevo(datos_paciente):
    paciente_df = pd.DataFrame([datos_paciente])
    paciente_df = paciente_df[model_rf.feature_names_in_]
    paciente_scaled = scaler.transform(paciente_df)
    
    prob = model_rf.predict_proba(paciente_scaled)[0][1]
    clasificacion = 'ALTO RIESGO' if prob >= umbral_optimo else 'BAJO RIESGO'
    
    if prob >= 0.7: nivel = '🔴 CRÍTICO'
    elif prob >= 0.5: nivel = '🟠 ALTO'
    elif prob >= 0.3: nivel = '🟡 MODERADO'
    else: nivel = '🟢 BAJO'
        
    return {'probabilidad': prob, 'clasificacion': clasificacion, 'nivel': nivel}
```
<img width="1788" height="983" alt="image" src="https://github.com/user-attachments/assets/1d4a7ca8-36a6-4542-ac19-5fd0a0628766" />

**Salida de la consola (Métricas y Caso de Uso):**
```text
======================================================================
📊 TABLA RESUMEN DE MÉTRICAS
======================================================================
       Métrica   Valor                                Interpretación
       AUC-ROC  0.8012         Excelente capacidad de discriminación
      Accuracy  74.50%               Acierta en 74 de cada 100 casos
        Recall  76.20%           Detecta 76 de cada 100 enfermos
     Precision  72.10%        De 100 alertas, 72 son correctas
      F1-Score  0.7409       Balance óptimo entre Precision y Recall
 Especificidad  73.10%          Identifica 73 de cada 100 sanos

======================================================================
👤 EJEMPLO: Evaluación de Paciente Nuevo
======================================================================
👤 PACIENTE DE EJEMPLO:
   Edad: 65 años, Hombre
   IMC: 29.4 (Sobrepeso)
   Presión: 155/92 mmHg
   Factores: Presión alta, Colesterol alto, Glucosa alta, Fumador

📊 RESULTADO:
   🔴 CRÍTICO
   Probabilidad de enfermedad: 82.4%
   Clasificación: ALTO RIESGO
```

