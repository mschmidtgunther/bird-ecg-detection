# Detección automática de Bloqueo Incompleto de Rama Derecha (BIRD)

## 📌 Descripción del Proyecto
[cite_start]Este proyecto implementa un sistema automatizado de procesamiento y análisis de señales electrocardiográficas (ECG) de 12 derivaciones para la identificación de patrones morfológicos compatibles con el Bloqueo Incompleto de Rama Derecha (BIRD)[cite: 10]. [cite_start]A diferencia de otras patologías que alteran el ritmo, el BIRD se manifiesta principalmente a través de cambios sutiles en la morfología del complejo QRS[cite: 5].

## 🚀 Pipeline de Procesamiento
El algoritmo está diseñado de manera modular siguiendo las siguientes etapas secuenciales:
1. **Carga de Datos:** Lectura y caracterización de señales multicanal provenientes de la base de datos PTB-XL (500 Hz).
2. **Preprocesamiento:** Filtrado digital mediante un filtro Pasa-Altos (0.5 Hz) para deriva de línea de base, un filtro Notch (50 Hz) para interferencia eléctrica, y un filtro Pasa-Bajos (35-45 Hz) para ruido muscular.
3. **Detección y Segmentación:** Localización de los complejos QRS mediante el Algoritmo de Pan-Tompkins y aislamiento de latidos individuales.
4. **Extracción de Características:** Cuantificación de la duración del QRS (criterio BIRD: entre 100ms y 120ms), detección de morfología rSR' (orejas de conejo) en derivaciones V1 y V2, y análisis de la onda S en las derivaciones laterales (I y V6).
5. **Clasificación:** Comparación de un sistema heurístico basado en reglas médicas frente a modelos de aprendizaje automático (Random Forest / SVM).

## 📂 Estructura del Repositorio
* `data/`: Almacenamiento local de la base de datos (protegido por `.gitignore`).
* `notebooks/`: Cuadernos de Jupyter para experimentación y gráficos rápidos.
* `src/`: Scripts modulares en Python con el núcleo algorítmico del pipeline.

## Ejecución del Notebook

### Requisitos

* macOS o Linux con Python 3.12.
* Dataset PTB-XL descargado localmente.
* Entorno virtual Python (`.venv`) dentro del repositorio.

El notebook principal es:

```bash
notebooks/pipeline_sin_ml.ipynb
```

### Dataset

El notebook espera la carpeta raíz del dataset PTB-XL, es decir, la carpeta que contiene:

```text
ptbxl_database.csv
records100/
records500/
RECORDS
```

En la máquina local usada para este proyecto, el dataset está en:

```bash
/Users/agustinaperini/Desktop/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3
```

Si el dataset está en otra ubicación, se puede indicar con la variable de entorno `PTBXL_DATA_DIR` antes de ejecutar el notebook.

### Crear el Entorno

Desde la raíz del repositorio:

```bash
cd /Users/agustinaperini/Documents/GitHub/bird-ecg-detection
/opt/homebrew/bin/python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Registrar el kernel de Jupyter:

```bash
python -m ipykernel install --user --name bird-ecg-detection --display-name "Python (.venv bird-ecg)"
```

### Ejecutar el Notebook Completo por Terminal

Con el entorno activado:

```bash
source .venv/bin/activate
PTBXL_DATA_DIR="/Users/agustinaperini/Desktop/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3" \
python -m jupyter nbconvert \
  --to notebook \
  --execute notebooks/pipeline_sin_ml.ipynb \
  --inplace \
  --ExecutePreprocessor.kernel_name=bird-ecg-detection \
  --ExecutePreprocessor.timeout=-1
```

El comando ejecuta todas las celdas y guarda las salidas dentro del mismo archivo `.ipynb`.

### Ejecutar Segmentos Puntuales

Para inspeccionar una celda o segmento específico, abrir Jupyter Lab:

```bash
source .venv/bin/activate
PTBXL_DATA_DIR="/Users/agustinaperini/Desktop/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3" \
python -m jupyter lab notebooks/pipeline_sin_ml.ipynb
```

Seleccionar el kernel `Python (.venv bird-ecg)`. Para ejecutar una celda puntual, correr primero las celdas anteriores necesarias con `Run > Run All Above Selected Cell` y luego ejecutar la celda seleccionada con `Shift + Enter`.

### Dependencias Principales

Las dependencias están declaradas en `requirements.txt`. Las más relevantes son:

* `wfdb==4.1.2`: lectura de señales ECG PTB-XL.
* `pandas==2.2.3`: carga y manipulación de metadatos.
* `numpy`, `scipy`: procesamiento numérico y filtrado de señales.
* `matplotlib`, `seaborn`: gráficos y matrices de confusión.
* `scikit-learn`: partición train/test, árbol de decisión y métricas.
* `jupyter`, `nbconvert`, `ipykernel`: ejecución interactiva y por terminal.

### Resultado Esperado

El notebook no escribe archivos `.csv`, modelos ni figuras externas. Genera y guarda las salidas dentro de `notebooks/pipeline_sin_ml.ipynb`: gráficos de señales, detección de QRS, extracción de features, matrices de confusión, reportes de clasificación y el clasificador final basado en reglas `if/else`.
