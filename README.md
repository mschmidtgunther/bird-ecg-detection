# CardioExpert V6 — Sistema Experto para Clasificación de Bloqueos de Rama en ECG

Sistema experto determinista (sin machine learning) para clasificar automáticamente bloqueos de rama a partir de señales ECG, desarrollado sobre el dataset público PTB-XL. El clasificador aplica reglas morfológicas derivadas del análisis de cuartiles del dataset para asignar uno de cinco diagnósticos: bloqueo completo de rama izquierda (CLBBB), bloqueo completo de rama derecha (CRBBB), bloqueo incompleto de rama izquierda (ILBBB), bloqueo incompleto de rama derecha (IRBBB), o registro normal (NORM).

---

## Descripción del proyecto

El pipeline procesa señales ECG crudas del dataset PTB-XL y extrae 9 descriptores morfológicos del complejo QRS sobre las derivaciones V1, V6 y DI. Estos descriptores alimentan un sistema de reglas if/else (Sistema Experto v6) que produce un diagnóstico clínico por registro. El proyecto incluye una interfaz gráfica en PyQt5 con tres modos de uso: análisis individual, procesamiento por lote y validación del algoritmo con métricas de scikit-learn.

### Descriptores extraídos

El pipeline calcula los siguientes 9 descriptores morfológicos por registro:

| Feature | Derivación | Descripción |
|---|---|---|
| `area_qrs_v1` | V1 | Área absoluta del complejo QRS (energía) |
| `ancho_qrs_lead_I` | DI | Duración del QRS usando umbral del 10 % de amplitud máxima |
| `polaridad_net_v1` | V1 | Integral con signo del QRS (positivo = R dominante, negativo = S/QS dominante) |
| `n_picos_pos_v1` | V1 | Número promedio de picos positivos por latido (patrón rSR') |
| `sep_r_rprime_v1` | V1 | Gap temporal entre R y R' (diferencia CRBBB vs IRBBB) |
| `ratio_rs_v1` | V1 | Cociente amplitud R / amplitud S |
| `s_wave_depth_v6` | V6 | Profundidad de onda S (marcador de bloqueo derecho) |
| `ratio_rs_v6` | V6 | Cociente amplitud R / amplitud S en V6 |
| `r_amp_lead_I` | DI | Amplitud máxima de onda R (baja en CRBBB, alta en LBBB) |

---

## Dataset

El proyecto utiliza **PTB-XL**, una base de datos pública de ECG de 12 derivaciones:

> Los datos **no se incluyen en el repositorio**. Descargalos desde [PhysioNet — PTB-XL](https://physionet.org/content/ptb-xl/) y colocalos según la estructura indicada en la sección siguiente.

---

## Estructura de carpetas

```
project-root/
│
├── data/
│   └── raw/                          ← Raíz del dataset PTB-XL
│       └── ptbxl_database        ← Metadatos del dataset (obligatorio)
│           ├── records100/               ← Señales a 100 Hz (formato WFDB .dat/.hea)
│           └── records500/               ← Señales a 500 Hz (formato WFDB .dat/.hea)
│
├── src/
│   ├── data_loader.py                ← Carga de metadatos, construcción del dataset y filtrado Butterworth
│   ├── features.py                   ← Detección de latidos y extracción de los 9 descriptores QRS
│   └── clasificador.py               ← Sistema experto de reglas morfológicas (v6) y evaluación
│
├── notebooks/                        ← Experimentación en Jupyter (exploración y desarrollo)
│
├── gui.py                            ← Interfaz gráfica PyQt5 (punto de entrada visual)
├── main.py                           ← Pipeline por consola (punto de entrada sin GUI)
└── requirements.txt
```

> **Importante:** La carpeta `data/` no se versiona. Asegurate de crearla manualmente y colocar el dataset antes de ejecutar.

---

## Instalación

Requiere Python 3.10 o superior.

```bash
pip install -r requirements.txt
```

Las dependencias principales son `wfdb`, `scipy`, `numpy`, `pandas`, `scikit-learn`, `matplotlib` y `PyQt5`.

---

## Uso

### Interfaz gráfica (recomendado)

```bash
python gui.py
```

La interfaz tiene tres pestañas:

**Registro individual** — Cargá un archivo `.hea` o `.dat` de PTB-XL. El sistema filtra la señal, detecta latidos, extrae los descriptores y muestra el diagnóstico junto con las trazas de V1, V6 y DI para los primeros 3 segundos.

**Análisis por lote** — Seleccioná la carpeta raíz del dataset (`data/raw`). El sistema procesa 30 registros por clase, clasifica cada uno y muestra la distribución de diagnósticos en un gráfico de barras.

**Evaluar algoritmo** — Seleccioná la carpeta raíz del dataset (`data/raw`). El sistema procesa 80 registros por clase, compara las predicciones con las etiquetas reales de PTB-XL y muestra la matriz de confusión y el reporte de precisión/recall/F1 por clase.

### Pipeline por consola

```bash
python main.py
```

Ejecuta el pipeline completo: carga 100 registros por clase, filtra las señales, extrae features, clasifica e imprime el reporte de métricas en consola. Editá la variable `PATH_DATA` en `main.py` si tu dataset está en una ruta distinta a `data/raw`.

---

## Pipeline técnico

```
ptbxl_database.csv
      │
      ▼
construir_dataset()       Selección aleatoria estratificada por clase (seed reproducible)
      │
      ▼
filtrar_dataset()         Filtro Butterworth pasabanda (0.5–40 Hz, orden 4, fase cero)
      │
      ▼
extraer_todas_las_features()
   ├── detectar_latidos_v1()     find_peaks sobre señal V1 invertida
   └── 9 × calcular_*()         Cálculo de descriptores morfológicos por latido
      │
      ▼
clasificar_dataset()      Sistema experto v6: reglas if/else sobre los 9 descriptores
      │
      ▼
evaluar()                 Reporte sklearn + matriz de confusión
```

---

## Módulos

**`src/data_loader.py`** — Lee `ptbxl_database.csv`, parsea los códigos SCP, selecciona registros por clase mediante muestreo aleatorio estratificado y aplica el filtro Butterworth a las derivaciones V1, V6 y DI de cada registro.

**`src/features.py`** — Detecta los complejos QRS en V1 usando `scipy.signal.find_peaks` sobre la señal invertida y calcula los 9 descriptores morfológicos. Cada función puede generar una gráfica diagnóstica con `plot=True`.

**`src/clasificador.py`** — Implementa `clasificador_reglas_v6()`, que evalúa los 9 descriptores en orden de expresión morfológica: CLBBB primero (área y polaridad más extremas), luego ILBBB, IRBBB, CRBBB y NORM por defecto. Los umbrales fueron derivados del análisis de cuartiles del dataset PTB-XL.

**`gui.py`** — Interfaz PyQt5 con tres pestañas que integra los tres módulos anteriores. Los gráficos se renderizan con `matplotlib` embebido en Qt.
