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