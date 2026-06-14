"""
clasificador.py
===============
Sistema experto clínico (sin machine learning) para clasificar bloqueos
de rama en señales ECG. Corresponde a la celda 29 del notebook.

El clasificador aplica reglas if/else sobre las 9 features morfológicas
del QRS. Los umbrales fueron derivados del análisis de cuartiles del
dataset PTB-XL (ver Tabla de estadísticos en el notebook).

Uso típico:
    from clasificador import clasificar_dataset, evaluar
    df = clasificar_dataset(df)
    evaluar(df)
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


# Orden canónico de clases para todas las métricas y matrices
CLASES = ['CLBBB', 'CRBBB', 'ILBBB', 'IRBBB', 'NORM']


# =============================================================================
# CELDA 29 — Clasificador de reglas (Sistema Experto v6)
# =============================================================================

def clasificador_reglas_v6(row: pd.Series) -> str:
    """
    Clasificador determinista basado en reglas morfológicas derivadas de los
    cuartiles del dataset PTB-XL. Opera sobre una fila del DataFrame de features.

    Orden de evaluación (de mayor a menor expresión morfológica):
      1. CLBBB  — área muy grande + polaridad muy negativa en V1
      2. ILBBB  — área moderada + polaridad negativa + R alta en I
      3. IRBBB  — múltiples picos positivos (rSR') + polaridad no negativa
      4. CRBBB  — QRS ancho + R reducida en I, o ratio alto en V6/V1
      5. NORM   — ninguna regla anterior cumplida

    Parámetros
    ----------
    row : pd.Series con las 9 features morfológicas del registro.

    Returns
    -------
    str : clase predicha ('CLBBB', 'CRBBB', 'ILBBB', 'IRBBB' o 'NORM').
    """
    area     = row['area_qrs_v1']
    ancho    = row['ancho_qrs_lead_I']
    polar    = row['polaridad_net_v1']
    picos    = row['n_picos_pos_v1']
    r_amp_I  = row['r_amp_lead_I']
    ratio_v1 = row['ratio_rs_v1']
    ratio_v6 = row['ratio_rs_v6']

    # ------------------------------------------------------------------
    # 1. CLBBB — Bloqueo Completo de Rama Izquierda
    # Criterio principal: área enorme (Q1 CLBBB=0.094) + polaridad muy
    # negativa (mediana CLBBB=-0.107).
    # Criterio secundario: área grande + polaridad muy negativa + QRS ancho,
    # captura casos con área entre 0.078 y 0.092.
    # ------------------------------------------------------------------
    if area >= 0.092 and polar < -0.045:
        return 'CLBBB'

    if area >= 0.078 and polar < -0.065 and ancho > 0.160:
        return 'CLBBB'

    # ------------------------------------------------------------------
    # 2. ILBBB — Bloqueo Incompleto de Rama Izquierda
    # Área moderada (>Q3 NORM=0.037) + polaridad negativa (mediana=-0.053)
    # + R alta en I (mediana ILBBB=0.60), para separarlo de CRBBB que
    # tiene R baja en I.
    # ------------------------------------------------------------------
    if area >= 0.042 and polar < -0.032 and r_amp_I > 0.45:
        return 'ILBBB'

    # ------------------------------------------------------------------
    # 3. IRBBB — Bloqueo Incompleto de Rama Derecha
    # Patrón rSR': múltiples picos positivos en V1 (mediana IRBBB=2.0)
    # con polaridad no muy negativa (descarta LBBB).
    # Criterio secundario: ratio R/S moderado + QRS levemente ancho +
    # polaridad cercana a cero + R normal en I.
    # ------------------------------------------------------------------
    if picos >= 1.68 and polar > -0.025:
        return 'IRBBB'

    if ratio_v1 > 0.35 and ancho > 0.120 and polar > -0.012 and r_amp_I > 0.45:
        return 'IRBBB'

    # ------------------------------------------------------------------
    # 4. CRBBB — Bloqueo Completo de Rama Derecha
    # Criterio de ratio_rs_v6: cuando la S en V6 es pequeña (ratio>>500)
    # con QRS ancho y sin componente izquierdo (polar > -0.040, area<0.065).
    # Criterio clásico: QRS muy ancho + R reducida en I (vector derecho
    # se aleja de Lead I). Mediana CRBBB r_amp_I=0.27 vs resto >0.58.
    # Criterio de ratio_v1: R' dominante en V1 con QRS moderadamente ancho.
    # ------------------------------------------------------------------
    if ratio_v6 > 500.0 and ancho > 0.145 and polar > -0.040 and area < 0.065:
        return 'CRBBB'

    if ancho >= 0.165 and r_amp_I < 0.45 and polar > -0.045:
        return 'CRBBB'

    if ratio_v1 > 1.2 and ancho > 0.135 and polar > -0.035:
        return 'CRBBB'

    # ------------------------------------------------------------------
    # 5. NORM — Registro Normal (por defecto)
    # ------------------------------------------------------------------
    return 'NORM'


# =============================================================================
# Función de aplicación y evaluación
# =============================================================================

def clasificar_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica el clasificador a cada fila del DataFrame y agrega la columna
    'prediccion' con el diagnóstico asignado.

    Parámetros
    ----------
    df : DataFrame con las 9 features (salida de features.extraer_todas_las_features).

    Returns
    -------
    El mismo DataFrame con la columna 'prediccion' agregada.
    """
    df = df.copy()
    df['prediccion'] = df.apply(clasificador_reglas_v6, axis=1)
    print(f"✓ Clasificación aplicada a {len(df)} registros.")
    return df


def evaluar(df: pd.DataFrame, col_real: str = 'clase_real', col_pred: str = 'prediccion') -> None:
    """
    Imprime el reporte de clasificación y muestra la matriz de confusión.

    Parámetros
    ----------
    df       : DataFrame con columnas de clase real y predicha.
    col_real : nombre de la columna con las etiquetas reales.
    col_pred : nombre de la columna con las predicciones.
    """
    y_true = df[col_real]
    y_pred = df[col_pred]

    print("=" * 60)
    print("REPORTE DE CLASIFICACIÓN — Sistema Experto Heurístico v6")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=CLASES, zero_division=0))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.grid(False)  # evita que las líneas de Jupyter corten los bloques
    cm   = confusion_matrix(y_true, y_pred, labels=CLASES)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASES)
    disp.plot(ax=ax, colorbar=True, cmap='Blues')
    ax.set_title('Matriz de Confusión — Sistema Experto Morfológico v6',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()
