"""
data_loader.py
==============
Carga del dataset PTB-XL, construcción del diccionario de trabajo y filtrado
de señales ECG. Corresponde a las celdas 2, 3, 4, 5 y 6 del notebook.

Uso típico:
    from data_loader import construir_dataset, filtrar_dataset
    dataset = construir_dataset(PATH_DATA)
    dataset = filtrar_dataset(dataset, PATH_DATA)
"""

import os
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
import scipy.signal as signal
from scipy.signal import butter, filtfilt


# =============================================================================
# CELDA 2 y 3 — Carga y parsing del CSV de metadata
# =============================================================================

def cargar_metadata(path_data: Path) -> pd.DataFrame:
    """
    Lee ptbxl_database.csv y devuelve el DataFrame con:
      - scp_codes convertido a dict de Python
      - columna 'patologia_principal' con el código SCP de mayor score

    Parámetros
    ----------
    path_data : Path
        Ruta raíz del dataset PTB-XL (la carpeta que contiene ptbxl_database.csv).

    Returns
    -------
    pd.DataFrame indexado por 'ecg_id'.
    """
    csv_path = path_data / 'ptbxl_database.csv'
    if not csv_path.exists():
        raise FileNotFoundError(
            f"No se encontró ptbxl_database.csv en {path_data.resolve()}.\n"
            "Descargá el dataset y colocalo en 'data/raw'."
        )

    df_meta = pd.read_csv(csv_path, index_col='ecg_id')
    df_meta['scp_codes'] = df_meta['scp_codes'].apply(ast.literal_eval)
    df_meta['patologia_principal'] = df_meta['scp_codes'].apply(_patologia_principal)
    return df_meta


def _patologia_principal(diccionario_codigos: dict) -> str:
    """Devuelve el código SCP con mayor score estadístico del registro."""
    if not diccionario_codigos:
        return 'UNKNOWN'
    return max(diccionario_codigos, key=diccionario_codigos.get)


# =============================================================================
# CELDA 4 — Construcción del diccionario de trabajo (dataset_proyecto)
# =============================================================================

def construir_dataset(
    path_data: Path,
    clases: list[str] | None = None,
    pacientes_por_clase: int = 100,
    seed: int = 42,
) -> dict:
    """
    Selecciona registros de PTB-XL para las clases indicadas y construye
    el diccionario de trabajo con todos los campos pre-armados.

    Parámetros
    ----------
    path_data : Path
        Ruta raíz del dataset PTB-XL.
    clases : list[str], opcional
        Clases a incluir. Por defecto ['NORM', 'IRBBB', 'CRBBB', 'CLBBB', 'ILBBB'].
    pacientes_por_clase : int
        Número máximo de registros a tomar por clase (se toma el mínimo con
        los disponibles para no romper en clases pequeñas como ILBBB).
    seed : int
        Semilla para reproducibilidad del muestreo aleatorio.

    Returns
    -------
    dict  {clave_paciente: {ecg_id, clase_clinica, file_path_lr/hr, fs_lr/hr,
                             senal_v1_limpia, senal_v6_limpia, senal_I_limpia,
                             indices_ondas_r, descriptores, prediccion}}
    """
    if clases is None:
        clases = ['NORM', 'IRBBB', 'CRBBB', 'CLBBB', 'ILBBB']

    df_meta = cargar_metadata(path_data)
    np.random.seed(seed)
    dataset = {}

    print("Construyendo dataset de trabajo...")
    for clase in clases:
        ids_disponibles = df_meta[df_meta['patologia_principal'] == clase].index.tolist()
        n = min(pacientes_por_clase, len(ids_disponibles))
        ids_sel = np.random.choice(ids_disponibles, n, replace=False)
        print(f"  • {clase:5s}: {n} registros  (disponibles totales: {len(ids_disponibles)})")

        for ecg_id in ids_sel:
            clave = f"paciente_{ecg_id}"
            dataset[clave] = {
                'ecg_id':        int(ecg_id),
                'clase_clinica': clase,
                'file_path_lr':  df_meta.loc[ecg_id, 'filename_lr'],
                'file_path_hr':  df_meta.loc[ecg_id, 'filename_hr'],
                'fs_lr': 100,
                'fs_hr': 500,
                # Señales filtradas (se rellenan en filtrar_dataset)
                'senal_v1_limpia': None,
                'senal_v6_limpia': None,
                'senal_I_limpia':  None,
                # Picos R detectados en V1 (se rellenan en features.py)
                'indices_ondas_r': None,
                # Las 9 features morfológicas (se rellenan en features.py)
                'descriptores': {
                    'area_qrs_v1':      None,
                    'ancho_qrs_lead_I': None,
                    'polaridad_net_v1': None,
                    'n_picos_pos_v1':   None,
                    'sep_r_rprime_v1':  None,
                    'ratio_rs_v1':      None,
                    's_wave_depth_v6':  None,
                    'ratio_rs_v6':      None,
                    'r_amp_lead_I':     None,
                },
                'prediccion': None,
            }

    print(f"\n✓ Dataset listo: {len(dataset)} registros totales.")
    return dataset


# =============================================================================
# CELDA 5 — Función de filtrado Butterworth
# =============================================================================

def filtrar_ecg_butterworth(
    senal_cruda: np.ndarray,
    fs: float = 100.0,
    f_corte_baja: float = 0.5,
    f_corte_alta: float = 40.0,
    orden: int = 4,
) -> np.ndarray:
    """
    Aplica un filtro Butterworth pasabanda de fase cero a una señal ECG.

    Parámetros
    ----------
    senal_cruda   : array de voltaje crudo (mV).
    fs            : frecuencia de muestreo (Hz).
    f_corte_baja  : límite inferior (elimina baseline wander, 0.5 Hz).
    f_corte_alta  : límite superior (elimina ruido muscular/red, 40 Hz).
    orden         : orden del filtro Butterworth (4 por defecto).

    Returns
    -------
    np.ndarray con la señal filtrada.
    """
    nyquist = fs / 2.0
    low  = f_corte_baja / nyquist
    high = f_corte_alta / nyquist
    b, a = signal.butter(orden, [low, high], btype='band')
    return signal.filtfilt(b, a, senal_cruda)


# =============================================================================
# CELDA 6 — Filtrado masivo del dataset
# =============================================================================

def filtrar_dataset(dataset: dict, path_data: Path) -> dict:
    """
    Lee las señales crudas de disco para cada registro del dataset,
    aplica el filtro Butterworth y guarda las señales limpias (V1, V6, I)
    dentro del mismo diccionario.

    Parámetros
    ----------
    dataset   : dict construido por construir_dataset().
    path_data : Path raíz del dataset PTB-XL.

    Returns
    -------
    El mismo dict con 'senal_v1_limpia', 'senal_v6_limpia' y 'senal_I_limpia'
    rellenos para cada paciente.
    """
    print(f"Filtrando señales de {len(dataset)} registros...")
    errores = []

    for clave, datos in dataset.items():
        try:
            ruta = path_data / datos['file_path_lr']
            signals, _ = wfdb.rdsamp(str(ruta))

            fs = datos['fs_lr']
            dataset[clave]['senal_v1_limpia'] = filtrar_ecg_butterworth(signals[:, 6],  fs)
            dataset[clave]['senal_v6_limpia'] = filtrar_ecg_butterworth(signals[:, 11], fs)
            dataset[clave]['senal_I_limpia']  = filtrar_ecg_butterworth(signals[:, 0],  fs)

        except Exception as e:
            errores.append(clave)
            print(f"  ⚠ Error en {clave}: {e}")

    print(f"✓ Filtrado completo. Errores: {len(errores)}")
    return dataset
