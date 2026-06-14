"""
features.py
===========
Detección de latidos y extracción de las 9 features morfológicas del QRS.
Corresponde a las celdas 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27 y 28
del notebook.

Cada función de feature:
  - Recibe el diccionario del dataset y la clave del paciente.
  - Puede mostrar una gráfica diagnóstica si plot=True.
  - Guarda el resultado en dataset[clave]['descriptores'][nombre_feature].
  - Devuelve el valor numérico calculado.

Uso típico:
    from features import extraer_todas_las_features
    df = extraer_todas_las_features(dataset)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.signal import find_peaks
from scipy.integrate import trapezoid


# =============================================================================
# CELDA 7 — Detección de latidos (picos R) en V1
# =============================================================================

def detectar_latidos_v1(
    diccionario_dataset: dict,
    clave_paciente: str,
    plot: bool = False,
) -> np.ndarray:
    """
    Detecta los complejos QRS (picos R) en la derivación V1 usando find_peaks.
    La señal se invierte antes de la detección porque en V1 los QRS suelen
    apuntar hacia abajo (deflexión dominante negativa).

    Los índices detectados quedan guardados en
    dataset[clave_paciente]['indices_ondas_r'].

    Parámetros
    ----------
    diccionario_dataset : dict del pipeline.
    clave_paciente      : clave del registro (p.ej. 'paciente_5918').
    plot                : si True, muestra la señal con los picos marcados.

    Returns
    -------
    np.ndarray con los índices de los picos R detectados.
    """
    senal_limpia = diccionario_dataset[clave_paciente]['senal_v1_limpia']
    fs           = diccionario_dataset[clave_paciente]['fs_lr']

    # find_peaks busca máximos; invertimos para convertir valles en picos
    senal_invertida = -senal_limpia
    picos, _ = find_peaks(senal_invertida, height=0.2, distance=int(0.4 * fs))

    diccionario_dataset[clave_paciente]['indices_ondas_r'] = picos

    if plot:
        tiempo = np.arange(len(senal_limpia)) / fs
        plt.figure(figsize=(14, 5))
        plt.plot(tiempo, senal_limpia, color='red', label='Señal Filtrada V1')
        plt.plot(tiempo[picos], senal_limpia[picos], "kx",
                 markersize=12, markeredgewidth=3, label='Latidos (QRS)')
        plt.title(f'Detección Automática de Latidos — {clave_paciente}', fontweight='bold')
        plt.xlabel('Tiempo (s)')
        plt.ylabel('Amplitud (mV)')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.show()

    return picos


# =============================================================================
# CELDA 9 — Feature 1: Área absoluta del QRS en V1
# =============================================================================

def calcular_area_qrs_v1(
    diccionario_dataset: dict,
    clave_paciente: str,
    ventana_izq_ms: int = 80,
    ventana_der_ms: int = 120,
    plot: bool = False,
) -> float:
    """
    Calcula el área absoluta (energía) del QRS en V1 integrando |señal|
    dentro de una ventana de [−80 ms, +120 ms] alrededor de cada pico R.
    Devuelve el promedio sobre todos los latidos del registro.

    Guarda el resultado en dataset[clave]['descriptores']['area_qrs_v1'].
    """
    senal   = diccionario_dataset[clave_paciente]['senal_v1_limpia']
    fs      = diccionario_dataset[clave_paciente]['fs_lr']
    picos_r = diccionario_dataset[clave_paciente]['indices_ondas_r']

    if picos_r is None or len(picos_r) == 0:
        diccionario_dataset[clave_paciente]['descriptores']['area_qrs_v1'] = 0.0
        return 0.0

    muestras_izq = int((ventana_izq_ms / 1000) * fs)
    muestras_der = int((ventana_der_ms / 1000) * fs)
    areas = []

    if plot:
        tiempo = np.arange(len(senal)) / fs
        plt.figure(figsize=(14, 5))
        plt.plot(tiempo, senal, color='#2c3e50', lw=1.5, label='Señal V1')
        plt.title(f'Feature 1: Área Absoluta del QRS — {clave_paciente}',
                  fontsize=12, fontweight='bold')
        plt.xlabel('Tiempo (s)')
        plt.ylabel('Amplitud (mV)')

    for r_idx in picos_r:
        inicio = max(0, r_idx - muestras_izq)
        fin    = min(len(senal), r_idx + muestras_der)
        seg    = senal[inicio:fin]
        areas.append(trapezoid(np.abs(seg), dx=1 / fs))

        if plot:
            t_seg = tiempo[inicio:fin]
            plt.fill_between(t_seg, 0, seg, color='#e74c3c', alpha=0.5)
            plt.plot(tiempo[r_idx], senal[r_idx], "kx", markersize=8, markeredgewidth=2)

    area_media = float(np.mean(areas))
    diccionario_dataset[clave_paciente]['descriptores']['area_qrs_v1'] = area_media

    if plot:
        plt.legend(handles=[
            plt.Line2D([0], [0], color='#2c3e50', lw=1.5, label='Señal V1'),
            plt.Line2D([0], [0], marker='x', color='w', markeredgecolor='k',
                       markersize=8, markeredgewidth=2, label='Pico R'),
            Patch(facecolor='#e74c3c', alpha=0.5, label='Área Integrada'),
        ], loc='upper right')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.show()

    return area_media


# =============================================================================
# CELDA 11 — Feature 2: Ancho del QRS en derivación I
# =============================================================================

def calcular_ancho_qrs_lead_I(
    diccionario_dataset: dict,
    clave_paciente: str,
    ventana_izq_ms: int = 100,
    ventana_der_ms: int = 150,
    plot: bool = False,
) -> float:
    """
    Calcula la duración (en segundos) del QRS en derivación I usando un
    umbral del 10 % de la amplitud máxima absoluta del segmento para
    delimitar los bordes del complejo.

    Guarda en dataset[clave]['descriptores']['ancho_qrs_lead_I'].
    """
    senal_I = diccionario_dataset[clave_paciente]['senal_I_limpia']
    fs      = diccionario_dataset[clave_paciente]['fs_lr']
    picos_r = diccionario_dataset[clave_paciente]['indices_ondas_r']

    if picos_r is None or len(picos_r) == 0:
        diccionario_dataset[clave_paciente]['descriptores']['ancho_qrs_lead_I'] = 0.0
        return 0.0

    muestras_izq = int((ventana_izq_ms / 1000) * fs)
    muestras_der = int((ventana_der_ms / 1000) * fs)
    anchos = []

    if plot:
        tiempo = np.arange(len(senal_I)) / fs
        plt.figure(figsize=(14, 5))
        plt.plot(tiempo, senal_I, color='#2980b9', lw=1.5, label='Señal Lead I')
        plt.title(f'Feature 2: Ancho del QRS en Lead I — {clave_paciente}',
                  fontsize=12, fontweight='bold')
        plt.xlabel('Tiempo (s)')
        plt.ylabel('Amplitud (mV)')

    for r_idx in picos_r:
        inicio = max(0, r_idx - muestras_izq)
        fin    = min(len(senal_I), r_idx + muestras_der)
        seg    = senal_I[inicio:fin]

        if len(seg) == 0:
            continue

        umbral = 0.10 * np.max(np.abs(seg))
        idx_activos = np.where(np.abs(seg) >= umbral)[0]

        if len(idx_activos) > 0:
            ancho_s = (idx_activos[-1] - idx_activos[0]) / fs
            anchos.append(ancho_s)

            if plot:
                t_ini = tiempo[inicio + idx_activos[0]]
                t_fin = tiempo[inicio + idx_activos[-1]]
                plt.axvspan(t_ini, t_fin, color='#f1c40f', alpha=0.3)
                plt.axvline(t_ini, color='#d35400', linestyle='--', alpha=0.7)
                plt.axvline(t_fin, color='#d35400', linestyle='--', alpha=0.7)

    ancho_medio = float(np.mean(anchos)) if anchos else 0.0
    diccionario_dataset[clave_paciente]['descriptores']['ancho_qrs_lead_I'] = ancho_medio

    if plot:
        plt.legend(handles=[
            plt.Line2D([0], [0], color='#2980b9', lw=1.5, label='Señal Lead I'),
            plt.Line2D([0], [0], color='#d35400', linestyle='--', label='Límites QRS (10%)'),
            Patch(facecolor='#f1c40f', alpha=0.3,
                  label=f'Ancho Prom.: {ancho_medio:.3f} s'),
        ], loc='upper right')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.show()

    return ancho_medio


# =============================================================================
# CELDA 13 — Feature 3: Polaridad neta en V1
# =============================================================================

def calcular_polaridad_net_v1(
    diccionario_dataset: dict,
    clave_paciente: str,
    ventana_izq_ms: int = 80,
    ventana_der_ms: int = 120,
    plot: bool = False,
) -> float:
    """
    Integra la señal V1 original (sin valor absoluto) dentro de la ventana QRS.
    Valor positivo → R dominante (RBBB).
    Valor negativo → S/QS dominante (LBBB).

    Guarda en dataset[clave]['descriptores']['polaridad_net_v1'].
    """
    senal   = diccionario_dataset[clave_paciente]['senal_v1_limpia']
    fs      = diccionario_dataset[clave_paciente]['fs_lr']
    picos_r = diccionario_dataset[clave_paciente]['indices_ondas_r']

    if picos_r is None or len(picos_r) == 0:
        diccionario_dataset[clave_paciente]['descriptores']['polaridad_net_v1'] = 0.0
        return 0.0

    muestras_izq = int((ventana_izq_ms / 1000) * fs)
    muestras_der = int((ventana_der_ms / 1000) * fs)
    polaridades  = []

    if plot:
        tiempo = np.arange(len(senal)) / fs
        plt.figure(figsize=(14, 5))
        plt.plot(tiempo, senal, color='#2c3e50', lw=1.5, label='Señal V1')
        plt.axhline(0, color='black', lw=1)
        plt.title(f'Feature 3: Polaridad Neta en V1 — {clave_paciente}',
                  fontsize=12, fontweight='bold')
        plt.xlabel('Tiempo (s)')
        plt.ylabel('Amplitud (mV)')

    for r_idx in picos_r:
        inicio = max(0, r_idx - muestras_izq)
        fin    = min(len(senal), r_idx + muestras_der)
        seg    = senal[inicio:fin]
        polaridades.append(trapezoid(seg, dx=1 / fs))

        if plot:
            t_seg = tiempo[inicio:fin]
            plt.fill_between(t_seg, 0, seg, where=(seg >= 0),
                             color='#2ecc71', alpha=0.5, interpolate=True)
            plt.fill_between(t_seg, 0, seg, where=(seg < 0),
                             color='#e74c3c', alpha=0.5, interpolate=True)
            plt.axvline(tiempo[r_idx], color='k', linestyle='--', alpha=0.3)

    pol_media = float(np.mean(polaridades))
    diccionario_dataset[clave_paciente]['descriptores']['polaridad_net_v1'] = pol_media

    if plot:
        plt.legend(handles=[
            plt.Line2D([0], [0], color='#2c3e50', lw=1.5, label='Señal V1'),
            Patch(facecolor='#2ecc71', alpha=0.5, label='Suma Positiva'),
            Patch(facecolor='#e74c3c', alpha=0.5, label='Resta Negativa'),
            Patch(facecolor='none', edgecolor='none',
                  label=f'Polaridad Neta: {pol_media:.4f}'),
        ], loc='upper right')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.show()

    return pol_media


# =============================================================================
# CELDA 15 — Feature 4: Número de picos positivos en V1 (patrón rSR')
# =============================================================================

def calcular_n_picos_pos_v1(
    diccionario_dataset: dict,
    clave_paciente: str,
    ventana_izq_ms: int = 80,
    ventana_der_ms: int = 120,
    plot: bool = False,
) -> float:
    """
    Cuenta los picos positivos dentro de la ventana QRS de V1.
    Un promedio ≥ 1.7 suele indicar el patrón rSR' ('orejas de conejo')
    típico del IRBBB/CRBBB.

    Guarda en dataset[clave]['descriptores']['n_picos_pos_v1'].
    """
    senal   = diccionario_dataset[clave_paciente]['senal_v1_limpia']
    fs      = diccionario_dataset[clave_paciente]['fs_lr']
    picos_r = diccionario_dataset[clave_paciente]['indices_ondas_r']

    if picos_r is None or len(picos_r) == 0:
        diccionario_dataset[clave_paciente]['descriptores']['n_picos_pos_v1'] = 0.0
        return 0.0

    muestras_izq = int((ventana_izq_ms / 1000) * fs)
    muestras_der = int((ventana_der_ms / 1000) * fs)
    conteos = []

    if plot:
        tiempo = np.arange(len(senal)) / fs
        plt.figure(figsize=(14, 5))
        plt.plot(tiempo, senal, color='#8e44ad', lw=1.5, label='Señal V1')
        plt.axhline(0, color='black', lw=1, alpha=0.5)
        plt.title(f"Feature 4: Picos Positivos (rSR') en V1 — {clave_paciente}",
                  fontsize=12, fontweight='bold')
        plt.xlabel('Tiempo (s)')
        plt.ylabel('Amplitud (mV)')

    for r_idx in picos_r:
        inicio = max(0, r_idx - muestras_izq)
        fin    = min(len(senal), r_idx + muestras_der)
        seg    = senal[inicio:fin]

        picos_loc, _ = find_peaks(seg, height=0.02, prominence=0.02)
        conteos.append(len(picos_loc))

        if plot:
            t_seg = tiempo[inicio:fin]
            plt.axvspan(tiempo[inicio], tiempo[fin - 1], color='#bdc3c7', alpha=0.2)
            if len(picos_loc) > 0:
                plt.plot(t_seg[picos_loc], seg[picos_loc], "o",
                         color='#f39c12', markersize=8,
                         markeredgecolor='k', markeredgewidth=1.5)

    n_medio = float(np.mean(conteos))
    diccionario_dataset[clave_paciente]['descriptores']['n_picos_pos_v1'] = n_medio

    if plot:
        plt.legend(handles=[
            plt.Line2D([0], [0], color='#8e44ad', lw=1.5, label='Señal V1'),
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor='#f39c12', markeredgecolor='k',
                       markersize=8, label='Picos Positivos'),
            Patch(facecolor='#bdc3c7', alpha=0.2, label='Ventana QRS'),
            Patch(facecolor='none', edgecolor='none',
                  label=f'Promedio: {n_medio:.2f}'),
        ], loc='upper right')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.show()

    return n_medio


# =============================================================================
# CELDA 17 — Feature 5: Separación temporal R–R' en V1
# =============================================================================

def calcular_sep_r_rprime_v1(
    diccionario_dataset: dict,
    clave_paciente: str,
    ventana_izq_ms: int = 80,
    ventana_der_ms: int = 120,
    plot: bool = False,
) -> float:
    """
    Mide el gap temporal (en segundos) entre el primer y el segundo pico
    positivo del QRS en V1. Si hay menos de 2 picos devuelve 0.0.
    Esta separación diferencia CRBBB (gap amplio ~0.04 s) de IRBBB (gap chico).

    Guarda en dataset[clave]['descriptores']['sep_r_rprime_v1'].
    """
    senal   = diccionario_dataset[clave_paciente]['senal_v1_limpia']
    fs      = diccionario_dataset[clave_paciente]['fs_lr']
    picos_r = diccionario_dataset[clave_paciente]['indices_ondas_r']

    if picos_r is None or len(picos_r) == 0:
        diccionario_dataset[clave_paciente]['descriptores']['sep_r_rprime_v1'] = 0.0
        return 0.0

    muestras_izq = int((ventana_izq_ms / 1000) * fs)
    muestras_der = int((ventana_der_ms / 1000) * fs)
    separaciones = []

    if plot:
        tiempo = np.arange(len(senal)) / fs
        plt.figure(figsize=(14, 5))
        plt.plot(tiempo, senal, color='#16a085', lw=1.5, label='Señal V1')
        plt.title(f"Feature 5: Separación R–R' en V1 — {clave_paciente}",
                  fontsize=12, fontweight='bold')
        plt.xlabel('Tiempo (s)')
        plt.ylabel('Amplitud (mV)')

    for r_idx in picos_r:
        inicio = max(0, r_idx - muestras_izq)
        fin    = min(len(senal), r_idx + muestras_der)
        seg    = senal[inicio:fin]

        picos_loc, _ = find_peaks(seg, height=0.02, prominence=0.02)

        if len(picos_loc) >= 2:
            gap_s = (picos_loc[1] - picos_loc[0]) / fs
            separaciones.append(gap_s)

            if plot:
                t_seg  = tiempo[inicio:fin]
                t_p1   = t_seg[picos_loc[0]]
                t_p2   = t_seg[picos_loc[1]]
                y_mid  = (seg[picos_loc[0]] + seg[picos_loc[1]]) / 2
                plt.plot([t_p1, t_p2], [seg[picos_loc[0]], seg[picos_loc[1]]],
                         "o", color='#f39c12', markersize=8)
                if len(separaciones) == 1:
                    plt.plot([t_p1, t_p2], [y_mid, y_mid],
                             color='#c0392b', lw=3, label="Gap R–R'")
        else:
            separaciones.append(0.0)

    sep_media = float(np.mean(separaciones)) if separaciones else 0.0
    diccionario_dataset[clave_paciente]['descriptores']['sep_r_rprime_v1'] = sep_media

    if plot:
        plt.legend(handles=[
            plt.Line2D([0], [0], color='#16a085', lw=1.5, label='Señal V1'),
            plt.Line2D([0], [0], color='#c0392b', lw=3, label="Medición Gap R–R'"),
            Patch(facecolor='none', edgecolor='none',
                  label=f'Sep. Promedio: {sep_media:.4f} s'),
        ], loc='upper right')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.show()

    return sep_media


# =============================================================================
# CELDA 19 — Feature 6: Ratio R/S en V1
# =============================================================================

def calcular_ratio_rs_v1(
    diccionario_dataset: dict,
    clave_paciente: str,
    ventana_izq_ms: int = 80,
    ventana_der_ms: int = 120,
    plot: bool = False,
) -> float:
    """
    Calcula amp_R / |amp_S| en V1. Ratio > 1 → R dominante (RBBB).
    Se suma 1e-6 al denominador para evitar división por cero.

    Guarda en dataset[clave]['descriptores']['ratio_rs_v1'].
    """
    senal   = diccionario_dataset[clave_paciente]['senal_v1_limpia']
    fs      = diccionario_dataset[clave_paciente]['fs_lr']
    picos_r = diccionario_dataset[clave_paciente]['indices_ondas_r']

    if picos_r is None or len(picos_r) == 0:
        diccionario_dataset[clave_paciente]['descriptores']['ratio_rs_v1'] = 0.0
        return 0.0

    muestras_izq = int((ventana_izq_ms / 1000) * fs)
    muestras_der = int((ventana_der_ms / 1000) * fs)
    ratios = []

    if plot:
        tiempo = np.arange(len(senal)) / fs
        plt.figure(figsize=(14, 5))
        plt.plot(tiempo, senal, color='#34495e', lw=1.5, label='Señal V1')
        plt.axhline(0, color='black', lw=1, alpha=0.5)
        plt.title(f'Feature 6: Ratio R/S en V1 — {clave_paciente}',
                  fontsize=12, fontweight='bold')
        plt.xlabel('Tiempo (s)')
        plt.ylabel('Amplitud (mV)')

    for r_idx in picos_r:
        inicio = max(0, r_idx - muestras_izq)
        fin    = min(len(senal), r_idx + muestras_der)
        seg    = senal[inicio:fin]

        amp_r = max(np.max(seg), 0.0)
        amp_s = abs(min(np.min(seg), 0.0))
        ratios.append(amp_r / (amp_s + 1e-6))

        if plot:
            idx_max = inicio + np.argmax(seg)
            idx_min = inicio + np.argmin(seg)
            if amp_r > 0:
                plt.plot(tiempo[idx_max], np.max(seg), "^",
                         color='#2ecc71', markersize=8, markeredgecolor='k')
            if amp_s > 0:
                plt.plot(tiempo[idx_min], np.min(seg), "v",
                         color='#e74c3c', markersize=8, markeredgecolor='k')

    ratio_medio = float(np.mean(ratios))
    diccionario_dataset[clave_paciente]['descriptores']['ratio_rs_v1'] = ratio_medio

    if plot:
        plt.legend(handles=[
            plt.Line2D([0], [0], color='#34495e', lw=1.5, label='Señal V1'),
            plt.Line2D([0], [0], marker='^', color='w',
                       markerfacecolor='#2ecc71', markeredgecolor='k',
                       markersize=8, label='Onda R'),
            plt.Line2D([0], [0], marker='v', color='w',
                       markerfacecolor='#e74c3c', markeredgecolor='k',
                       markersize=8, label='Onda S'),
            Patch(facecolor='none', edgecolor='none',
                  label=f'Ratio Prom.: {ratio_medio:.3f}'),
        ], loc='upper right')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.show()

    return ratio_medio


# =============================================================================
# CELDA 21 — Feature 7: Profundidad de onda S en V6
# =============================================================================

def calcular_s_wave_depth_v6(
    diccionario_dataset: dict,
    clave_paciente: str,
    ventana_izq_ms: int = 80,
    ventana_der_ms: int = 120,
    plot: bool = False,
) -> float:
    """
    Mide |pico más negativo| del QRS en V6. Una S profunda en V6 es
    marcador de bloqueo de rama derecha.

    Guarda en dataset[clave]['descriptores']['s_wave_depth_v6'].
    """
    senal   = diccionario_dataset[clave_paciente]['senal_v6_limpia']
    fs      = diccionario_dataset[clave_paciente]['fs_lr']
    picos_r = diccionario_dataset[clave_paciente]['indices_ondas_r']

    if picos_r is None or len(picos_r) == 0:
        diccionario_dataset[clave_paciente]['descriptores']['s_wave_depth_v6'] = 0.0
        return 0.0

    muestras_izq = int((ventana_izq_ms / 1000) * fs)
    muestras_der = int((ventana_der_ms / 1000) * fs)
    profundidades = []

    if plot:
        tiempo = np.arange(len(senal)) / fs
        plt.figure(figsize=(14, 5))
        plt.plot(tiempo, senal, color='#27ae60', lw=1.5, label='Señal V6')
        plt.axhline(0, color='black', lw=1, alpha=0.5)
        plt.title(f'Feature 7: Profundidad Onda S en V6 — {clave_paciente}',
                  fontsize=12, fontweight='bold')
        plt.xlabel('Tiempo (s)')
        plt.ylabel('Amplitud (mV)')

    for r_idx in picos_r:
        inicio = max(0, r_idx - muestras_izq)
        fin    = min(len(senal), r_idx + muestras_der)
        seg    = senal[inicio:fin]

        min_val = np.min(seg)
        prof    = abs(min_val) if min_val < 0 else 0.0
        profundidades.append(prof)

        if plot and min_val < 0:
            idx_min = inicio + np.argmin(seg)
            plt.plot(tiempo[idx_min], min_val, "v",
                     color='#c0392b', markersize=8, markeredgecolor='k')
            plt.vlines(tiempo[idx_min], ymin=min_val, ymax=0,
                       color='#e74c3c', linestyle='--', lw=2)

    prof_media = float(np.mean(profundidades)) if profundidades else 0.0
    diccionario_dataset[clave_paciente]['descriptores']['s_wave_depth_v6'] = prof_media

    if plot:
        plt.legend(handles=[
            plt.Line2D([0], [0], color='#27ae60', lw=1.5, label='Señal V6'),
            plt.Line2D([0], [0], marker='v', color='w',
                       markerfacecolor='#c0392b', markeredgecolor='k',
                       markersize=8, label='Fondo Onda S'),
            plt.Line2D([0], [0], color='#e74c3c', linestyle='--', lw=2,
                       label='Profundidad'),
            Patch(facecolor='none', edgecolor='none',
                  label=f'Profundidad Prom.: {prof_media:.3f} mV'),
        ], loc='upper right')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.show()

    return prof_media


# =============================================================================
# CELDA 23 — Feature 8: Ratio R/S en V6
# =============================================================================

def calcular_ratio_rs_v6(
    diccionario_dataset: dict,
    clave_paciente: str,
    ventana_izq_ms: int = 80,
    ventana_der_ms: int = 120,
    plot: bool = False,
) -> float:
    """
    Calcula amp_R / |amp_S| en V6. En NORM el ratio es muy alto (>10)
    porque la S es ínfima. En CRBBB la S crece y el ratio baja.
    NOTA: por divisiones con valores casi nulos los outliers son grandes;
    usarlo como señal binaria (>500 = sin S significativa) es más robusto.

    Guarda en dataset[clave]['descriptores']['ratio_rs_v6'].
    """
    senal   = diccionario_dataset[clave_paciente]['senal_v6_limpia']
    fs      = diccionario_dataset[clave_paciente]['fs_lr']
    picos_r = diccionario_dataset[clave_paciente]['indices_ondas_r']

    if picos_r is None or len(picos_r) == 0:
        diccionario_dataset[clave_paciente]['descriptores']['ratio_rs_v6'] = 0.0
        return 0.0

    muestras_izq = int((ventana_izq_ms / 1000) * fs)
    muestras_der = int((ventana_der_ms / 1000) * fs)
    ratios = []

    if plot:
        tiempo = np.arange(len(senal)) / fs
        plt.figure(figsize=(14, 5))
        plt.plot(tiempo, senal, color='#2980b9', lw=1.5, label='Señal V6')
        plt.axhline(0, color='black', lw=1, alpha=0.5)
        plt.title(f'Feature 8: Ratio R/S en V6 — {clave_paciente}',
                  fontsize=12, fontweight='bold')
        plt.xlabel('Tiempo (s)')
        plt.ylabel('Amplitud (mV)')

    for r_idx in picos_r:
        inicio = max(0, r_idx - muestras_izq)
        fin    = min(len(senal), r_idx + muestras_der)
        seg    = senal[inicio:fin]

        amp_r = max(np.max(seg), 0.0)
        amp_s = abs(min(np.min(seg), 0.0))
        ratios.append(amp_r / (amp_s + 1e-6))

        if plot:
            idx_max = inicio + np.argmax(seg)
            idx_min = inicio + np.argmin(seg)
            if amp_r > 0:
                plt.plot(tiempo[idx_max], np.max(seg), "^",
                         color='#2ecc71', markersize=8, markeredgecolor='k')
            if amp_s > 0:
                plt.plot(tiempo[idx_min], np.min(seg), "v",
                         color='#e74c3c', markersize=8, markeredgecolor='k')

    ratio_medio = float(np.mean(ratios)) if ratios else 0.0
    diccionario_dataset[clave_paciente]['descriptores']['ratio_rs_v6'] = ratio_medio

    if plot:
        plt.legend(handles=[
            plt.Line2D([0], [0], color='#2980b9', lw=1.5, label='Señal V6'),
            plt.Line2D([0], [0], marker='^', color='w',
                       markerfacecolor='#2ecc71', markeredgecolor='k',
                       markersize=8, label='Onda R'),
            plt.Line2D([0], [0], marker='v', color='w',
                       markerfacecolor='#e74c3c', markeredgecolor='k',
                       markersize=8, label='Onda S'),
            Patch(facecolor='none', edgecolor='none',
                  label=f'Ratio Prom.: {ratio_medio:.2f}'),
        ], loc='upper right')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.show()

    return ratio_medio


# =============================================================================
# CELDA 25 — Feature 9: Amplitud de onda R en derivación I
# =============================================================================

def calcular_r_amp_lead_I(
    diccionario_dataset: dict,
    clave_paciente: str,
    ventana_izq_ms: int = 80,
    ventana_der_ms: int = 120,
    plot: bool = False,
) -> float:
    """
    Mide el valor máximo positivo dentro del QRS en derivación I.
    Una R alta y ancha en Lead I es marcador de LBBB; una R pequeña
    (<0.45 mV) junto con QRS ancho orienta hacia CRBBB.

    Guarda en dataset[clave]['descriptores']['r_amp_lead_I'].
    """
    senal   = diccionario_dataset[clave_paciente]['senal_I_limpia']
    fs      = diccionario_dataset[clave_paciente]['fs_lr']
    picos_r = diccionario_dataset[clave_paciente]['indices_ondas_r']

    if picos_r is None or len(picos_r) == 0:
        diccionario_dataset[clave_paciente]['descriptores']['r_amp_lead_I'] = 0.0
        return 0.0

    muestras_izq = int((ventana_izq_ms / 1000) * fs)
    muestras_der = int((ventana_der_ms / 1000) * fs)
    amplitudes   = []

    if plot:
        tiempo = np.arange(len(senal)) / fs
        plt.figure(figsize=(14, 5))
        plt.plot(tiempo, senal, color='#8e44ad', lw=1.5, label='Señal Lead I')
        plt.axhline(0, color='black', lw=1, alpha=0.5)
        plt.title(f'Feature 9: Amplitud Onda R en Lead I — {clave_paciente}',
                  fontsize=12, fontweight='bold')
        plt.xlabel('Tiempo (s)')
        plt.ylabel('Amplitud (mV)')

    for r_idx in picos_r:
        inicio = max(0, r_idx - muestras_izq)
        fin    = min(len(senal), r_idx + muestras_der)
        seg    = senal[inicio:fin]

        max_val = np.max(seg)
        amp_r   = max_val if max_val > 0 else 0.0
        amplitudes.append(amp_r)

        if plot and amp_r > 0:
            idx_max = inicio + np.argmax(seg)
            plt.plot(tiempo[idx_max], max_val, "^",
                     color='#f1c40f', markersize=8, markeredgecolor='k')
            plt.vlines(tiempo[idx_max], ymin=0, ymax=max_val,
                       color='#f39c12', linestyle='--', lw=2)

    amp_media = float(np.mean(amplitudes)) if amplitudes else 0.0
    diccionario_dataset[clave_paciente]['descriptores']['r_amp_lead_I'] = amp_media

    if plot:
        plt.legend(handles=[
            plt.Line2D([0], [0], color='#8e44ad', lw=1.5, label='Señal Lead I'),
            plt.Line2D([0], [0], marker='^', color='w',
                       markerfacecolor='#f1c40f', markeredgecolor='k',
                       markersize=8, label='Pico Onda R'),
            plt.Line2D([0], [0], color='#f39c12', linestyle='--', lw=2,
                       label='Amplitud medida'),
            Patch(facecolor='none', edgecolor='none',
                  label=f'Amplitud Prom.: {amp_media:.3f} mV'),
        ], loc='upper right')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.show()

    return amp_media


# =============================================================================
# CELDAS 27/28 — Extracción masiva + construcción del DataFrame
# =============================================================================

FEATURES_ORDEN = [
    'area_qrs_v1', 'ancho_qrs_lead_I', 'polaridad_net_v1',
    'n_picos_pos_v1', 'sep_r_rprime_v1', 'ratio_rs_v1',
    's_wave_depth_v6', 'ratio_rs_v6', 'r_amp_lead_I',
]

_CALCULADORAS = [
    calcular_area_qrs_v1,
    calcular_ancho_qrs_lead_I,
    calcular_polaridad_net_v1,
    calcular_n_picos_pos_v1,
    calcular_sep_r_rprime_v1,
    calcular_ratio_rs_v1,
    calcular_s_wave_depth_v6,
    calcular_ratio_rs_v6,
    calcular_r_amp_lead_I,
]


def extraer_todas_las_features(dataset: dict) -> pd.DataFrame:
    """
    Recorre todo el dataset, detecta latidos y calcula las 9 features para
    cada registro. Devuelve un DataFrame listo para el clasificador.

    Los registros sin latidos detectados reciben 0.0 en todas las features
    y son excluidos del DataFrame resultante.

    Parámetros
    ----------
    dataset : dict construido por data_loader.construir_dataset() y
              pasado por data_loader.filtrar_dataset().

    Returns
    -------
    pd.DataFrame con columnas: paciente, clase_real, + las 9 features.
    """
    print(f"Iniciando extracción masiva para {len(dataset)} registros...")
    sin_latidos = []

    for clave in dataset:
        try:
            picos = detectar_latidos_v1(dataset, clave, plot=False)

            if len(picos) > 0:
                for fn in _CALCULADORAS:
                    fn(dataset, clave, plot=False)
            else:
                sin_latidos.append(clave)
                for feat in FEATURES_ORDEN:
                    dataset[clave]['descriptores'][feat] = 0.0

        except Exception as e:
            print(f"  ⚠ Error en {clave}: {e}")
            sin_latidos.append(clave)

    print(f"✓ Extracción completa. Registros sin latidos: {len(sin_latidos)}")

    # Construir DataFrame excluyendo registros sin latidos
    filas = []
    for clave, datos in dataset.items():
        if clave not in sin_latidos:
            fila = datos['descriptores'].copy()
            fila['paciente']  = clave
            fila['clase_real'] = datos['clase_clinica']
            filas.append(fila)

    df = pd.DataFrame(filas)
    print(f"  DataFrame final: {len(df)} registros × {len(df.columns)} columnas.")
    return df
