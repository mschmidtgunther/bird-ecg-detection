"""
gui_ecg.py
==========
Interfaz Gráfica de Usuario (GUI) en PyQt5 para el Sistema Experto de ECG.
Proporciona 3 modalidades operativas:
  1. Análisis Individual: Carga un registro .dat/.hea, grafica señales y extrae características.
  2. Análisis por Lote: Procesa una carpeta completa y muestra un gráfico de distribución.
  3. Validación del Sistema: Ejecuta el pipeline completo de métricas y matriz de confusión.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import wfdb

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                            QMessageBox, QFrame, QScrollArea, QTabWidget, 
                            QTableWidget, QTableWidgetItem, QTextEdit, QHeaderView)
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtCore import Qt

# Integración de Matplotlib en PyQt5
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Importamos tus módulos modulares
import src.data_loader
import src.features
import src.clasificador


# =============================================================================
# CANVAS DE MATPLOTLIB ADAPTADO PARA PYQT5
# =============================================================================
class MplCanvas(FigureCanvas):
    """Canvas interactivo para renderizar gráficos dentro de la interfaz."""
    def __init__(self, parent=None, width=7, height=5, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)


# =============================================================================
# VENTANA PRINCIPAL DEL SISTEMA EXPERTO
# =============================================================================
class VentanaPrincipalECG(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CardioExpert V6 — Sistema Experto de Bloqueos de Rama")
        self.resize(1100, 750)
        
        # Estilo CSS Global (Limpio, moderno, clínico)
        self.setStyleSheet("""
            QMainWindow { background-color: #F4F6F9; }
            QTabWidget::pane { border: 1px solid #D1D5DB; background: white; border-radius: 4px; }
            QTabWidget::tab-bar { alignment: center; }
            QTabBar::tab { background: #E5E7EB; color: #374151; min-width: 200px; padding: 14px 26px; font-weight: bold; font-size: 13px; border-top-left-radius: 4px; border-top-right-radius: 4px; border-bottom: 2px solid transparent; }
            QTabBar::tab:selected { background: white; color: #1E40AF; border-bottom: 2px solid #1E40AF; }
            QPushButton { background-color: #1E40AF; color: white; border-radius: 5px; font-weight: bold; padding: 10px 15px; font-size: 13px; }
            QPushButton:hover { background-color: #1D4ED8; }
            QPushButton:pressed { background-color: #1E3A8A; }
            QLabel { color: #1F2937; }
            QTableWidget { gridline-color: #E5E7EB; border: 1px solid #D1D5DB; }
            QHeaderView::section { background-color: #F3F4F6; font-weight: bold; border: 1px solid #D1D5DB; }
        """)

        # Inicialización del Widget Central y pestañas (Las 3 opciones solicitadas)
        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)

        # Crear las tres vistas/pestañas
        self.init_tab_individual()
        self.init_tab_lote()
        self.init_tab_algoritmo()

    # =========================================================================
    # OPCIÓN 1: ANALIZAR UN REGISTRO INDIVIDUAL (Imagen/Señal)
    # =========================================================================
    def init_tab_individual(self):
        widget = QWidget()
        layout = QHBoxLayout(widget)

        # Panel Izquierdo: Controles y Resultados
        panel_izq = QVBoxLayout()

        lbl_titulo = QLabel("Análisis de Registro Único")
        lbl_titulo.setFont(QFont("Arial", 14, QFont.Bold))
        panel_izq.addWidget(lbl_titulo)

        btn_cargar = QPushButton("📂 Seleccionar Registro WFDB (.dat/.hea)")
        btn_cargar.clicked.connect(self.procesar_registro_individual)
        panel_izq.addWidget(btn_cargar)

        self.lbl_ruta_registro = QLabel("Archivo: No disponible")
        self.lbl_ruta_registro.setWordWrap(True)
        self.lbl_ruta_registro.setStyleSheet("color: #4B5563; font-size: 11px;")
        panel_izq.addWidget(self.lbl_ruta_registro)

        # Tarjeta de Diagnóstico destacado
        self.frame_diagnostico = QFrame()
        self.frame_diagnostico.setObjectName("frameDiagnostico")
        self.frame_diagnostico.setStyleSheet("#frameDiagnostico { background-color: #EFF6FF; border: 2px solid #BFDBFE; border-radius: 8px; }")
        layout_diag = QVBoxLayout(self.frame_diagnostico)
        self.lbl_diag_titulo = QLabel("Diagnóstico final")
        self.lbl_diag_titulo.setFont(QFont("Arial", 11, QFont.Bold))
        self.lbl_diag_titulo.setStyleSheet("background: transparent; border: none; color: #1F2937;")
        self.lbl_diag_resultado = QLabel("Esperando registro...")
        self.lbl_diag_resultado.setFont(QFont("Arial", 18, QFont.Bold))
        self.lbl_diag_resultado.setStyleSheet("background: transparent; border: none; color: #1E40AF;")
        self.lbl_diag_resultado.setAlignment(Qt.AlignCenter)
        self.lbl_interpretacion = QLabel("Interpretación: Cargue un registro para ver el resultado.")
        self.lbl_interpretacion.setWordWrap(True)
        self.lbl_interpretacion.setStyleSheet("background: transparent; border: none; color: #1F2937;")
        layout_diag.addWidget(self.lbl_diag_titulo)
        layout_diag.addWidget(self.lbl_diag_resultado)
        layout_diag.addWidget(self.lbl_interpretacion)
        panel_izq.addWidget(self.frame_diagnostico)

        self.features_principales_gui = [
            ("V1 — área y polaridad", "Área QRS", "area_qrs_v1", 4, "mV·s"),
            ("V1 — área y polaridad", "Polaridad neta", "polaridad_net_v1", 4, "mV·s"),
            ("V6 — relación R/S", "Ratio R/S", "ratio_rs_v6", 2, ""),
            ("DI — ancho QRS", "Ancho QRS", "ancho_qrs_lead_I", 3, "s"),
        ]

        panel_izq.addWidget(QLabel("<b>Features morfológicas principales</b>"))
        lbl_subtitulo_features = QLabel("Variables utilizadas por el sistema experto basado en reglas")
        lbl_subtitulo_features.setStyleSheet("color: #4B5563; font-size: 11px;")
        panel_izq.addWidget(lbl_subtitulo_features)

        self.tabla_features = QTableWidget(len(self.features_principales_gui), 3)
        self.tabla_features.setHorizontalHeaderLabels(["Derivación", "Feature", "Valor"])
        self.tabla_features.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tabla_features.verticalHeader().setVisible(False)
        self.tabla_features.setStyleSheet("""
            QTableWidget { background-color: white; color: #111827; gridline-color: #E5E7EB; border: 1px solid #D1D5DB; }
            QTableWidget::item { background-color: white; color: #111827; }
            QTableWidget::item:selected { background-color: #DBEAFE; color: #111827; }
            QHeaderView::section { background-color: #F3F4F6; color: #111827; border: 1px solid #D1D5DB; }
        """)
        for idx, (derivacion, feature, _, _, _) in enumerate(self.features_principales_gui):
            self.tabla_features.setItem(idx, 0, QTableWidgetItem(derivacion))
            self.tabla_features.setItem(idx, 1, QTableWidgetItem(feature))
            self.tabla_features.setItem(idx, 2, QTableWidgetItem("No disponible"))
        panel_izq.addWidget(self.tabla_features)

        panel_izq.addWidget(QLabel("<b>Explicación de la clasificación:</b>"))
        self.txt_explicacion = QTextEdit()
        self.txt_explicacion.setReadOnly(True)
        self.txt_explicacion.setMinimumHeight(120)
        self.txt_explicacion.setStyleSheet("background-color: white; color: #111827; border: 1px solid #D1D5DB; border-radius: 4px;")
        self.txt_explicacion.setText("Cargue un registro para ver la explicación.")
        panel_izq.addWidget(self.txt_explicacion)

        if hasattr(self, "lbl_accuracy"):
            self.lbl_accuracy.setVisible(False)

        layout.addLayout(panel_izq, stretch=2)

        # Panel Derecho: Gráficas de las Derivaciones Cardíacas
        self.canvas_individual = MplCanvas(self, width=6, height=6)
        layout.addWidget(self.canvas_individual, stretch=3)

        self.tab_widget.addTab(widget, "Registro individual")

    def procesar_registro_individual(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Abrir Registro ECG",
            "",
            "Archivos WFDB (*.hea *.dat);;Todos los archivos (*)"
        )
        if not file_path:
            return

        path_objeto = Path(file_path)
        record_path = path_objeto.with_suffix("") if path_objeto.suffix.lower() in (".hea", ".dat") else path_objeto
        nombre_base = record_path.name
        self.lbl_ruta_registro.setText(f"Archivo: {record_path}")

        try:
            # 1. Cargar la señal cruda
            signals, fields = wfdb.rdsamp(str(record_path))
            fs = fields['fs']

            # 2. Filtrar canales utilizando tu data_loader
            s_v1 = src.data_loader.filtrar_ecg_butterworth(signals[:, 6], fs)
            s_v6 = src.data_loader.filtrar_ecg_butterworth(signals[:, 11], fs)
            s_i  = src.data_loader.filtrar_ecg_butterworth(signals[:, 0], fs)

            # 3. Construir Micro-Dataset para acoplar con tu módulo features.py
            micro_dataset = {
                'PACIENTE_GUI': {
                    'senal_v1_limpia': s_v1,
                    'senal_v6_limpia': s_v6,
                    'senal_I_limpia': s_i,
                    'fs_lr': fs,
                    'clase_clinica': 'DESCONOCIDA',
                    'descriptores': {}
                }
            }

            # 4. Extracción de Features usando tu pipeline exacto
            df_feat = src.features.extraer_todas_las_features(micro_dataset)
            
            if df_feat.empty:
                QMessageBox.warning(self, "Advertencia", "No se detectaron complejos QRS válidos en este registro.")
                return

            # 5. Diagnóstico mediante tu Clasificador v6
            df_clasificado = src.clasificador.clasificar_dataset(df_feat)
            diagnostico = df_clasificado.iloc[0]['prediccion']

            # --- Actualizar Interfaz ---
            self.lbl_diag_resultado.setText(diagnostico)
            interpretaciones = {
                "NORM": "Interpretación: Registro sin patrón morfológico compatible con bloqueo de rama.",
                "IRBBB": "Interpretación: Patrón compatible con bloqueo incompleto de rama derecha.",
                "CRBBB": "Interpretación: Patrón compatible con bloqueo completo de rama derecha.",
                "ILBBB": "Interpretación: Patrón compatible con bloqueo incompleto de rama izquierda.",
                "CLBBB": "Interpretación: Patrón compatible con bloqueo completo de rama izquierda.",
            }
            self.lbl_interpretacion.setText(interpretaciones.get(diagnostico, "Interpretación: No disponible."))
            
            # Rellenar solo las features principales visibles.
            descriptores_dict = micro_dataset['PACIENTE_GUI']['descriptores']
            for idx, (derivacion, feature, llave, decimales, unidad) in enumerate(self.features_principales_gui):
                valor = descriptores_dict.get(llave)
                if valor is None or pd.isna(valor):
                    texto_valor = "No disponible"
                else:
                    texto_valor = f"{float(valor):.{decimales}f}"
                    if unidad:
                        texto_valor += f" {unidad}"

                self.tabla_features.setItem(idx, 0, QTableWidgetItem(derivacion))
                self.tabla_features.setItem(idx, 1, QTableWidgetItem(feature))
                self.tabla_features.setItem(idx, 2, QTableWidgetItem(texto_valor))

            explicaciones = {
                "NORM": "- Ancho QRS dentro de rango compatible con normalidad.\n- Polaridad y área sin patrón marcado de bloqueo completo.\n- No se detecta patrón morfológico suficiente para clasificar como bloqueo.",
                "IRBBB": "- Cambios morfológicos leves en V1.\n- Posible patrón compatible con rSR'.\n- No cumple criterios de bloqueo completo.",
                "CRBBB": "- Alteraciones marcadas compatibles con bloqueo derecho.\n- Cambios en V1 y V6 superan umbrales definidos.\n- Patrón compatible con bloqueo completo de rama derecha.",
                "ILBBB": "- Cambios compatibles con bloqueo izquierdo incompleto.\n- Clase con mayor incertidumbre por menor cantidad de casos.\n- Alteraciones menos marcadas que en bloqueo completo.",
                "CLBBB": "- Alteraciones marcadas compatibles con bloqueo izquierdo.\n- Área, polaridad y ancho QRS compatibles con bloqueo completo.\n- Patrón compatible con bloqueo completo de rama izquierda.",
            }
            self.txt_explicacion.setText(explicaciones.get(diagnostico, "No hay explicación disponible para esta clase."))

            # Graficar Canales Clínicos Clave (V1, V6, I)
            self.canvas_individual.fig.clear()
            axes = self.canvas_individual.fig.subplots(3, 1, sharex=True)
            self.canvas_individual.axes = axes[0]
            t = np.arange(len(s_v1)) / fs
            # Mostramos los primeros 3 segundos para una visualización clínica óptima
            max_samples = int(fs * 3) 
            
            axes[0].plot(t[:max_samples], s_v1[:max_samples], label="V1 — área y polaridad", color="red")
            axes[0].set_ylabel("V1 (mV)")
            axes[0].legend(loc="upper right")

            axes[1].plot(t[:max_samples], s_v6[:max_samples], label="V6 — relación R/S", color="blue")
            axes[1].set_ylabel("V6 (mV)")
            axes[1].legend(loc="upper right")

            axes[2].plot(t[:max_samples], s_i[:max_samples], label="DI — ancho QRS", color="green")
            axes[2].set_ylabel("DI (mV)")
            axes[2].set_xlabel("Tiempo (segundos)")
            axes[2].legend(loc="upper right")

            for ax in axes:
                ax.set_xlim(0, 3)
                ax.grid(True, linestyle='--', alpha=0.3)

            self.canvas_individual.fig.suptitle(f"Trazado Electrocardiográfico — Registro: {nombre_base}")
            self.canvas_individual.fig.tight_layout()
            self.canvas_individual.draw()

        except Exception as e:
            QMessageBox.critical(self, "Error de Procesamiento", f"No se pudo analizar el registro:\\n{str(e)}")


    # =========================================================================
    # OPCIÓN 2: ANALIZAR VARIOS REGISTROS (Procesamiento por Lote)
    # =========================================================================
    def init_tab_lote(self):
        widget = QWidget()
        layout = QHBoxLayout(widget)

        panel_izq = QVBoxLayout()
        lbl_titulo = QLabel("Procesamiento Masivo por Lote")
        lbl_titulo.setFont(QFont("Arial", 14, QFont.Bold))
        panel_izq.addWidget(lbl_titulo)

        btn_seleccionar = QPushButton("📁 Seleccionar Carpeta de Datos PTB-XL")
        btn_seleccionar.clicked.connect(self.procesar_lote_masivo)
        panel_izq.addWidget(btn_seleccionar)

        self.lbl_estado_lote = QLabel("Estado: Esperando asignación de lote.")
        self.lbl_estado_lote.setWordWrap(True)
        panel_izq.addWidget(self.lbl_estado_lote)
        
        panel_izq.addStretch()
        layout.addLayout(panel_izq, stretch=2)

        # Gráfico de barras para ver qué patologías se encontraron en el lote
        self.canvas_lote = MplCanvas(self, width=5, height=5)
        layout.addWidget(self.canvas_lote, stretch=3)

        self.tab_widget.addTab(widget, "Análisis por lote")

    def procesar_lote_masivo(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Seleccionar Carpeta del Dataset")
        if not dir_path:
            return

        path_data = Path(dir_path)
        self.lbl_estado_lote.setText("Procesando lote en segundo plano... Por favor espere.")
        QApplication.processEvents() # Actualiza la UI de inmediato

        try:
            # Reutiliza tus funciones estructuradas de data_loader de forma nativa
            dataset = src.data_loader.construir_dataset(path_data, pacientes_por_clase=30, seed=42)
            dataset = src.data_loader.filtrar_dataset(dataset, path_data)
            
            df_features = src.features.extraer_todas_las_features(dataset)
            df_clasificado = src.clasificador.clasificar_dataset(df_features)

            # Contar la distribución de diagnósticos arrojados por el Sistema Experto
            conteo = df_clasificado['prediccion'].value_counts()
            
            # Graficar la distribución en el Canvas
            self.canvas_lote.axes.clear()
            conteo.plot(kind='bar', ax=self.canvas_lote.axes, color=['#1E40AF', '#3B82F6', '#60A5FA', '#93C5FD', '#C4B5FD'])
            self.canvas_lote.axes.set_title("Distribución de Diagnósticos en el Lote Analizado")
            self.canvas_lote.axes.set_ylabel("Cantidad de Registros")
            self.canvas_lote.axes.set_xlabel("Patología Detectada")
            self.canvas_lote.fig.tight_layout()
            self.canvas_lote.draw()

            self.lbl_estado_lote.setText(f"✓ ¡Éxito! Lote completado de forma íntegra. Se procesaron {len(df_clasificado)} registros válidos.")

        except Exception as e:
            self.lbl_estado_lote.setText("Estado: Error en la última ejecución.")
            QMessageBox.critical(self, "Error de Lote", f"Error ejecutando el pipeline masivo:\\n{str(e)}")


    # =========================================================================
    # OPCIÓN 3: CHEQUEAR EL ALGORITMO (Métricas Globales de Validación)
    # =========================================================================
    def init_tab_algoritmo(self):
        widget = QWidget()
        layout = QHBoxLayout(widget)

        panel_izq = QVBoxLayout()
        lbl_titulo = QLabel("Métricas de Control de Calidad del Algoritmo")
        lbl_titulo.setFont(QFont("Arial", 14, QFont.Bold))
        panel_izq.addWidget(lbl_titulo)

        btn_validar = QPushButton("🚀 Ejecutar Validación de Reglas Clínicas")
        btn_validar.clicked.connect(self.ejecutar_validacion_algoritmo)
        panel_izq.addWidget(btn_validar)

        panel_izq.addWidget(QLabel("<b>Reporte de Clasificación (Scikit-Learn):</b>"))
        self.txt_reporte = QTextEdit()
        self.txt_reporte.setFont(QFont("Arial", 12))
        self.txt_reporte.setStyleSheet("background-color: white; color: #111827; border: 1px solid #D1D5DB; font-size: 12pt;")
        self.txt_reporte.setReadOnly(True)
        panel_izq.addWidget(self.txt_reporte)

        layout.addLayout(panel_izq, stretch=2)

        # Espacio para inyectar la matriz de confusión directamente en el layout
        self.canvas_matriz = MplCanvas(self, width=5, height=5)
        layout.addWidget(self.canvas_matriz, stretch=3)

        self.tab_widget.addTab(widget, "Evaluar algoritmo")

    def ejecutar_validacion_algoritmo(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Seleccionar Carpeta con ptbxl_database.csv")
        if not dir_path:
            return

        path_data = Path(dir_path)
        self.txt_reporte.setText("Corriendo matriz de confusión y cálculo de Precision/Recall...")
        QApplication.processEvents()

        try:
            # Ejecución del pipeline exacto del notebook final utilizando tus .py
            dataset = src.data_loader.construir_dataset(path_data, pacientes_por_clase=80, seed=42)
            dataset = src.data_loader.filtrar_dataset(dataset, path_data)
            df_features = src.features.extraer_todas_las_features(dataset)
            df_clasificado = src.clasificador.clasificar_dataset(df_features)

            # Extraer vectores de evaluación de Scikit-Learn
            y_true = df_clasificado['clase_real']
            y_pred = df_clasificado['prediccion']

            # Generar Reporte de texto
            from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
            reporte = classification_report(y_true, y_pred, target_names=src.clasificador.CLASES, zero_division=0, output_dict=True)
            filas = ""
            for clase in src.clasificador.CLASES:
                datos = reporte[clase]
                filas += (
                    f"<tr><td><b>{clase}</b></td><td>{datos['precision']:.2f}</td>"
                    f"<td>{datos['recall']:.2f}</td><td>{datos['f1-score']:.2f}</td>"
                    f"<td>{int(datos['support'])}</td></tr>"
                )
            for nombre in ["macro avg", "weighted avg"]:
                datos = reporte[nombre]
                filas += (
                    f"<tr><td><b>{nombre}</b></td><td>{datos['precision']:.2f}</td>"
                    f"<td>{datos['recall']:.2f}</td><td>{datos['f1-score']:.2f}</td>"
                    f"<td>{int(datos['support'])}</td></tr>"
                )
            filas += f"<tr><td><b>accuracy</b></td><td colspan='3'>{reporte['accuracy']:.2f}</td><td>{len(y_true)}</td></tr>"
            self.txt_reporte.setHtml(f"""
                <table border="1" cellspacing="0" cellpadding="6" style="width:100%; border-collapse:collapse; font-family:Arial; font-size:12pt; color:#111827;">
                    <tr style="background-color:#F3F4F6;">
                        <th>Clase</th><th>Precision</th><th>Recall</th><th>F1-score</th><th>Support</th>
                    </tr>
                    {filas}
                </table>
            """)

            # Dibujar Matriz de Confusión sobre el Canvas de la interfaz
            self.canvas_matriz.axes.clear()
            self.canvas_matriz.axes.grid(False) # Forzar remoción de líneas blancas
            
            cm = confusion_matrix(y_true, y_pred, labels=src.clasificador.CLASES)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=src.clasificador.CLASES)
            disp.plot(ax=self.canvas_matriz.axes, colorbar=True, cmap='Blues')
            
            self.canvas_matriz.axes.set_title("Matriz de Confusión — Sistema Experto", fontsize=11, fontweight='bold')
            self.canvas_matriz.fig.tight_layout()
            self.canvas_matriz.draw()

        except Exception as e:
            self.txt_reporte.clear()
            QMessageBox.critical(self, "Error de Validación", f"Error al generar las métricas teóricas:\\n{str(e)}")


# =============================================================================
# BLOQUE DE ARRANQUE DE LA APLICACIÓN
# =============================================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    ventana = VentanaPrincipalECG()
    ventana.show()
    sys.exit(app.exec_())
