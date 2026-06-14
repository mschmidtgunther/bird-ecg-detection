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
            QTabBar::tab { background: #E5E7EB; color: #374151; padding: 12px 20px; font-weight: bold; border-top-left-radius: 4px; border-top-right-radius: 4px; }
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

        # Tarjeta de Diagnóstico destacado
        self.frame_diagnostico = QFrame()
        self.frame_diagnostico.setStyleSheet("background-color: #EFF6FF; border: 2px solid #BFDBFE; border-radius: 8px;")
        layout_diag = QVBoxLayout(self.frame_diagnostico)
        self.lbl_diag_titulo = QLabel("DIAGNÓSTICO FINAL:")
        self.lbl_diag_titulo.setFont(QFont("Arial", 11, QFont.Bold))
        self.lbl_diag_resultado = QLabel("Esperando registro...")
        self.lbl_diag_resultado.setFont(QFont("Arial", 18, QFont.Bold))
        self.lbl_diag_resultado.setStyleSheet("color: #1E40AF;")
        self.lbl_diag_resultado.setAlignment(Qt.AlignCenter)
        layout_diag.addWidget(self.lbl_diag_titulo)
        layout_diag.addWidget(self.lbl_diag_resultado)
        panel_izq.addWidget(self.frame_diagnostico)

        # Tabla de Descriptores Extraídos
        panel_izq.addWidget(QLabel("<b>Descriptores Morfológicos Extraídos (QRS):</b>"))
        self.tabla_features = QTableWidget(9, 2)
        self.tabla_features.setHorizontalHeaderLabels(["Descriptor", "Valor"])
        self.tabla_features.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        panel_izq.addWidget(self.tabla_features)

        layout.addLayout(panel_izq, stretch=2)

        # Panel Derecho: Gráficas de las Derivaciones Cardíacas
        self.canvas_individual = MplCanvas(self, width=6, height=6)
        layout.addWidget(self.canvas_individual, stretch=3)

        self.tab_widget.addTab(widget, "⚡ Registro Individual")

    def procesar_registro_individual(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Abrir Registro ECG", "", "Archivos WFDB (*.hea *.dat)")
        if not file_path:
            return

        path_objeto = Path(file_path)
        nombre_base = path_objeto.stem  # Extrae el nombre sin extensión
        directorio = path_objeto.parent

        try:
            # 1. Cargar la señal cruda
            signals, fields = wfdb.rdsamp(str(directorio / nombre_base))
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
            
            # Rellenar Tabla de Características
            descriptores_dict = micro_dataset['PACIENTE_GUI']['descriptores']
            for idx, (llave, valor) in enumerate(descriptores_dict.items()):
                self.tabla_features.setItem(idx, 0, QTableWidgetItem(llave))
                self.tabla_features.setItem(idx, 1, QTableWidgetItem(f"{valor:.4f}"))

            # Graficar Canales Clínicos Clave (V1, V6, I)
            self.canvas_individual.axes.clear()
            t = np.arange(len(s_v1)) / fs
            # Mostramos los primeros 3 segundos para una visualización clínica óptima
            max_samples = int(fs * 3) 
            
            self.canvas_individual.axes.plot(t[:max_samples], s_v1[:max_samples], label="V1 (Polaridad/Área)", color='#EF4444')
            self.canvas_individual.axes.plot(t[:max_samples], s_v6[:max_samples], label="V6 (R/S Ratio)", color='#3B82F6')
            self.canvas_individual.axes.plot(t[:max_samples], s_i[:max_samples], label="Lead I (Ancho QRS)", color='#10B981')
            
            self.canvas_individual.axes.set_title(f"Trazado Electrocardiográfico — Registro: {nombre_base}")
            self.canvas_individual.axes.set_xlabel("Tiempo (segundos)")
            self.canvas_individual.axes.set_ylabel("Amplitud (mV)")
            self.canvas_individual.axes.legend(loc="upper right")
            self.canvas_individual.axes.grid(True, linestyle='--', alpha=0.5)
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

        self.tab_widget.addTab(widget, "📦 Análisis por Lote")

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

            self.lbl_estado_lote.setText(f"✓ ¡Éxito! Lote completado de forma íntegra.\\nSe procesaron {len(df_clasificado)} registros válidos.")

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
        self.txt_reporte.setFont(QFont("Courier New", 10)) # Fuente monoespaciada estricta para reportes
        self.txt_reporte.setReadOnly(True)
        panel_izq.addWidget(self.txt_reporte)

        layout.addLayout(panel_izq, stretch=2)

        # Espacio para inyectar la matriz de confusión directamente en el layout
        self.canvas_matriz = MplCanvas(self, width=5, height=5)
        layout.addWidget(self.canvas_matriz, stretch=3)

        self.tab_widget.addTab(widget, "🔬 Evaluar Algoritmo")

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
            reporte_str = classification_report(y_true, y_pred, target_names=src.clasificador.CLASES, zero_division=0)
            self.txt_reporte.setText(reporte_str)

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
