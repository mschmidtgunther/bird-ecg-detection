from pathlib import Path
import pandas as pd
from src.data_loader import construir_dataset, filtrar_dataset
from src.features import extraer_todas_las_features
from src.clasificador import clasificar_dataset, evaluar

# 1. Configuración de Rutas (Ajustá según tu estructura local)
PATH_DATA = Path("data/raw") 

def ejecutar_pipeline_clinico():
    # 2. Carga y Muestreo Orientado a Cuartiles
    dataset = construir_dataset(PATH_DATA, pacientes_por_clase=100, seed=42)
    
    # 3. Procesamiento y Filtrado de Señales (Butterworth Pasabanda)
    dataset = filtrar_dataset(dataset, PATH_DATA)
    
    # 4. Extracción de Descriptores Fisiológicos del QRS
    df_features = extraer_todas_las_features(dataset)
    
    # 5. Clasificación Mediante Sistema Experto Heurístico (v6)
    df_clasificado = clasificar_dataset(df_features)
    
    # 6. Evaluación de Métricas Finales y Matriz de Confusión
    evaluar(df_clasificado, col_real='clase_real', col_pred='prediccion')

if __name__ == "__main__":
    ejecutar_pipeline_clinico()