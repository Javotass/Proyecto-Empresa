"""
Configuracion centralizada del proyecto
Mejora las mejores practicas al evitar valores hardcodeados
"""

# ========== CONFIGURACION GENERAL ==========
PROJECT_NAME = "Sistema de Deteccion de Operaciones Atipicas"
VERSION = "1.0.0"
AUTHOR = "Javier Revilla"

# ========== CONFIGURACION DE DATOS ==========
N_TRANSACTIONS = 10000
ANOMALY_RATIO = 0.05
RANDOM_STATE = 42

# ========== CONFIGURACION DE MODELOS ==========
CONTAMINATION = 0.05
TEST_SIZE = 0.3
CV_FOLDS = 5

# ========== CONFIGURACION DE RUTAS ==========
DATA_DIR = 'data'
DATASET_PATH = f'{DATA_DIR}/transactions.csv'
RESULTS_PATH = f'{DATA_DIR}/transactions_analyzed_complete.csv'
ALERTS_PATH = f'{DATA_DIR}/alerts_report.csv'
COMPARISON_PATH = f'{DATA_DIR}/model_comparison_results.csv'

# ========== CONFIGURACION DE VISUALIZACIONES ==========
FIGURE_DPI = 300
FIGURE_FORMAT = 'png'

# Rutas de graficos
PLOTS = {
    'eda_amount': f'{DATA_DIR}/eda_amount_distribution.png',
    'eda_temporal': f'{DATA_DIR}/eda_temporal_patterns.png',
    'eda_customer': f'{DATA_DIR}/eda_customer_behavior.png',
    'eda_correlation': f'{DATA_DIR}/eda_correlation.png',
    'country_heatmap': f'{DATA_DIR}/country_heatmap.png',
    'analyzed_transactions': f'{DATA_DIR}/transactions_analyzed_complete.csv',
}

# ========== CONFIGURACION DEL SISTEMA DE ALERTAS ==========
ALERT_PERCENTILE = 95  # Percentil para umbral de alertas
RISK_LEVELS = ['MEDIO', 'ALTO', 'CRITICO']  # Niveles de riesgo ordenados

# ========== MENSAJES DEL SISTEMA ==========
SEPARATOR = "=" * 80
SEPARATOR_SHORT = "=" * 70

MESSAGES = {
    'header': f"\n{SEPARATOR}\n {PROJECT_NAME}\n Version Completa con Multiples Modelos y Validacion\n{SEPARATOR}",
    'phase_1': "\n[FASE 1] Generacion de datos sinteticos",
    'phase_2': "\n[FASE 2] Analisis Exploratorio de Datos",
    'phase_3': "\n[FASE 3] Procesamiento y transformacion de datos",
    'phase_4': "\n[FASE 4] Division del dataset en Train/Test",
    'phase_5': "\n[FASE 5] Entrenamiento y Comparacion de Multiples Modelos",
    'phase_6': "\n[FASE 6] Optimizacion de Hiperparametros",
    'phase_7': "\n[FASE 7] Validacion Cruzada de Modelos Supervisados",
    'phase_8': "\n[FASE 8] Prediccion y Evaluacion en Conjunto de Test",
    'phase_9': "\n[FASE 9] Analisis de Importancia de Caracteristicas",
    'phase_10': "\n[FASE 10] Evaluacion Detallada del Mejor Modelo",
    'phase_11': "\n[FASE 11] Visualizacion de Resultados",
    'phase_12': "\n[FASE 12] Exportacion de Resultados Finales",
    'phase_13': "\n[FASE 13] Sistema de Alertas",
    'complete': " PROCESO COMPLETADO EXITOSAMENTE",
    'no_alerts': "\n=== SISTEMA DE ALERTAS ===\nNo se generaron alertas (umbral no alcanzado)",
}
