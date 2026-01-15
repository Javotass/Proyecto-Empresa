# Sistema de Detección de Operaciones Atípicas en Transacciones Financieras

Sistema completo de detección de fraude en transacciones financieras utilizando Machine Learning, con múltiples modelos de detección de anomalías y análisis exploratorio.

## 📋 Características Principales

### ✅ Funcionalidades Implementadas

1. **Generación y obtención de datos** 
   - Generación de datasets sintéticos realistas con `faker`
   - Simulación de transacciones normales y fraudulentas
   - Importación de datos desde CSV externos
   - Etiquetado claro de fraude/no fraude

2. **Gestión y preparación de datos** 
   - Normalización y transformación de variables
   - Eliminación de inconsistencias y valores erróneos
   - Tratamiento de desbalanceo de clases
   - Encoding de variables categóricas

3. **Análisis exploratorio completo** 
   - Análisis de distribuciones de importes y frecuencias
   - Patrones temporales (hora, día, semana)
   - Comportamiento por usuario
   - Detección de outliers estadísticos
   - Matrices de correlación

4. **División del dataset** 
   - Separación train/test (70/30)
   - Estratificación para mantener proporciones
   - Evaluación objetiva del rendimiento

5. **Múltiples modelos predictivos** 
   - **No supervisados**: Isolation Forest, Local Outlier Factor
   - **Supervisados**: Random Forest, Regresión Logística, Árbol de Decisión
   - Comparación automática de rendimiento

6. **Ajuste y validación** 
   - Optimización de hiperparámetros con GridSearchCV
   - Validación cruzada (k-fold)
   - Métricas completas: Precision, Recall, F1-Score, AUC, Matriz de Confusión

7. **Sistema de alertas** 
   - Clasificación por nivel de riesgo (CRÍTICO/ALTO/MEDIO)
   - Generación automática de reportes
   - Identificación de transacciones sospechosas

8. **Monitorización y registro** 
   - Registro completo de transacciones analizadas
   - Trazabilidad de decisiones
   - Histórico de alertas

9. **Interpretabilidad** 
   - Análisis de importancia de características
   - Visualización de variables influyentes
   - Explicación de detecciones

10. **Visualización avanzada** 
    - Curvas ROC, Matrices de confusión
    - Distribuciones de anomalías
    - Patrones temporales, Mapas de calor geográficos
    - Timeline de clientes

11. **Arquitectura modular** 
    - Sistema completamente modular
    - Fácil sustitución de modelos
    - Escalable y extensible

12. **Documentación completa** 
    - Código documentado
    - README detallado
    - Resultados exportados

## 🚀 Instalación

### Requisitos previos
- Python 3.8 o superior
- pip

### Instalación de dependencias

```bash
pip install -r requirements.txt
```

## 💻 Uso

### Ejecución completa

```bash
python main_completo.py
```

Este comando ejecutará las 13 fases del sistema:
1. Generación de datos sintéticos
2. Análisis exploratorio de datos (EDA)
3. Procesamiento y transformación
4. División train/test (70/30)
5. Entrenamiento de 5 modelos
6. Optimización de hiperparámetros
7. Validación cruzada
8. Predicción y evaluación en test
9. Análisis de importancia de características
10. Evaluación detallada del mejor modelo
11. Visualización de resultados
12. Exportación de resultados
13. Sistema de alertas

## 📊 Estructura del Proyecto

```
Proyecto-Empresa/
│
├── main_completo.py                 # Script principal completo (13 fases)
├── requirements.txt                 # Dependencias del proyecto
├── README.md                        # Este archivo
│
├── data/                            # Directorio de datos y resultados
│   ├── transactions.csv             # Dataset generado
│   ├── transactions_analyzed_complete.csv  # Resultados del análisis
│   ├── alerts_report.csv            # Reporte de alertas
│   ├── model_comparison_results.csv # Comparación de modelos
│   ├── feature_importance_*.csv     # Importancia de características
│   └── *.png                        # 12+ visualizaciones generadas
│
└── src/                             # Código fuente modular
    ├── __init__.py                  # Inicializador del paquete
    ├── config.py                    # Configuración centralizada
    ├── data_generator.py            # Generación de datos sintéticos
    ├── data_processor.py            # Preprocesamiento de datos
    ├── exploratory_analyzer.py      # Análisis exploratorio (EDA)
    ├── model_comparator.py          # Entrenamiento y comparación de 5 modelos
    ├── evaluator.py                 # Evaluación de modelos
    └── visualizer.py                # Visualización de resultados
```

## 🧠 Modelos Implementados

### Modelos No Supervisados (Detección de Anomalías)
1. **Isolation Forest**: Aísla anomalías mediante particiones aleatorias
2. **Local Outlier Factor (LOF)**: Detecta outliers basándose en densidad local

### Modelos Supervisados (Clasificación)
3. **Random Forest**: Ensemble de árboles de decisión
4. **Regresión Logística**: Clasificación lineal probabilística
5. **Árbol de Decisión**: Modelo interpretable basado en reglas

## 📈 Métricas de Evaluación

El sistema evalúa cada modelo con:
- **Accuracy**: Precisión general
- **Precision**: Proporción de detecciones correctas
- **Recall**: Capacidad de detectar fraudes
- **F1-Score**: Media armónica de precision y recall
- **AUC-ROC**: Área bajo la curva ROC
- **Matriz de Confusión**: Verdaderos/Falsos Positivos y Negativos

## 🔍 Análisis Exploratorio

El sistema genera automáticamente:
- Distribución de importes normales vs fraudulentos
- Patrones de actividad por hora y día
- Comportamiento de clientes (frecuencia y montos)
- Detección de outliers estadísticos (IQR)
- Matrices de correlación entre variables
- Heatmaps geográficos de transacciones

## ⚠️ Sistema de Alertas

Las transacciones se clasifican en 3 niveles de riesgo basándose en el percentil 95 de anomalía:
- **🔴 CRÍTICO**: Top tercil de alertas (scores más altos)
- **🟠 ALTO**: Tercil medio de alertas
- **🟡 MEDIO**: Tercil inferior de alertas

Solo se generan alertas para transacciones con score > percentil 95

## 📁 Archivos Generados

### CSV:
- `transactions_analyzed_complete.csv`: Análisis completo con predicciones
- `alerts_report.csv`: Alertas clasificadas por riesgo
- `model_comparison_results.csv`: Comparación de modelos
- `feature_importance_*.csv`: Importancia de características

### Visualizaciones:
- Distribuciones, patrones temporales, comportamiento
- Curvas ROC, matrices de confusión
- Mapas de calor, timelines de clientes

## 🛠️ Tecnologías

- **Python 3.10+**
- **Análisis de datos**: pandas 2.3.3, numpy 2.2.6
- **Machine Learning**: scikit-learn 1.7.2
- **Generación de datos**: faker 33.1.0
- **Visualización**: matplotlib 3.10.0, seaborn 0.13.2

## 👤 Autor

**Javier Revilla** - Versión 1.0.0

