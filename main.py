import pandas as pd
import numpy as np
from src.data_generator import TransactionDataGenerator
from src.data_processor import DataProcessor
from src.model_trainer import AnomalyDetector
from src.evaluator import ModelEvaluator
from src.visualizer import ResultVisualizer

def main():
    """Flujo principal del sistema de detección de anomalías"""
    
    print("="*70)
    print(" SISTEMA DE DETECCIÓN DE OPERACIONES ATÍPICAS EN TRANSACCIONES")
    print("="*70)
    
    # ========== FASE 1: GENERACIÓN DE DATOS ==========
    print("\n[FASE 1] Generación de datos sintéticos")
    generator = TransactionDataGenerator(
        n_transactions=10000,
        anomaly_ratio=0.05,
        seed=42
    )
    df = generator.generate_dataset()
    generator.save_to_csv(df, 'data/transactions.csv')
    
    # ========== FASE 2: PREPROCESAMIENTO ==========
    print("\n[FASE 2] Procesamiento y transformación de datos")
    processor = DataProcessor()
    df_processed = processor.preprocess(df)
    df_encoded = processor.encode_features(df_processed)
    
    # Preparar características
    X = processor.prepare_features(df_encoded)
    X_scaled = processor.scale_features(X, fit=True)
    
    # Etiquetas reales (para evaluación)
    y_true = df_encoded['is_anomaly'].values
    
    print(f"\nForma del conjunto de características: {X_scaled.shape}")
    print(f"Características utilizadas: {list(X.columns)}")
    
    # ========== FASE 3: ENTRENAMIENTO DEL MODELO ==========
    print("\n[FASE 3] Entrenamiento del modelo de detección")
    detector = AnomalyDetector(
        model_type='isolation_forest',
        contamination=0.05,
        random_state=42
    )
    detector.train(X_scaled)
    
    # Predicciones
    y_pred = detector.predict(X_scaled)
    anomaly_scores = detector.score_samples(X_scaled)
    
    # ========== FASE 4: EVALUACIÓN ==========
    print("\n[FASE 4] Evaluación del modelo")
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(y_true, y_pred, anomaly_scores)
    
    # Análisis de anomalías
    df_analysis = evaluator.analyze_anomalies(df_encoded, anomaly_scores, top_n=15)
    
    # ========== FASE 5: VISUALIZACIÓN ==========
    print("\n[FASE 5] Visualización de resultados")
    visualizer = ResultVisualizer()
    
    # Gráfico de puntuaciones
    visualizer.plot_anomaly_scores(anomaly_scores, y_true, save_path='data/anomaly_scores.png')
    
    # Curva ROC
    visualizer.plot_roc_curve(metrics, save_path='data/roc_curve.png')
    
    # Matriz de confusión
    visualizer.plot_confusion_matrix(metrics['confusion_matrix'], save_path='data/confusion_matrix.png')
    
    # Mapa de calor de países
    visualizer.plot_country_heatmap(df_analysis, save_path='data/country_heatmap.png')
    
    # Timeline de un cliente con anomalías
    anomaly_customers = df_analysis[df_analysis['is_anomaly'] == 1]['customer_id'].unique()
    if len(anomaly_customers) > 0:
        sample_customer = anomaly_customers[0]
        visualizer.plot_transaction_timeline(df_analysis, sample_customer, 
                                            save_path='data/customer_timeline.png')
    
    # ========== EXPORTAR RESULTADOS ==========
    print("\n[FASE 6] Exportación de resultados")
    df_analysis.to_csv('data/transactions_analyzed.csv', index=False)
    print("Resultados guardados en: data/transactions_analyzed.csv")
    
    print("\n" + "="*70)
    print(" PROCESO COMPLETADO EXITOSAMENTE")
    print("="*70)

if __name__ == "__main__":
    main()