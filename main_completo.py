# -*- coding: utf-8 -*-
"""Sistema de Deteccion de Operaciones Atipicas - Version Completa con Multiples Modelos"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.data_generator import TransactionDataGenerator
from src.data_processor import DataProcessor
from src.model_comparator import ModelComparator
from src.evaluator import ModelEvaluator
from src.visualizer import ResultVisualizer
from src.exploratory_analyzer import ExploratoryAnalyzer
from src import config


def print_dataset_split_info(X_train, y_train, X_test, y_test):
    """Muestra informacion sobre la division del dataset"""
    print(f"Conjunto de entrenamiento: {len(X_train)} muestras")
    print(f"  - Normales: {sum(y_train == 0)} ({sum(y_train == 0)/len(y_train)*100:.2f}%)")
    print(f"  - Anomalias: {sum(y_train == 1)} ({sum(y_train == 1)/len(y_train)*100:.2f}%)")
    print(f"\nConjunto de prueba: {len(X_test)} muestras")
    print(f"  - Normales: {sum(y_test == 0)} ({sum(y_test == 0)/len(y_test)*100:.2f}%)")
    print(f"  - Anomalias: {sum(y_test == 1)} ({sum(y_test == 1)/len(y_test)*100:.2f}%)")

def main():
    """Flujo principal del sistema de deteccion de anomalias"""
    
    print(config.MESSAGES['header'])
    
    # ========== FASE 1: GENERACION DE DATOS ==========
    print(config.MESSAGES['phase_1'])
    generator = TransactionDataGenerator(
        n_transactions=config.N_TRANSACTIONS,
        anomaly_ratio=config.ANOMALY_RATIO,
        seed=config.RANDOM_STATE
    )
    df = generator.generate_dataset()
    generator.save_to_csv(df, config.DATASET_PATH)
    
    # ========== FASE 2: ANALISIS EXPLORATORIO ==========
    print(config.MESSAGES['phase_2'])
    analyzer = ExploratoryAnalyzer()
    
    analyzer.analyze_distribution(df)
    analyzer.plot_amount_distribution(df, save_path=config.PLOTS['eda_amount'])
    analyzer.plot_temporal_patterns(df, save_path=config.PLOTS['eda_temporal'])
    analyzer.plot_customer_behavior(df, save_path=config.PLOTS['eda_customer'])
    outliers = analyzer.detect_outliers(df)
    
    # ========== FASE 3: PREPROCESAMIENTO ==========
    print(config.MESSAGES['phase_3'])
    processor = DataProcessor()
    df_processed = processor.preprocess(df)
    df_encoded = processor.encode_features(df_processed)
    
    analyzer.correlation_analysis(df_encoded, save_path=config.PLOTS['eda_correlation'])
    
    X = processor.prepare_features(df_encoded)
    X_scaled = processor.scale_features(X, fit=True)
    y_true = df_encoded['is_anomaly'].values
    
    print(f"\nForma del conjunto de caracteristicas: {X_scaled.shape}")
    print(f"Caracteristicas utilizadas: {list(X.columns)}")
    
    # ========== FASE 4: DIVISION TRAIN/TEST ==========
    print(config.MESSAGES['phase_4'])
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X_scaled, y_true, df_encoded.index,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y_true
    )
    
    print_dataset_split_info(X_train, y_train, X_test, y_test)
    
    # ========== FASE 5: ENTRENAMIENTO DE MULTIPLES MODELOS ==========
    print(config.MESSAGES['phase_5'])
    comparator = ModelComparator(
        contamination=config.CONTAMINATION,
        random_state=config.RANDOM_STATE
    )
    comparator.train_all_models(X_train, y_train)
    
    # ========== FASE 6: OPTIMIZACION DE HIPERPARAMETROS ==========
    print(config.MESSAGES['phase_6'])
    comparator.optimize_hyperparameters('random_forest', X_train, y_train)
    comparator.optimize_hyperparameters('decision_tree', X_train, y_train)
    
    # ========== FASE 7: VALIDACION CRUZADA ==========
    print(config.MESSAGES['phase_7'])
    comparator.cross_validate_model('random_forest', X_train, y_train, cv=config.CV_FOLDS)
    comparator.cross_validate_model('decision_tree', X_train, y_train, cv=config.CV_FOLDS)
    comparator.cross_validate_model('logistic_regression', X_train, y_train, cv=config.CV_FOLDS)
    
    # ========== FASE 8: PREDICCION Y EVALUACION EN TEST ==========
    print(config.MESSAGES['phase_8'])
    predictions, scores = comparator.predict_all(X_test)
    results_df = comparator.compare_models(y_test, predictions, scores)
    results_df.to_csv(config.COMPARISON_PATH, index=False)
    print(f"\n[OK] Resultados de comparacion guardados en: {config.COMPARISON_PATH}")
    
    # ========== FASE 9: ANALISIS DE IMPORTANCIA DE CARACTERISTICAS ==========
    print(config.MESSAGES['phase_9'])
    feature_importance_rf = comparator.get_feature_importance('random_forest', X.columns)
    if feature_importance_rf is not None:
        feature_importance_rf.to_csv(f'{config.DATA_DIR}/feature_importance_rf.csv', index=False)
    
    feature_importance_dt = comparator.get_feature_importance('decision_tree', X.columns)
    if feature_importance_dt is not None:
        feature_importance_dt.to_csv(f'{config.DATA_DIR}/feature_importance_dt.csv', index=False)
    
    # ========== FASE 10: EVALUACION DETALLADA DEL MEJOR MODELO ==========
    print(config.MESSAGES['phase_10'])
    best_model_name = results_df.iloc[0]['Modelo']
    print(f"\nEvaluando en detalle: {best_model_name}")
    
    y_pred_best = predictions[best_model_name]
    y_scores_best = scores[best_model_name]
    
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(y_test, y_pred_best, y_scores_best)
    
    df_test = df_encoded.loc[idx_test].copy()
    df_analysis = evaluator.analyze_anomalies(df_test, y_scores_best, top_n=15)
    
    # ========== FASE 11: VISUALIZACION DE RESULTADOS ==========
    print(config.MESSAGES['phase_11'])
    visualizer = ResultVisualizer()
    
    visualizer.plot_anomaly_scores(
        y_scores_best, y_test,
        save_path=f'{config.DATA_DIR}/{best_model_name}_anomaly_scores.png'
    )
    visualizer.plot_roc_curve(
        metrics,
        save_path=f'{config.DATA_DIR}/{best_model_name}_roc_curve.png'
    )
    visualizer.plot_confusion_matrix(
        metrics['confusion_matrix'],
        save_path=f'{config.DATA_DIR}/{best_model_name}_confusion_matrix.png'
    )
    visualizer.plot_country_heatmap(df_analysis, save_path=config.PLOTS['country_heatmap'])
    
    anomaly_customers = df_analysis[df_analysis['is_anomaly'] == 1]['customer_id'].unique()
    if len(anomaly_customers) > 0:
        visualizer.plot_transaction_timeline(
            df_analysis, anomaly_customers[0],
            save_path=f'{config.DATA_DIR}/customer_timeline.png'
        )
    
    # ========== FASE 12: EXPORTAR RESULTADOS FINALES ==========
    print("\n[FASE 12] Exportacion de Resultados Finales")
    
    # Anadir predicciones de todos los modelos al analisis
    for model_name in predictions.keys():
        df_analysis[f'pred_{model_name}'] = predictions[model_name]
        df_analysis[f'score_{model_name}'] = scores[model_name]
    
    df_analysis.to_csv(config.PLOTS['analyzed_transactions'], index=False)
    print(f"[OK] Analisis completo guardado en: {config.PLOTS['analyzed_transactions']}")
    
    # Generar reporte de alertas
    print(config.MESSAGES['phase_12'])
    generate_alerts_report(df_analysis, best_model_name)
    
    print("\n" + "="*80)
    print(config.MESSAGES['complete'])
    print("="*80 + "\n")
    print(f"\n>> Resumen Final:")
    print(f"  - Total transacciones analizadas: {len(df)}")
    print(f"  - Modelos entrenados: {len(predictions)}")
    print(f"  - Mejor modelo: {best_model_name}")
    print(f"  - F1-Score del mejor modelo: {results_df.iloc[0]['F1-Score']:.4f}")
    print(f"  - Anomalias detectadas en test: {sum(y_pred_best)}")
    print(f"  - Alertas generadas: Ver {config.DATA_DIR}/alerts_report.csv")


def generate_alerts_report(df_analysis, best_model_name):
    """Genera reporte de alertas para transacciones sospechosas"""
    
    # Filtrar transacciones con alta probabilidad de fraude
    score_col = f'score_{best_model_name}'
    threshold = df_analysis[score_col].quantile(config.ALERT_PERCENTILE / 100)
    
    alerts = df_analysis[df_analysis[score_col] > threshold].copy()
    
    if len(alerts) == 0:
        print(config.MESSAGES['no_alerts'])
        return
    
    # Clasificar alertas por nivel de riesgo usando qcut para manejar duplicados
    try:
        # Intentar clasificar en 3 categorias
        alerts['risk_level'] = pd.qcut(
            alerts[score_col],
            q=3,
            labels=config.RISK_LEVELS,
            duplicates='drop'
        )
    except ValueError:
        # Si no hay suficiente variabilidad, asignar por percentiles manualmente
        p33 = alerts[score_col].quantile(0.33)
        p67 = alerts[score_col].quantile(0.67)
        
        alerts['risk_level'] = config.RISK_LEVELS[2]  # Por defecto CRITICO
        alerts.loc[alerts[score_col] <= p33, 'risk_level'] = config.RISK_LEVELS[0]  # MEDIO
        alerts.loc[(alerts[score_col] > p33) & (alerts[score_col] < p67), 'risk_level'] = config.RISK_LEVELS[1]  # ALTO
    
    # Seleccionar columnas relevantes
    alert_columns = [
        'transaction_id', 'customer_id', 'timestamp', 'amount', 
        'origin_country', 'destination_country', 'channel',
        score_col, 'risk_level', 'is_anomaly'
    ]
    
    alerts_report = alerts[alert_columns].sort_values(score_col, ascending=False)
    alerts_report.to_csv(f'{config.DATA_DIR}/alerts_report.csv', index=False)
    
    print(f"\n=== SISTEMA DE ALERTAS ===")
    print(f"Total de alertas generadas: {len(alerts_report)}")
    print(f"  - Riesgo {config.RISK_LEVELS[2]}: {sum(alerts_report['risk_level'] == config.RISK_LEVELS[2])}")
    print(f"  - Riesgo {config.RISK_LEVELS[1]}: {sum(alerts_report['risk_level'] == config.RISK_LEVELS[1])}")
    print(f"  - Riesgo {config.RISK_LEVELS[0]}: {sum(alerts_report['risk_level'] == config.RISK_LEVELS[0])}")
    print(f"\nTop 5 alertas criticas:")
    print(alerts_report.head()[['transaction_id', 'customer_id', 'amount', 'risk_level']])
    print(f"\n[OK] Reporte de alertas guardado en: {config.DATA_DIR}/alerts_report.csv")


if __name__ == "__main__":
    main()
