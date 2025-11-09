from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import pandas as pd
import numpy as np

class ModelEvaluator:
    """Evaluación de modelos de detección de anomalías"""
    
    def __init__(self):
        self.metrics = {}
        
    def evaluate(self, y_true, y_pred, y_scores=None):
        """Evalúa el rendimiento del modelo"""
        print("\n=== EVALUACIÓN DEL MODELO ===\n")
        
        # Reporte de clasificación
        print("Reporte de Clasificación:")
        print(classification_report(y_true, y_pred, target_names=['Normal', 'Anomalía']))
        
        # Matriz de confusión
        cm = confusion_matrix(y_true, y_pred)
        print("\nMatriz de Confusión:")
        print(f"                Predicho Normal  Predicho Anomalía")
        print(f"Real Normal          {cm[0][0]:<15}  {cm[0][1]}")
        print(f"Real Anomalía        {cm[1][0]:<15}  {cm[1][1]}")
        
        # Métricas adicionales
        if y_scores is not None and len(np.unique(y_true)) > 1:
            auc = roc_auc_score(y_true, y_scores)
            print(f"\nÁrea bajo la curva ROC: {auc:.4f}")
            self.metrics['auc'] = auc
            self.metrics['fpr'], self.metrics['tpr'], self.metrics['thresholds'] = roc_curve(y_true, y_scores)
        
        self.metrics['confusion_matrix'] = cm
        self.metrics['y_true'] = y_true
        self.metrics['y_pred'] = y_pred
        if y_scores is not None:
            self.metrics['y_scores'] = y_scores
        
        return self.metrics
    
    def analyze_anomalies(self, df, anomaly_scores, top_n=10):
        """Analiza las principales anomalías detectadas"""
        df_analysis = df.copy()
        df_analysis['anomaly_score'] = anomaly_scores
        df_analysis['predicted_anomaly'] = (anomaly_scores > np.percentile(anomaly_scores, 95)).astype(int)
        
        print(f"\n=== TOP {top_n} TRANSACCIONES MÁS ANÓMALAS ===\n")
        top_anomalies = df_analysis.nlargest(top_n, 'anomaly_score')[
            ['transaction_id', 'customer_id', 'amount', 'destination_country', 
             'timestamp', 'anomaly_score', 'is_anomaly']
        ]
        print(top_anomalies.to_string(index=False))
        
        return df_analysis