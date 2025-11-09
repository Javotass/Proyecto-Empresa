import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

class ResultVisualizer:
    """Visualización de resultados del modelo"""
    
    def __init__(self, style='darkgrid'):
        sns.set_style(style)
        plt.rcParams['figure.figsize'] = (12, 6)
        
    def plot_anomaly_scores(self, scores, is_anomaly=None, save_path=None):
        """Histograma de puntuaciones de anomalía"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        if is_anomaly is not None:
            ax.hist(scores[is_anomaly == 0], bins=50, alpha=0.7, label='Normal', color='green')
            ax.hist(scores[is_anomaly == 1], bins=50, alpha=0.7, label='Anomalía', color='red')
            ax.legend()
        else:
            ax.hist(scores, bins=50, alpha=0.7, color='blue')
        
        ax.set_xlabel('Puntuación de Anomalía')
        ax.set_ylabel('Frecuencia')
        ax.set_title('Distribución de Puntuaciones de Anomalía')
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_roc_curve(self, metrics, save_path=None):
        """Curva ROC"""
        if 'fpr' not in metrics or 'tpr' not in metrics:
            print("No hay datos suficientes para generar la curva ROC")
            return
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        ax.plot(metrics['fpr'], metrics['tpr'], 'b-', linewidth=2, 
                label=f"AUC = {metrics.get('auc', 0):.4f}")
        ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Aleatorio')
        
        ax.set_xlabel('Tasa de Falsos Positivos')
        ax.set_ylabel('Tasa de Verdaderos Positivos')
        ax.set_title('Curva ROC')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_confusion_matrix(self, cm, save_path=None):
        """Mapa de calor de la matriz de confusión"""
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Normal', 'Anomalía'],
                    yticklabels=['Normal', 'Anomalía'])
        
        ax.set_ylabel('Valor Real')
        ax.set_xlabel('Valor Predicho')
        ax.set_title('Matriz de Confusión')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_transaction_timeline(self, df, customer_id, save_path=None):
        """Serie temporal de transacciones de un cliente"""
        customer_data = df[df['customer_id'] == customer_id].sort_values('timestamp')
        
        if len(customer_data) == 0:
            print(f"No se encontraron transacciones para el cliente {customer_id}")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
        
        # Gráfico de importes
        colors = ['red' if x == 1 else 'green' for x in customer_data['is_anomaly']]
        ax1.scatter(customer_data['timestamp'], customer_data['amount'], 
                   c=colors, alpha=0.6, s=100)
        ax1.set_ylabel('Importe')
        ax1.set_title(f'Transacciones del Cliente {customer_id}')
        ax1.grid(True, alpha=0.3)
        
        # Gráfico de frecuencia
        customer_data['date'] = customer_data['timestamp'].dt.date
        daily_counts = customer_data.groupby('date').size()
        ax2.plot(daily_counts.index, daily_counts.values, marker='o', linewidth=2)
        ax2.set_xlabel('Fecha')
        ax2.set_ylabel('Número de Transacciones')
        ax2.set_title('Frecuencia Diaria de Transacciones')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_country_heatmap(self, df, save_path=None):
        """Mapa de calor de transacciones por país"""
        country_matrix = pd.crosstab(df['origin_country'], df['destination_country'])
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        sns.heatmap(country_matrix, annot=True, fmt='d', cmap='YlOrRd', ax=ax)
        
        ax.set_title('Matriz de Transacciones por País (Origen vs Destino)')
        ax.set_xlabel('País de Destino')
        ax.set_ylabel('País de Origen')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()