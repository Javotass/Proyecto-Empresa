import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

class ExploratoryAnalyzer:
    """Analisis exploratorio de datos de transacciones"""
    
    def __init__(self):
        sns.set_style('whitegrid')
        
    def analyze_distribution(self, df):
        """Analisis de distribucion de variables principales"""
        print("\n=== ANALISIS DE DISTRIBUCION ===\n")
        
        # Distribucion de importes
        print("Estadisticas de importes:")
        print(df['amount'].describe())
        
        # Distribucion de fraudes
        if 'is_anomaly' in df.columns:
            fraud_counts = df['is_anomaly'].value_counts()
            print(f"\nDistribucion de transacciones:")
            print(f"  Normales: {fraud_counts.get(0, 0)} ({fraud_counts.get(0, 0)/len(df)*100:.2f}%)")
            print(f"  Fraudulentas: {fraud_counts.get(1, 0)} ({fraud_counts.get(1, 0)/len(df)*100:.2f}%)")
        
        # Distribucion por canal
        print(f"\nTransacciones por canal:")
        print(df['channel'].value_counts())
        
        # Distribucion por pais
        print(f"\nTop 10 paises de origen:")
        print(df['origin_country'].value_counts().head(10))
    
    def plot_amount_distribution(self, df, save_path=None):
        """Grafico de distribucion de importes"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histograma general
        axes[0].hist(df['amount'], bins=50, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Importe')
        axes[0].set_ylabel('Frecuencia')
        axes[0].set_title('Distribucion de Importes')
        axes[0].grid(True, alpha=0.3)
        
        # Boxplot por tipo de transaccion
        if 'is_anomaly' in df.columns:
            df['tipo'] = df['is_anomaly'].map({0: 'Normal', 1: 'Fraude'})
            df.boxplot(column='amount', by='tipo', ax=axes[1])
            axes[1].set_xlabel('Tipo de Transaccion')
            axes[1].set_ylabel('Importe')
            axes[1].set_title('Importes por Tipo')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_temporal_patterns(self, df, save_path=None):
        """Analisis de patrones temporales"""
        df_copy = df.copy()
        df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])
        df_copy['hour'] = df_copy['timestamp'].dt.hour
        df_copy['day_of_week'] = df_copy['timestamp'].dt.dayofweek
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Transacciones por hora
        hourly_counts = df_copy['hour'].value_counts().sort_index()
        axes[0, 0].bar(hourly_counts.index, hourly_counts.values, edgecolor='black')
        axes[0, 0].set_xlabel('Hora del dia')
        axes[0, 0].set_ylabel('Numero de transacciones')
        axes[0, 0].set_title('Transacciones por Hora')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Transacciones por dia de la semana
        days = ['Lun', 'Mar', 'Mie', 'Jue', 'Vie', 'Sab', 'Dom']
        daily_counts = df_copy['day_of_week'].value_counts().sort_index()
        axes[0, 1].bar(range(len(daily_counts)), daily_counts.values, 
                       tick_label=[days[i] for i in daily_counts.index],
                       edgecolor='black')
        axes[0, 1].set_xlabel('Dia de la semana')
        axes[0, 1].set_ylabel('Numero de transacciones')
        axes[0, 1].set_title('Transacciones por Dia de la Semana')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Heatmap: hora vs dia de semana
        pivot_table = df_copy.pivot_table(
            values='amount', 
            index='hour', 
            columns='day_of_week', 
            aggfunc='count',
            fill_value=0
        )
        sns.heatmap(pivot_table, ax=axes[1, 0], cmap='YlOrRd', annot=False)
        axes[1, 0].set_xlabel('Dia de la semana')
        axes[1, 0].set_ylabel('Hora')
        axes[1, 0].set_title('Heatmap: Actividad por Hora y Dia')
        
        # Importes promedio por hora
        if 'is_anomaly' in df_copy.columns:
            hourly_avg = df_copy.groupby(['hour', 'is_anomaly'])['amount'].mean().unstack()
            hourly_avg.plot(ax=axes[1, 1], marker='o', linewidth=2)
            axes[1, 1].set_xlabel('Hora del dia')
            axes[1, 1].set_ylabel('Importe promedio')
            axes[1, 1].set_title('Importe Promedio por Hora y Tipo')
            axes[1, 1].legend(['Normal', 'Fraude'])
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_customer_behavior(self, df, save_path=None):
        """Analisis del comportamiento de clientes"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Transacciones por cliente
        transactions_per_customer = df.groupby('customer_id').size()
        axes[0].hist(transactions_per_customer, bins=30, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Numero de transacciones')
        axes[0].set_ylabel('Numero de clientes')
        axes[0].set_title('Distribucion de Transacciones por Cliente')
        axes[0].grid(True, alpha=0.3)
        
        # Importe promedio por cliente
        avg_amount_per_customer = df.groupby('customer_id')['amount'].mean()
        axes[1].hist(avg_amount_per_customer, bins=30, edgecolor='black', 
                     alpha=0.7, color='orange')
        axes[1].set_xlabel('Importe promedio')
        axes[1].set_ylabel('Numero de clientes')
        axes[1].set_title('Distribucion de Importe Promedio por Cliente')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def detect_outliers(self, df):
        """Detecta outliers estadisticos en importes"""
        print("\n=== DETECCION DE OUTLIERS ===\n")
        
        Q1 = df['amount'].quantile(0.25)
        Q3 = df['amount'].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df['amount'] < lower_bound) | (df['amount'] > upper_bound)]
        
        print(f"Limite inferior: {lower_bound:.2f}")
        print(f"Limite superior: {upper_bound:.2f}")
        print(f"Numero de outliers: {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")
        print(f"\nTop 10 outliers por importe:")
        print(outliers.nlargest(10, 'amount')[['transaction_id', 'customer_id', 'amount', 'origin_country']])
        
        return outliers
    
    def correlation_analysis(self, df, save_path=None):
        """Analisis de correlacion entre variables numericas"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            print("No hay suficientes columnas numericas para analisis de correlacion")
            return
        
        correlation_matrix = df[numeric_cols].corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', 
                    cmap='coolwarm', center=0, square=True)
        plt.title('Matriz de Correlacion')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\n=== CORRELACIONES MAS FUERTES ===\n")
        # Encontrar correlaciones fuertes (excluyendo diagonal)
        strong_corr = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = correlation_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:
                    strong_corr.append({
                        'Variable 1': correlation_matrix.columns[i],
                        'Variable 2': correlation_matrix.columns[j],
                        'Correlacion': corr_val
                    })
        
        if strong_corr:
            df_corr = pd.DataFrame(strong_corr).sort_values('Correlacion', 
                                                             key=abs, 
                                                             ascending=False)
            print(df_corr.to_string(index=False))
        else:
            print("No se encontraron correlaciones fuertes (> 0.5)")
