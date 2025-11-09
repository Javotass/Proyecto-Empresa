import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

class DataProcessor:
    """Procesamiento y transformación de datos de transacciones"""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def preprocess(self, df):
        """Preprocesamiento completo del dataset"""
        print("Iniciando preprocesamiento...")
        
        df = df.copy()
        
        # Convertir timestamp a datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Eliminar duplicados
        initial_len = len(df)
        df = df.drop_duplicates(subset=['transaction_id'])
        print(f"Duplicados eliminados: {initial_len - len(df)}")
        
        # Eliminar valores nulos
        df = df.dropna()
        
        # Extraer características temporales
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        
        # Calcular características agregadas por cliente
        df = self._add_customer_features(df)
        
        print("Preprocesamiento completado")
        return df
    
    def _add_customer_features(self, df):
        """Añade características derivadas del comportamiento del cliente"""
        print("Calculando características por cliente...")
        
        customer_stats = df.groupby('customer_id').agg({
            'amount': ['mean', 'std', 'min', 'max', 'count'],
            'destination_country': 'nunique',
            'channel': 'nunique'
        }).reset_index()
        
        customer_stats.columns = [
            'customer_id',
            'customer_avg_amount',
            'customer_std_amount',
            'customer_min_amount',
            'customer_max_amount',
            'customer_transaction_count',
            'customer_unique_countries',
            'customer_unique_channels'
        ]
        
        # Rellenar desviaciones nulas con 0
        customer_stats['customer_std_amount'].fillna(0, inplace=True)
        
        df = df.merge(customer_stats, on='customer_id', how='left')
        
        # Calcular desviación de la transacción respecto al comportamiento del cliente
        df['amount_deviation'] = (df['amount'] - df['customer_avg_amount']) / (df['customer_std_amount'] + 1e-6)
        
        return df
    
    def encode_features(self, df):
        """Codifica variables categóricas"""
        df = df.copy()
        
        categorical_cols = ['currency', 'origin_country', 'destination_country', 'channel', 'device_type']
        
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[col + '_encoded'] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                df[col + '_encoded'] = self.label_encoders[col].transform(df[col].astype(str))
        
        return df
    
    def prepare_features(self, df):
        """Prepara el conjunto de características para el modelo"""
        feature_columns = [
            'amount',
            'hour',
            'day_of_week',
            'day_of_month',
            'month',
            'customer_avg_amount',
            'customer_std_amount',
            'customer_transaction_count',
            'customer_unique_countries',
            'customer_unique_channels',
            'amount_deviation',
            'currency_encoded',
            'origin_country_encoded',
            'destination_country_encoded',
            'channel_encoded',
            'device_type_encoded'
        ]
        
        X = df[feature_columns].copy()
        X = X.fillna(0)
        
        return X
    
    def scale_features(self, X, fit=True):
        """Normaliza las características"""
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)