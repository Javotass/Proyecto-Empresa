import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random

class TransactionDataGenerator:
    """Generador de datos sintéticos de transacciones financieras"""
    
    def __init__(self, n_transactions=10000, anomaly_ratio=0.05, seed=42):
        """
        Args:
            n_transactions: Número total de transacciones a generar
            anomaly_ratio: Proporción de transacciones anómalas (0-1)
            seed: Semilla para reproducibilidad
        """
        self.n_transactions = n_transactions
        self.anomaly_ratio = anomaly_ratio
        self.seed = seed
        self.fake = Faker()
        Faker.seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        
    def generate_dataset(self):
        """Genera el conjunto completo de transacciones"""
        n_anomalies = int(self.n_transactions * self.anomaly_ratio)
        n_normal = self.n_transactions - n_anomalies
        
        print(f"Generando {n_normal} transacciones normales y {n_anomalies} anómalas...")
        
        # Generar transacciones normales
        normal_transactions = self._generate_normal_transactions(n_normal)
        
        # Generar transacciones anómalas
        anomaly_transactions = self._generate_anomaly_transactions(n_anomalies)
        
        # Combinar y mezclar
        all_transactions = pd.concat([normal_transactions, anomaly_transactions], ignore_index=True)
        all_transactions = all_transactions.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        
        # Asignar IDs únicos
        all_transactions['transaction_id'] = [f"TXN{str(i).zfill(8)}" for i in range(len(all_transactions))]
        
        print(f"Dataset generado: {len(all_transactions)} transacciones")
        return all_transactions
    
    def _generate_normal_transactions(self, n):
        """Genera transacciones con comportamiento normal"""
        n_customers = max(100, n // 50)  # Aproximadamente 50 transacciones por cliente
        customer_ids = [f"CUST{str(i).zfill(6)}" for i in range(n_customers)]
        
        transactions = []
        start_date = datetime.now() - timedelta(days=365)
        
        currencies = ['USD', 'EUR', 'GBP', 'JPY', 'CHF']
        channels = ['web', 'app', 'cajero', 'sucursal']
        devices = ['desktop', 'mobile', 'tablet', 'atm', 'pos']
        countries = ['US', 'GB', 'DE', 'FR', 'ES', 'IT', 'JP', 'CA', 'AU', 'MX']
        
        for _ in range(n):
            customer_id = random.choice(customer_ids)
            timestamp = start_date + timedelta(
                days=random.randint(0, 365),
                hours=random.randint(6, 22),  # Horario normal
                minutes=random.randint(0, 59)
            )
            
            # Importes normales con distribución log-normal
            amount = round(np.random.lognormal(mean=5, sigma=1), 2)
            amount = min(amount, 10000)  # Limitar importes normales
            
            currency = random.choice(currencies)
            origin_country = random.choice(countries[:5])  # Países más comunes
            destination_country = random.choice(countries[:5])
            channel = random.choice(channels)
            device_type = random.choice(devices)
            
            transactions.append({
                'transaction_id': None,
                'customer_id': customer_id,
                'timestamp': timestamp,
                'amount': amount,
                'currency': currency,
                'origin_country': origin_country,
                'destination_country': destination_country,
                'channel': channel,
                'device_type': device_type,
                'is_anomaly': 0
            })
        
        return pd.DataFrame(transactions)
    
    def _generate_anomaly_transactions(self, n):
        """Genera transacciones anómalas con patrones inusuales"""
        normal_df = self._generate_normal_transactions(n)
        
        for idx in range(len(normal_df)):
            anomaly_type = random.choice(['high_amount', 'unusual_country', 'unusual_time', 'mixed'])
            
            if anomaly_type == 'high_amount' or anomaly_type == 'mixed':
                # Importes excepcionalmente altos
                normal_df.loc[idx, 'amount'] = round(random.uniform(50000, 200000), 2)
            
            if anomaly_type == 'unusual_country' or anomaly_type == 'mixed':
                # Países de destino inusuales
                unusual_countries = ['KP', 'IR', 'SY', 'AF', 'YE']
                normal_df.loc[idx, 'destination_country'] = random.choice(unusual_countries)
            
            if anomaly_type == 'unusual_time' or anomaly_type == 'mixed':
                # Horarios inusuales (madrugada)
                timestamp = normal_df.loc[idx, 'timestamp']
                unusual_hour = random.randint(0, 4)
                normal_df.loc[idx, 'timestamp'] = timestamp.replace(hour=unusual_hour)
            
            normal_df.loc[idx, 'is_anomaly'] = 1
        
        return normal_df
    
    def save_to_csv(self, df, filepath='data/transactions.csv'):
        """Guarda el dataset en formato CSV"""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
        print(f"Dataset guardado en: {filepath}")