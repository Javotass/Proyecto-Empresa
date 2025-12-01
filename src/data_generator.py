import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random

class TransactionDataGenerator:
    """Generador de datos sintéticos de transacciones financieras con reglas controladas"""
    
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
        
        # Perfiles de clientes con patrones de comportamiento
        self.customer_profiles = {}
        
    def _create_customer_profile(self, customer_id):
        """Crea un perfil de comportamiento para un cliente"""
        # Tipos de cliente con diferentes patrones de gasto
        customer_types = ['low_spender', 'medium_spender', 'high_spender', 'business']
        customer_type = random.choice(customer_types)
        
        # Definir rangos típicos según el tipo de cliente
        if customer_type == 'low_spender':
            typical_range = (10, 500)
            typical_hour_range = (9, 21)
            typical_countries = ['ES', 'FR', 'IT']
        elif customer_type == 'medium_spender':
            typical_range = (100, 3000)
            typical_hour_range = (8, 22)
            typical_countries = ['ES', 'FR', 'DE', 'IT', 'GB']
        elif customer_type == 'high_spender':
            typical_range = (1000, 15000)
            typical_hour_range = (8, 23)
            typical_countries = ['ES', 'FR', 'DE', 'IT', 'GB', 'US']
        else:  # business
            typical_range = (500, 25000)
            typical_hour_range = (7, 19)
            typical_countries = ['ES', 'FR', 'DE', 'US', 'GB', 'CH']
        
        profile = {
            'type': customer_type,
            'typical_range': typical_range,
            'typical_hour_range': typical_hour_range,
            'typical_countries': typical_countries,
            'home_country': random.choice(typical_countries),
            'transaction_count': 0
        }
        
        self.customer_profiles[customer_id] = profile
        return profile
    
    def _get_customer_profile(self, customer_id):
        """Obtiene o crea el perfil de un cliente"""
        if customer_id not in self.customer_profiles:
            return self._create_customer_profile(customer_id)
        return self.customer_profiles[customer_id]
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
        """Genera transacciones normales siguiendo reglas controladas"""
        n_customers = max(100, n // 50)  # Aproximadamente 50 transacciones por cliente
        customer_ids = [f"CUST{str(i).zfill(6)}" for i in range(n_customers)]
        
        transactions = []
        start_date = datetime.now() - timedelta(days=365)
        
        currencies = ['EUR', 'USD', 'GBP', 'CHF']
        channels = ['web', 'app', 'cajero', 'sucursal']
        devices = ['desktop', 'mobile', 'tablet', 'atm', 'pos']
        
        for _ in range(n):
            customer_id = random.choice(customer_ids)
            profile = self._get_customer_profile(customer_id)
            profile['transaction_count'] += 1
            
            # REGLA 1: Transacciones < 1000€ - frecuencia alta, riesgo bajo
            # REGLA 2: Transacciones 1000-10000€ - moderadas
            # REGLA 3: Transacciones 10000-100000€ - solo para clientes apropiados
            
            # Generar importe basado en el perfil del cliente (NO aleatorio)
            min_amount, max_amount = profile['typical_range']
            
            # Distribución controlada: 70% en rango bajo, 25% medio, 5% alto del rango
            rand_val = random.random()
            if rand_val < 0.70:
                # Rango bajo del perfil
                amount = round(random.uniform(min_amount, min_amount + (max_amount - min_amount) * 0.3), 2)
            elif rand_val < 0.95:
                # Rango medio del perfil
                amount = round(random.uniform(min_amount + (max_amount - min_amount) * 0.3, 
                                             min_amount + (max_amount - min_amount) * 0.8), 2)
            else:
                # Rango alto del perfil (solo ocasionalmente)
                amount = round(random.uniform(min_amount + (max_amount - min_amount) * 0.8, max_amount), 2)
            
            # Limitar al rango 0-100000
            amount = min(amount, 100000)
            
            # Horario típico del cliente (NO aleatorio)
            hour_min, hour_max = profile['typical_hour_range']
            hour = random.randint(hour_min, hour_max)
            
            timestamp = start_date + timedelta(
                days=random.randint(0, 365),
                hours=hour,
                minutes=random.randint(0, 59)
            )
            
            # País típico del cliente (NO aleatorio)
            origin_country = profile['home_country']
            destination_country = random.choice(profile['typical_countries'])
            
            currency = 'EUR' if origin_country in ['ES', 'FR', 'DE', 'IT'] else random.choice(currencies)
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
        """
        Genera transacciones anómalas siguiendo reglas específicas:
        - Importes > 10000-100000€ marcados como anómalos si:
          * El usuario no suele hacer pagos grandes
          * Se realizan fuera del horario típico del cliente
          * Se hacen desde un país distinto al habitual
        """
        n_customers = max(50, n // 10)
        customer_ids = [f"CUST{str(i).zfill(6)}" for i in range(n_customers)]
        
        transactions = []
        start_date = datetime.now() - timedelta(days=365)
        
        currencies = ['EUR', 'USD', 'GBP', 'CHF']
        channels = ['web', 'app', 'cajero', 'sucursal']
        devices = ['desktop', 'mobile', 'tablet', 'atm', 'pos']
        unusual_countries = ['RU', 'CN', 'KP', 'IR', 'SY', 'AF', 'YE', 'NG', 'PK']
        
        for _ in range(n):
            customer_id = random.choice(customer_ids)
            profile = self._get_customer_profile(customer_id)
            
            # Tipo de anomalía según reglas
            anomaly_type = random.choice([
                'high_amount_unusual_user',      # Usuario que no suele hacer pagos grandes
                'unusual_time',                  # Fuera del horario típico
                'unusual_country',               # País distinto al habitual
                'combined_anomaly'               # Múltiples factores anómalos
            ])
            
            # Inicializar con valores normales del cliente
            min_amount, max_amount = profile['typical_range']
            hour_min, hour_max = profile['typical_hour_range']
            origin_country = profile['home_country']
            destination_country = random.choice(profile['typical_countries'])
            
            # Aplicar reglas de anomalía
            if anomaly_type == 'high_amount_unusual_user':
                # REGLA: Importe alto (>10000€) para usuario que normalmente no paga tanto
                if max_amount < 5000:  # Cliente de gasto bajo/medio
                    amount = round(random.uniform(10000, 100000), 2)
                else:
                    # Si ya es de gasto alto, exceder mucho su rango
                    amount = round(random.uniform(max_amount * 2, 100000), 2)
                    
            elif anomaly_type == 'unusual_time':
                # REGLA: Horario inusual (fuera del patrón del cliente)
                # Madrugada (0-5) o muy tarde (23-24)
                hour = random.choice(list(range(0, 6)) + [23])
                amount = round(random.uniform(max_amount * 0.5, max_amount * 3), 2)
                
            elif anomaly_type == 'unusual_country':
                # REGLA: País distinto al habitual
                destination_country = random.choice(unusual_countries)
                amount = round(random.uniform(max_amount * 0.3, max_amount * 2), 2)
                
            else:  # combined_anomaly
                # REGLA: Combinación de factores anómalos
                amount = round(random.uniform(15000, 100000), 2)
                hour = random.choice(list(range(0, 6)) + [23])
                destination_country = random.choice(unusual_countries)
            
            # Generar timestamp con la hora determinada
            if 'hour' not in locals() or anomaly_type not in ['unusual_time', 'combined_anomaly']:
                hour = random.randint(hour_min, hour_max)
            
            timestamp = start_date + timedelta(
                days=random.randint(0, 365),
                hours=hour,
                minutes=random.randint(0, 59)
            )
            
            currency = 'EUR' if origin_country in ['ES', 'FR', 'DE', 'IT'] else random.choice(currencies)
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
                'is_anomaly': 1
            })
        
        return pd.DataFrame(transactions)
    
    def save_to_csv(self, df, filepath='data/transactions.csv'):
        """Guarda el dataset en formato CSV"""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
        print(f"Dataset guardado en: {filepath}")