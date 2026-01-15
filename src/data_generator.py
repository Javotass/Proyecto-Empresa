import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random

class TransactionDataGenerator:
    """Generador de datos sinteticos de transacciones financieras con reglas controladas"""
    
    # ========== CONSTANTES DE CONFIGURACION ==========
    
    # Monedas disponibles
    CURRENCIES = ['EUR', 'USD', 'GBP', 'CHF']
    
    # Canales de transaccion
    CHANNELS = ['web', 'app', 'cajero', 'sucursal']
    
    # Tipos de dispositivo
    DEVICES = ['desktop', 'mobile', 'tablet', 'atm', 'pos']
    
    # Paises inusuales para deteccion de anomalias
    UNUSUAL_COUNTRIES = ['RU', 'CN', 'KP', 'IR', 'SY', 'AF', 'YE', 'NG', 'PK']
    
    # Distribucion de probabilidades para importes
    LOW_RANGE_PROBABILITY = 0.70    # 70% transacciones en rango bajo
    MEDIUM_RANGE_PROBABILITY = 0.95  # 95% acumulado hasta rango medio
    # El 5% restante sera rango alto
    
    # Multiplicadores de rango
    LOW_RANGE_MULTIPLIER = 0.3
    MEDIUM_RANGE_MIN_MULTIPLIER = 0.3
    MEDIUM_RANGE_MAX_MULTIPLIER = 0.8
    HIGH_RANGE_MIN_MULTIPLIER = 0.8
    
    # Limites de transacciones
    MAX_TRANSACTION_AMOUNT = 100000
    
    # Configuracion de clientes
    MIN_CUSTOMERS = 100
    TRANSACTIONS_PER_CUSTOMER = 50
    ANOMALY_CUSTOMERS_RATIO = 10
    
    # Perfiles de cliente predefinidos
    CUSTOMER_PROFILES = {
        'low_spender': {
            'typical_range': (10, 500),
            'typical_hour_range': (9, 21),
            'typical_countries': ['ES', 'FR', 'IT']
        },
        'medium_spender': {
            'typical_range': (100, 3000),
            'typical_hour_range': (8, 22),
            'typical_countries': ['ES', 'FR', 'DE', 'IT', 'GB']
        },
        'high_spender': {
            'typical_range': (1000, 15000),
            'typical_hour_range': (8, 23),
            'typical_countries': ['ES', 'FR', 'DE', 'IT', 'GB', 'US']
        },
        'business': {
            'typical_range': (500, 25000),
            'typical_hour_range': (7, 19),
            'typical_countries': ['ES', 'FR', 'DE', 'US', 'GB', 'CH']
        }
    }
    
    # Tipos de anomalias
    ANOMALY_TYPES = [
        'high_amount_unusual_user',
        'unusual_time',
        'unusual_country',
        'combined_anomaly'
    ]
    
    # Umbrales para anomalias
    LOW_SPENDER_THRESHOLD = 5000
    HIGH_AMOUNT_MIN = 10000
    UNUSUAL_HOURS = list(range(0, 6)) + [23]
    
    # Configuracion temporal
    DAYS_LOOKBACK = 365
    
    # ========== FIN DE CONSTANTES ==========
    
    def __init__(self, n_transactions=10000, anomaly_ratio=0.05, seed=42):
        """
        Args:
            n_transactions: Numero total de transacciones a generar
            anomaly_ratio: Proporcion de transacciones anomalas (0-1)
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
        customer_types = list(self.CUSTOMER_PROFILES.keys())
        customer_type = random.choice(customer_types)
        
        # Obtener configuracion del perfil
        profile_config = self.CUSTOMER_PROFILES[customer_type]
        
        profile = {
            'type': customer_type,
            'typical_range': profile_config['typical_range'],
            'typical_hour_range': profile_config['typical_hour_range'],
            'typical_countries': profile_config['typical_countries'],
            'home_country': random.choice(profile_config['typical_countries']),
            'transaction_count': 0
        }
        
        self.customer_profiles[customer_id] = profile
        return profile
    
    def _get_customer_profile(self, customer_id):
        """Obtiene o crea el perfil de un cliente"""
        if customer_id not in self.customer_profiles:
            return self._create_customer_profile(customer_id)
        return self.customer_profiles[customer_id]
    
    def generate_dataset(self):
        """Genera el conjunto completo de transacciones"""
        n_anomalies = int(self.n_transactions * self.anomaly_ratio)
        n_normal = self.n_transactions - n_anomalies
        
        print(f"Generando {n_normal} transacciones normales y {n_anomalies} anomalas...")
        
        # Generar transacciones normales
        normal_transactions = self._generate_normal_transactions(n_normal)
        
        # Generar transacciones anomalas
        anomaly_transactions = self._generate_anomaly_transactions(n_anomalies)
        
        # Combinar y mezclar
        all_transactions = pd.concat([normal_transactions, anomaly_transactions], ignore_index=True)
        all_transactions = all_transactions.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        
        # Asignar IDs unicos
        all_transactions['transaction_id'] = [f"TXN{str(i).zfill(8)}" for i in range(len(all_transactions))]
        
        print(f"Dataset generado: {len(all_transactions)} transacciones")
        return all_transactions
    
    def _generate_normal_transactions(self, n):
        """Genera transacciones normales siguiendo reglas controladas"""
        n_customers = max(self.MIN_CUSTOMERS, n // self.TRANSACTIONS_PER_CUSTOMER)
        customer_ids = [f"CUST{str(i).zfill(6)}" for i in range(n_customers)]
        
        transactions = []
        start_date = datetime.now() - timedelta(days=self.DAYS_LOOKBACK)
        
        for _ in range(n):
            customer_id = random.choice(customer_ids)
            profile = self._get_customer_profile(customer_id)
            profile['transaction_count'] += 1
            
            # Generar importe basado en el perfil del cliente
            min_amount, max_amount = profile['typical_range']
            range_span = max_amount - min_amount
            
            # Distribucion controlada usando constantes
            rand_val = random.random()
            if rand_val < self.LOW_RANGE_PROBABILITY:
                # Rango bajo del perfil
                amount = round(random.uniform(
                    min_amount, 
                    min_amount + range_span * self.LOW_RANGE_MULTIPLIER
                ), 2)
            elif rand_val < self.MEDIUM_RANGE_PROBABILITY:
                # Rango medio del perfil
                amount = round(random.uniform(
                    min_amount + range_span * self.MEDIUM_RANGE_MIN_MULTIPLIER,
                    min_amount + range_span * self.MEDIUM_RANGE_MAX_MULTIPLIER
                ), 2)
            else:
                # Rango alto del perfil
                amount = round(random.uniform(
                    min_amount + range_span * self.HIGH_RANGE_MIN_MULTIPLIER,
                    max_amount
                ), 2)
            
            # Limitar al maximo permitido
            amount = min(amount, self.MAX_TRANSACTION_AMOUNT)
            
            # Horario tipico del cliente
            hour_min, hour_max = profile['typical_hour_range']
            hour = random.randint(hour_min, hour_max)
            
            timestamp = start_date + timedelta(
                days=random.randint(0, self.DAYS_LOOKBACK),
                hours=hour,
                minutes=random.randint(0, 59)
            )
            
            # Pais tipico del cliente
            origin_country = profile['home_country']
            destination_country = random.choice(profile['typical_countries'])
            
            # Seleccionar moneda segun el pais
            currency = 'EUR' if origin_country in ['ES', 'FR', 'DE', 'IT'] else random.choice(self.CURRENCIES)
            channel = random.choice(self.CHANNELS)
            device_type = random.choice(self.DEVICES)
            
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
        Genera transacciones anomalas siguiendo reglas especificas:
        - Importes > 10000-100000€ marcados como anomalos si:
          * El usuario no suele hacer pagos grandes
          * Se realizan fuera del horario tipico del cliente
          * Se hacen desde un pais distinto al habitual
        """
        n_customers = max(self.MIN_CUSTOMERS // 2, n // self.ANOMALY_CUSTOMERS_RATIO)
        customer_ids = [f"CUST{str(i).zfill(6)}" for i in range(n_customers)]
        
        transactions = []
        start_date = datetime.now() - timedelta(days=self.DAYS_LOOKBACK)
        
        for _ in range(n):
            customer_id = random.choice(customer_ids)
            profile = self._get_customer_profile(customer_id)
            
            # Tipo de anomalia segun reglas
            anomaly_type = random.choice(self.ANOMALY_TYPES)
            
            # Inicializar con valores normales del cliente
            min_amount, max_amount = profile['typical_range']
            hour_min, hour_max = profile['typical_hour_range']
            origin_country = profile['home_country']
            destination_country = random.choice(profile['typical_countries'])
            hour = random.randint(hour_min, hour_max)  # Inicializar hora por defecto
            
            # Aplicar reglas de anomalia
            if anomaly_type == 'high_amount_unusual_user':
                # REGLA: Importe alto para usuario que normalmente no paga tanto
                if max_amount < self.LOW_SPENDER_THRESHOLD:
                    amount = round(random.uniform(self.HIGH_AMOUNT_MIN, self.MAX_TRANSACTION_AMOUNT), 2)
                else:
                    # Si ya es de gasto alto, exceder mucho su rango
                    amount = round(random.uniform(max_amount * 2, self.MAX_TRANSACTION_AMOUNT), 2)
                    
            elif anomaly_type == 'unusual_time':
                # REGLA: Horario inusual (fuera del patron del cliente)
                hour = random.choice(self.UNUSUAL_HOURS)
                amount = round(random.uniform(max_amount * 0.5, max_amount * 3), 2)
                
            elif anomaly_type == 'unusual_country':
                # REGLA: Pais distinto al habitual
                destination_country = random.choice(self.UNUSUAL_COUNTRIES)
                amount = round(random.uniform(max_amount * 0.3, max_amount * 2), 2)
                
            else:  # combined_anomaly
                # REGLA: Combinacion de factores anomalos
                amount = round(random.uniform(15000, self.MAX_TRANSACTION_AMOUNT), 2)
                hour = random.choice(self.UNUSUAL_HOURS)
                destination_country = random.choice(self.UNUSUAL_COUNTRIES)
            
            # Generar timestamp con la hora determinada
            timestamp = start_date + timedelta(
                days=random.randint(0, self.DAYS_LOOKBACK),
                hours=hour,
                minutes=random.randint(0, 59)
            )
            
            currency = 'EUR' if origin_country in ['ES', 'FR', 'DE', 'IT'] else random.choice(self.CURRENCIES)
            channel = random.choice(self.CHANNELS)
            device_type = random.choice(self.DEVICES)
            
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