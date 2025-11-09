from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import numpy as np

class AnomalyDetector:
    """Entrenador de modelos de detección de anomalías"""
    
    def __init__(self, model_type='isolation_forest', contamination=0.05, random_state=42):
        """
        Args:
            model_type: Tipo de modelo ('isolation_forest' o 'lof')
            contamination: Proporción esperada de anomalías
            random_state: Semilla para reproducibilidad
        """
        self.model_type = model_type
        self.contamination = contamination
        self.random_state = random_state
        self.model = None
        
    def train(self, X):
        """Entrena el modelo de detección de anomalías"""
        print(f"Entrenando modelo: {self.model_type}")
        
        if self.model_type == 'isolation_forest':
            self.model = IsolationForest(
                contamination=self.contamination,
                random_state=self.random_state,
                n_estimators=100,
                max_samples='auto',
                verbose=1
            )
            self.model.fit(X)
            
        elif self.model_type == 'lof':
            self.model = LocalOutlierFactor(
                contamination=self.contamination,
                novelty=True,
                n_neighbors=20
            )
            self.model.fit(X)
        
        else:
            raise ValueError(f"Modelo no soportado: {self.model_type}")
        
        print("Entrenamiento completado")
        return self
    
    def predict(self, X):
        """Predice si las transacciones son anómalas"""
        predictions = self.model.predict(X)
        # Convertir: -1 (anomalía) -> 1, 1 (normal) -> 0
        return (predictions == -1).astype(int)
    
    def score_samples(self, X):
        """Calcula puntuaciones de anomalía para cada muestra"""
        if self.model_type == 'isolation_forest':
            scores = self.model.score_samples(X)
        elif self.model_type == 'lof':
            scores = self.model.score_samples(X)
        else:
            scores = np.zeros(len(X))
        
        # Normalizar para que valores más negativos = más anómalos
        return -scores