from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd

class ModelComparator:
    """Comparacion de multiples modelos de deteccion de anomalias"""
    
    def __init__(self, contamination=0.05, random_state=42):
        self.contamination = contamination
        self.random_state = random_state
        self.models = {}
        self.results = {}
        
    def train_all_models(self, X_train, y_train=None):
        """Entrena multiples modelos de deteccion"""
        print("\n=== ENTRENANDO MULTIPLES MODELOS ===\n")
        
        # 1. ISOLATION FOREST (No Supervisado)
        print("[1/5] Entrenando Isolation Forest...")
        self.models['isolation_forest'] = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=100,
            max_samples='auto'
        )
        self.models['isolation_forest'].fit(X_train)
        
        # 2. LOCAL OUTLIER FACTOR (No Supervisado)
        print("[2/5] Entrenando Local Outlier Factor...")
        self.models['lof'] = LocalOutlierFactor(
            contamination=self.contamination,
            novelty=True,
            n_neighbors=20
        )
        self.models['lof'].fit(X_train)
        
        # Si tenemos etiquetas, entrenar modelos supervisados
        if y_train is not None:
            # 3. RANDOM FOREST (Supervisado)
            print("[3/5] Entrenando Random Forest...")
            self.models['random_forest'] = RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                max_depth=10,
                class_weight='balanced'
            )
            self.models['random_forest'].fit(X_train, y_train)
            
            # 4. REGRESION LOGISTICA (Supervisado)
            print("[4/5] Entrenando Regresion Logistica...")
            self.models['logistic_regression'] = LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                class_weight='balanced'
            )
            self.models['logistic_regression'].fit(X_train, y_train)
            
            # 5. ARBOL DE DECISION (Supervisado)
            print("[5/5] Entrenando Arbol de Decision...")
            self.models['decision_tree'] = DecisionTreeClassifier(
                random_state=self.random_state,
                max_depth=8,
                class_weight='balanced'
            )
            self.models['decision_tree'].fit(X_train, y_train)
        
        print("\n[OK] Todos los modelos entrenados exitosamente\n")
        return self
    
    def predict_all(self, X_test):
        """Genera predicciones con todos los modelos"""
        predictions = {}
        scores = {}
        
        for name, model in self.models.items():
            if name in ['isolation_forest', 'lof']:
                # Modelos no supervisados
                pred = model.predict(X_test)
                predictions[name] = (pred == -1).astype(int)
                scores[name] = -model.score_samples(X_test)
            else:
                # Modelos supervisados
                predictions[name] = model.predict(X_test)
                if hasattr(model, 'predict_proba'):
                    scores[name] = model.predict_proba(X_test)[:, 1]
                else:
                    scores[name] = predictions[name]
        
        return predictions, scores
    
    def cross_validate_model(self, model_name, X, y, cv=5):
        """Validacion cruzada de un modelo especifico"""
        if model_name not in ['random_forest', 'logistic_regression', 'decision_tree']:
            print(f"Validacion cruzada no disponible para {model_name}")
            return None
        
        print(f"\nValidacion cruzada para {model_name} (cv={cv})...")
        
        model = self.models.get(model_name)
        if model is None:
            print(f"Modelo {model_name} no entrenado")
            return None
        
        scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
        
        print(f"Scores por fold: {scores}")
        print(f"Media: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        return scores
    
    def optimize_hyperparameters(self, model_type, X_train, y_train):
        """Optimizacion de hiperparametros usando GridSearch"""
        print(f"\n=== OPTIMIZANDO HIPERPARAMETROS: {model_type} ===\n")
        
        if model_type == 'random_forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15],
                'min_samples_split': [2, 5, 10]
            }
            base_model = RandomForestClassifier(
                random_state=self.random_state,
                class_weight='balanced'
            )
            
        elif model_type == 'decision_tree':
            param_grid = {
                'max_depth': [5, 8, 10, 15],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            base_model = DecisionTreeClassifier(
                random_state=self.random_state,
                class_weight='balanced'
            )
            
        elif model_type == 'logistic_regression':
            param_grid = {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l2'],
                'solver': ['lbfgs', 'liblinear']
            }
            base_model = LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                class_weight='balanced'
            )
        
        elif model_type == 'isolation_forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_samples': ['auto', 0.5, 0.8],
                'contamination': [0.03, 0.05, 0.07]
            }
            base_model = IsolationForest(random_state=self.random_state)
            # Para Isolation Forest necesitamos un enfoque diferente
            print("Nota: Isolation Forest no soporta GridSearchCV estandar")
            return None
        
        else:
            print(f"Optimizacion no disponible para {model_type}")
            return None
        
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=3,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"\nMejores parametros: {grid_search.best_params_}")
        print(f"Mejor score F1: {grid_search.best_score_:.4f}")
        
        # Actualizar el modelo con los mejores parametros
        self.models[model_type] = grid_search.best_estimator_
        
        return grid_search.best_params_, grid_search.best_score_
    
    def compare_models(self, y_true, predictions, scores):
        """Genera tabla comparativa de rendimiento de modelos"""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score
        )
        
        print("\n" + "="*80)
        print(" COMPARACION DE MODELOS")
        print("="*80 + "\n")
        
        results = []
        
        for name in predictions.keys():
            y_pred = predictions[name]
            y_score = scores[name]
            
            try:
                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                
                # AUC solo si tenemos scores continuos
                if len(np.unique(y_true)) > 1:
                    auc = roc_auc_score(y_true, y_score)
                else:
                    auc = 0.0
                
                results.append({
                    'Modelo': name,
                    'Accuracy': accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'F1-Score': f1,
                    'AUC': auc
                })
            except Exception as e:
                print(f"Error evaluando {name}: {e}")
        
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('F1-Score', ascending=False)
        
        print(df_results.to_string(index=False))
        print("\n" + "="*80 + "\n")
        
        # Identificar mejor modelo
        best_model = df_results.iloc[0]['Modelo']
        best_f1 = df_results.iloc[0]['F1-Score']
        print(f">> MEJOR MODELO: {best_model} (F1-Score: {best_f1:.4f})")
        
        self.results = df_results
        return df_results
    
    def get_feature_importance(self, model_name, feature_names):
        """Obtiene importancia de caracteristicas para modelos que lo soportan"""
        model = self.models.get(model_name)
        
        if model is None:
            print(f"Modelo {model_name} no encontrado")
            return None
        
        if not hasattr(model, 'feature_importances_'):
            print(f"Modelo {model_name} no soporta feature_importances_")
            return None
        
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        print(f"\n=== IMPORTANCIA DE CARACTERISTICAS: {model_name} ===\n")
        print(feature_importance_df.to_string(index=False))
        
        return feature_importance_df
