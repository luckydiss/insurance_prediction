from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    OrdinalEncoder
)
from sklearn.impute import SimpleImputer

from eval_utils import evaluate_and_report

def get_baseline_preprocessing(
    df: pd.DataFrame,
    numerical_features: List[str],
    binary_features: List[str],
    ordinal_features: List[str],
    nominal_features: List[str]
) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
    """Базовый препроцессинг с time-based split и FrequencyEncoder"""
    
    X = df.drop(['Premium_Amount','Policy_Start_Date'], axis=1, errors='ignore')
    y = np.log1p(df['Premium_Amount'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), numerical_features),
    
    ('bin_ord', Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ]), binary_features + ordinal_features),
    
    ('nom', Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
    ]), nominal_features)
    ], remainder='drop')

    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    return X_train_proc, X_test_proc, y_train, y_test

def evaluate_model(
    model: Any,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
    model_name: str
) -> Dict[str, Any]:
    """Обучает модель и возвращает метрики"""
    
    model.fit(X_train, y_train)
    y_pred_log = model.predict(X_test)

    return evaluate_and_report(y_true_log=y_test, y_pred_log=y_pred_log, model_name=model_name)


def get_baseline_models() -> Dict[str, Any]:
    """Возвращает словарь бейслайн моделей"""

    return {
        'DecisionTreeRegressor': DecisionTreeRegressor(
            max_depth=10,    
            min_samples_leaf=5,  
            random_state=42
        ),
        'RandomForestRegressor' : RandomForestRegressor(
            max_depth = 5, 
            n_estimators=10, 
            random_state=42
        ),
        'XGBoost': XGBRegressor(
            objective='reg:squarederror',
            eval_metric='rmse',   
            n_estimators=20, 
            max_depth=6, 
            learning_rate=0.1,
            random_state=42,
        ),
        'LightGBM': LGBMRegressor(
            objective='regression',
            metric='rmse',
            n_estimators=20,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
        )
    }


def run_experiment(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
    experiment_name: str
) -> List[Dict[str, Any]]:
    """Запускает все модели и возвращает метрики"""

    print(f"\n{'='*60}")
    print(f"ЭКСПЕРИМЕНТ: {experiment_name}")
    print(f"{'='*60}")
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    models = get_baseline_models()
    results: List[Dict[str, Any]] = []
    
    for name, model in models.items():
        metrics = evaluate_model(model, X_train, X_test, y_train, y_test, name)
        results.append(metrics)
    
    return results
