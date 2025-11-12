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
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    numerical_features: List[str],
    binary_features: List[str],
    ordinal_features: List[str],
    nominal_features: List[str]
) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
    """
    Универсальный препроцессинг - принимает УЖЕ разделенные данные
    Не создает признаки, только трансформирует существующие
    """
    
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

# def evaluate_model(
#     model: Any,
#     X_train: np.ndarray,
#     X_test: np.ndarray,
#     y_train: pd.Series,
#     y_test: pd.Series,
#     model_name: str
# ) -> Dict[str, Any]:
#     """Обучает модель и возвращает метрики"""
    
#     model.fit(X_train, y_train)
#     y_pred_log = model.predict(X_test)

#     return evaluate_and_report(y_true_log=y_test, y_pred_log=y_pred_log, model_name=model_name)

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from sklearn.model_selection import cross_validate

def rmsle_on_log_target(y_true_log, y_pred_log):
    """RMSLE напрямую на log-трансформированном таргете"""
    return -np.sqrt(mean_squared_error(y_true_log, y_pred_log))

def mse_on_original_scale(y_true_log, y_pred_log):
    """MSE на оригинальной шкале (после expm1)"""
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    return -mean_squared_error(y_true, y_pred)

def rmse_on_original_scale(y_true_log, y_pred_log):
    """RMSE на оригинальной шкале (после expm1)"""
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    return -np.sqrt(mean_squared_error(y_true, y_pred))

def mae_on_original_scale(y_true_log, y_pred_log):
    """MAE на оригинальной шкале (после expm1)"""
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    return -mean_absolute_error(y_true, y_pred)

def r2_on_original_scale(y_true_log, y_pred_log):
    """R2 на оригинальной шкале (после expm1)"""
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    return r2_score(y_true, y_pred)

# Создаем scorers
scorers = {
    'rmsle': make_scorer(rmsle_on_log_target, greater_is_better=True),
    'mse': make_scorer(mse_on_original_scale, greater_is_better=True),
    'rmse': make_scorer(rmse_on_original_scale, greater_is_better=True),
    'mae': make_scorer(mae_on_original_scale, greater_is_better=True),
    'r2': make_scorer(r2_on_original_scale, greater_is_better=True)
}

def evaluate_model(
    model: Any,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
    model_name: str,
    use_cv: bool = True,
    cv_folds: int = 5
) -> Dict[str, Any]:
    """Обучает модель и возвращает метрики с опциональной кросс-валидацией"""
    
    # Обучаем на полном train
    model.fit(X_train, y_train)
    y_pred_log = model.predict(X_test)
    
    # Базовые метрики на test set через твою функцию
    test_metrics = evaluate_and_report(
        y_true_log=y_test, 
        y_pred_log=y_pred_log, 
        model_name=model_name
    )
    
    # Добавляем кросс-валидацию
    if use_cv:
        print(f"\n{model_name} - Кросс-валидация ({cv_folds} folds)...")
        
        cv_results = cross_validate(
            model, 
            X_train, 
            y_train,
            cv=cv_folds,
            scoring=scorers,
            return_train_score=False,
            n_jobs=-1
        )
        
        # Добавляем CV метрики
        test_metrics['cv_mse_mean'] = -cv_results['test_mse'].mean()
        test_metrics['cv_mse_std'] = cv_results['test_mse'].std()
        test_metrics['cv_rmse_mean'] = -cv_results['test_rmse'].mean()
        test_metrics['cv_rmse_std'] = cv_results['test_rmse'].std()
        test_metrics['cv_mae_mean'] = -cv_results['test_mae'].mean()
        test_metrics['cv_mae_std'] = cv_results['test_mae'].std()
        test_metrics['cv_r2_mean'] = cv_results['test_r2'].mean()
        test_metrics['cv_r2_std'] = cv_results['test_r2'].std()
        test_metrics['cv_rmsle_mean'] = -cv_results['test_rmsle'].mean()
        test_metrics['cv_rmsle_std'] = cv_results['test_rmsle'].std()
        
        print(f"CV MSE: {test_metrics['cv_mse_mean']:.4f} (+/- {test_metrics['cv_mse_std']:.4f})")
        print(f"CV RMSE: {test_metrics['cv_rmse_mean']:.4f} (+/- {test_metrics['cv_rmse_std']:.4f})")
        print(f"CV RMSLE: {test_metrics['cv_rmsle_mean']:.4f} (+/- {test_metrics['cv_rmsle_std']:.4f})")
    
    return test_metrics


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
