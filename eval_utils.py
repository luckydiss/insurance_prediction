"""
Утилиты для оценки и визуализации результатов машинного обучения.
Содержит функции для оценки моделей и построения графиков.
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import cm


from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

from typing import List, Optional, Tuple, Union, Dict, Any

from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, 
    mean_squared_log_error, r2_score
)

def evaluate_and_report(
    y_true_log: Union[np.ndarray, pd.Series, List],
    y_pred_log: Union[np.ndarray, pd.Series, List],
    model_name: str,
    target_names: Optional[List[str]] = None,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Оценивает модель и выводит детальный отчет с метриками.
    """
    # Преобразуем в numpy arrays для совместимости
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    rmsle = np.sqrt(mean_squared_error(y_true_log, y_pred_log))
    
    metrics = {
        'model': model_name,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'rmsle': rmsle
    }
    print(metrics)
    
    return metrics


def plot_prediction_distr(
    y_true: Union[np.ndarray, pd.Series, List],
    y_pred: Union[np.ndarray, pd.Series, List],
    model_name: str,
    figsize: Tuple[int, int] = (18, 5), 
) -> None:
    """
    Строит распределение истинных и предсказанных значений
    и график остатков
    """

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    viridis_cmap = cm.get_cmap("viridis", 3)

    axes[0].hist(
        y_true,
        bins=30,
        color=viridis_cmap(0),
        alpha=0.6,
        edgecolor="black",
        label="Истинные значения",
    )
    axes[0].hist(
        y_pred,
        bins=30,
        color=viridis_cmap(0.5),
        alpha=0.6,
        edgecolor="black",
        label=f"Предсказанные значения ({model_name})",
    )
    axes[0].set_title(
        "Распределение предсказаний",
        fontsize=14,
    )
    axes[0].set_xlabel("Логарифмированная сумма премии (log(y + 1))", fontsize=12)
    axes[0].set_ylabel("Частота", fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, linestyle="--", alpha=0.6)

    residuals = y_true - y_pred

    sns.scatterplot(
        x=y_pred,
        y=residuals,
        alpha=0.6,
        color="#007acc",
        ax=axes[1]
    )
    axes[1].axhline(y=0, color="red", linestyle="--", linewidth=1.5)
    axes[1].set_title(
        "Остатки vs Предсказанные значения",
        fontsize=14,
        fontweight="bold",
    )
    axes[1].set_xlabel("Предсказанные значения", fontsize=12)
    axes[1].set_ylabel("Остатки", fontsize=12)
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.show()

def plot_feature_importance(
    model: Any,
    feature_names: Optional[List[str]] = None,
    top_n: int = 30,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> None:
    """
    Строит график важности признаков для обученной модели.
    """

    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        importance_type = "Feature Importance (Tree-based)"
    elif hasattr(model, 'coef_'):
        coef = np.ravel(model.coef_)
        importance = np.abs(coef)
        importance_type = "Coefficient Magnitude (Linear Model)"
    else:
        raise ValueError("Модель должна иметь атрибут 'feature_importances_' или 'coef_'")

    n_features = len(importance)

    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(n_features)]
    elif len(feature_names) != n_features:
        print(f"Предупреждение: Количество имен признаков ({len(feature_names)}) "
              f"не совпадает с количеством признаков ({n_features}). Используются автоматически сгенерированные имена.")
        feature_names = [f'Feature_{i}' for i in range(n_features)]

    importance_df = (
        pd.DataFrame({'feature': feature_names, 'importance': importance})
        .sort_values('importance', ascending=True)
        .tail(top_n)
    )

    plt.figure(figsize=figsize)
    bars = plt.barh(range(len(importance_df)), importance_df['importance'])
    plt.yticks(range(len(importance_df)), importance_df['feature'])
    plt.xlabel(importance_type)
    plt.title(title or f'Top {top_n} {importance_type}')
    plt.grid(axis='x', alpha=0.3)

    for bar in bars:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height() / 2,
                 f'{width:.4f}', ha='left', va='center', fontsize=9)

    plt.tight_layout()
    plt.show()


def compare_models_metrics(
    metrics_list: List[Dict[str, Any]],
    metrics_to_compare: List[str] = ['mse', 'rmse', 'mae', 'r2', 'rmsle'],
    figsize: Tuple[int, int] = (10, 6)
) -> pd.DataFrame:
    """
    Сравнивает метрики нескольких моделей в виде таблицы и графика.
    """
    metrics_df = pd.DataFrame(metrics_list)
    
    plt.figure(figsize=figsize)
    
    heatmap_data = metrics_df.set_index('model')[metrics_to_compare]
    
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdBu_r', 
                cbar=True, center=0.5)
    plt.title('Сравнение метрик моделей')
    plt.tight_layout()
    
    plt.show()
    
    return metrics_df


def compare_after_add(new_df: pd.DataFrame, 
                      base_df: pd.DataFrame, 
                      title: str, 
                      figsize: Tuple[int, int] = (10, 6),
                      normalize: bool = True) -> pd.DataFrame:
    """
    Сравнивает метрики моделей относительно предыдущих результатов
    """
    df_diff = new_df.set_index('model') - base_df.set_index('model')
    
    if normalize:
        df_diff_norm = (df_diff - df_diff.mean()) / df_diff.std()
    else:
        df_diff_norm = df_diff
    
    colors = ["Green", "White", "Red"]
    cmap = LinearSegmentedColormap.from_list("gwr", colors)
    
    plt.figure(figsize=figsize)
    sns.heatmap(df_diff_norm, annot=df_diff, fmt='.3f',
                cmap=cmap, cbar=True, center=0)
    plt.title(title)
    plt.tight_layout()
    
    return df_diff
