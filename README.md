Проект по прогнозированию размера страховой премии (того сколько клиент заплатит за страховку) на основе сгенерированного датасета с Kaggle: Regression with an Insurance Dataset
([https://www.kaggle.com/datasets/kartik2112/fraud-detection/data](https://www.kaggle.com/competitions/playground-series-s4e12))

Проект включает несколько этапов:

EDA и построение бейзлайн моделей (baseline.ipynb)

Первичный анализ данных
Построение бейзлайн моделей
Feature engineering (improvements.ipynb)

Создание новых признаков на основе бинаризации и создания пар взаимодействий числовых признаков.
Тестирование моделей с учетом добавления новых признаков
Подбор гиперпараметров (hyperparam.ipynb)

Оптимизация параметров моделей с использованием GridSearchCV и BayesSearchCV.
Выбор оптимальных порогов классификации на основе RMSLE
