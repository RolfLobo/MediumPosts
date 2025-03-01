import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt
from cycler import cycler
plt.rcParams['axes.prop_cycle'] = cycler(color=['#118AB2', '#b13e2b', '#39b26c', '#dfdf07', '#852293'])

# Defining Extra Metrics
def fp(y_true, y_pred):
    return np.sum(np.logical_and(y_pred == 1, y_true == 0))

def tp(y_true, y_pred):
    return np.sum(np.logical_and(y_pred == 1, y_true == 1))

def tn(y_true, y_pred):
    return np.sum(np.logical_and(y_pred == 0, y_true == 0))

def fn(y_true, y_pred):
    return np.sum(np.logical_and(y_pred == 0, y_true == 1))

# Define Simple models
class BinaryMean():
    @staticmethod
    def run_benchmark(df_train, df_test):
        np.random.seed(21)
        return np.random.choice(a=[1, 0], size=len(df_test), p=[df_train['y'].mean(), 1 - df_train['y'].mean()])
    
class SimpleXbg():
    @staticmethod
    def run_benchmark(df_train, df_test):
        model = xgb.XGBClassifier()
        model.fit(df_train.select_dtypes(include=np.number).drop(columns='y'), df_train['y'])
        return model.predict(df_test.select_dtypes(include=np.number).drop(columns='y'))
    
class MajorityClass():
    @staticmethod
    def run_benchmark(df_train, df_test):
        majority_class = df_train['y'].mode()[0]
        return np.full(len(df_test), majority_class)

class SimpleKNN():
    @staticmethod
    def run_benchmark(df_train, df_test):
        model = KNeighborsClassifier()
        model.fit(df_train.select_dtypes(include=np.number).drop(columns='y'), df_train['y'])
        return model.predict(df_test.select_dtypes(include=np.number).drop(columns='y'))
    
class RandomOrdered():
    @staticmethod
    def run_benchmark(df_train, df_test):
        np.random.seed(21)
        df = df_test.sample(n=len(df_test), replace = False).reset_index(drop=True)
        return df.index / len(df)

class SimpleXbgProba():
    @staticmethod
    def run_benchmark(df_train, df_test):
        model = xgb.XGBClassifier()
        model.fit(df_train.select_dtypes(include=np.number).drop(columns='y'), df_train['y'])
        return [x[1] for x in model.predict_proba(df_test.select_dtypes(include=np.number).drop(columns='y'))]

# Defining Benchmark class
class ChurnBinaryBenchmark():
    def __init__(
        self,
        metrics = [f1_score],
        benchmark_models = [BinaryMean],
        ):
        self.metrics = metrics
        self.benchmark_models = benchmark_models

    def compare_pred_with_benchmark(
        self,
        df_train,
        df_test,
        my_predictions,
    ):
        
        output_metrics = {
            'Prediction': self._calculate_metrics(df_test['y'], my_predictions)
        }

        dct_benchmarks = {}
        for model in self.benchmark_models:
            dct_benchmarks[model.__name__] = model.run_benchmark(df_train = df_train, df_test = df_test)
            output_metrics[f'Benchmark - {model.__name__}'] = self._calculate_metrics(df_test['y'], dct_benchmarks[model.__name__])

        return output_metrics
    
    def _calculate_metrics(self, y_true, y_pred):
        return {getattr(func, '__name__', 'Unknown') : func(y_true = y_true, y_pred = y_pred) for func in self.metrics}


class ChurnProbaBenchmark():
    def __init__(
        self,
        metrics = [roc_auc_score],
        benchmark_models = [RandomOrdered],
        ):
        self.metrics = metrics
        self.benchmark_models = benchmark_models

    def compare_pred_with_benchmark(
        self,
        df_train,
        df_test,
        my_predictions,
        plots = False
    ):
        
        output_metrics = {
            'Prediction': self._calculate_metrics(df_test['y'], my_predictions)
        }

        self.dct_benchmarks = {
            'Prediction': my_predictions
        }

        for model in self.benchmark_models:
            self.dct_benchmarks[model.__name__] = model.run_benchmark(df_train = df_train, df_test = df_test)
            output_metrics[f'{model.__name__}'] = self._calculate_metrics(df_test['y'], self.dct_benchmarks[model.__name__])

        if plots:
            for key, item in output_metrics.items():
                fpr, tpr, _ = roc_curve(df_test['y'], self.dct_benchmarks[key])
                plt.plot(fpr, tpr, label = f'{key} - AUC: {np.round(roc_auc_score(df_test['y'], self.dct_benchmarks[key]),3)}')
            plt.legend(loc='center left', bbox_to_anchor = (1, 0.5))
            plt.show

        return output_metrics
    
    def _calculate_metrics(self, y_true, y_pred):
        return {getattr(func, '__name__', 'Unknown') : func(y_true, y_pred) for func in self.metrics}
