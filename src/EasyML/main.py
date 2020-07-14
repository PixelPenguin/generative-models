import copy
from functools import partial

import lightgbm as lgb
import numpy as np
import optuna
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold


# DLの場合は挙動が違うので、分けて、Main, EasyML, EasyDLとする......?
# TODO: regression, multiclass, multilabel (まずは実験的にマルチクラス、マルチラベルやってみる)
# TODO: テスト書く、irisで動くか確かめる。
class Main:
    """
    Attributes:
    Methods:
    """
    def __init__(self, input='table', output='binary', algorithm='lgb', metric='auc', data_save=None, model_save=None):
        """
        """
        self.input = input
        self.output = output
        self.algorithm = algorithm
        self.metric = metric
        self.data_save = data_save
        self.model_save = model_save

        self._estimator_list = []
        self.oof_pred = np.array()
        self.score = {}
        self.X = None
        self.y = np.array()
        self.X_train_list = []
        self.y_train_list = []
        self.X_val_list = []
        self.y_val_list = []

    def fit(self, X, y=None, cv=5, stratify=True, tune_hparams=True):
        """

        Args:
            X (np.ndarray, str)
            y (np.ndarray, optional,)
        Returns:
        """
        if self.data_save:
            self.X = X
            self.y = y

        if stratify:
            kf = StratifiedKFold(n_splits=cv)
        else:
            kf = KFold(n_splits=cv)

        self.oof_pred = np.zeros_like(y)
        train_score_list = []
        val_score_list = []

        for train_idx, val_idx in kf.split(X, y):
            X_train = X[train_idx]
            y_train = y[train_idx]
            X_val = X[val_idx]
            y_val = y[val_idx]
            if self.data_save:
                self.X_train_list.append(X_train)
                self.y_train_list.append(y_train)
                self.X_val_list.append(X_val)
                self.y_val_list.append(y_val)

            if tune_hparams:
                study = optuna.create_study(direction='maximize', sampler=optuna.samplers.RandomSampler(seed=0))
                study.optimize(partial(self._objective, X_train, y_train, X_val, y_val), n_trials=100)
                estimator = self._select_estimator(**study.best_params)
            else:
                estimator = self._select_estimator()
            estimator.fit(X_train, y_train, **self._select_estimator_training_params())
            _, train_score = self._predict_score(estimator, X_train, y_train)
            y_val_pred, val_score = self._predict_score(estimator, X_val, y_val)

            self._estimator_list.append(copy.deepcopy(estimator))
            self.oof_pred[val_idx] = y_val_pred
            train_score_list.append(train_score)
            val_score_list.append(val_score)

        self.score = {
            'train': (np.mean(train_score_list), np.std(train_score_list)),
            'val': (np.mean(val_score_list), np.std(val_score_list)),
        }

    def predict(self, X, y=None):
        y_pred_list = []
        for estimator in self._estimator_list:
            y_pred, _ = self._predict_score(estimator, X, y)
            y_pred_list.append(y_pred)
        return np.stack(y_pred_list).mean(axis=0)

    def _objective(self, X_train, y_train, X_val, y_val, trial):
        estimator = self._select_estimator(self._select_estimator_hparams(trial))
        estimator.fit(X_train, y_train, **self._select_estimator_training_params())
        _, score = self._predict_score(estimator, X_val, y_val)
        return score

    def _select_estimator(self, hparams={}):
        if (self.input, self.output, self.algorithm) == ('table', 'binary', 'lgb'):
            return lgb.LGBMClassifier(**hparams)
        else:
            print('There is no available model for this task. Change `input`, `output`, or `algorithm`.')
            raise NotImplementedError

    def _select_estimator_hparams(self, trial):
        """
        # カテゴリ変数
        suggest_categorical(name, choices)
        # カテゴリ変数 例
        kernel = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf'])

        # 離散パラメータ
        suggest_discrete_uniform(name, low, high, 離散値のステップ)
        # 離散パラメータ 例
        subsample = trial.suggest_discrete_uniform('subsample', 0.1, 1.0, 0.1)

        # 整数パラメータ
        suggest_int（name, low, high）
        # 整数パラメータ 例
        n_estimators = trial.suggest_int('n_estimators', 50, 400)

        # 連続パラメータ(log)
        suggest_loguniform（name, low, high）
        # 連続パラメータ(log) 例
        c = trial.suggest_loguniform('c', 1e-5, 1e2)

        # 連続パラメータ
        suggest_uniform（name, low, high）
        # 連続パラメータ 例
        dropout_rate = trial.suggest_uniform('dropout_rate', 0, 1.0)

        # 小数パラメータ(ver1.3.0より実装）
        suggest_float(name, low, high, )
        # 小数パラメータ 例
        trial.suggest_float('momentum', 0.0, 1.0)
        trial.suggest_float('power_t', 0.2, 0.8, step=0.1) # 離散化ステップの設定
        trial.suggest_float('learning_rate_init',1e-5, 1e-3, log=True) # logで設定
        """
        if (self.input, self.output, self.algorithm) == ('table', 'binary', 'lgb'):
            return {
                'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
                'num_leaves': trial.suggest_int('num_leaves', 8, 256),
                'min_child_samples': trial.suggest_int('min_child_samples', 2, 128),
                'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.3),
                'n_estimators': 1000
            }
        else:
            print('There is no available model for this task. Change `input`, `output`, or `algorithm`.')
            raise NotImplementedError

    def _select_estimator_training_params(self):
        if (self.input, self.output, self.algorithm) == ('table', 'binary', 'lgb'):
            return {
                'early_stopping_rounds': 100,
                'verbose': False
            }
        else:
            print('There is no available model for this task. Change `input`, `output`, or `algorithm`.')
            raise NotImplementedError

    def _predict_score(self, estimator, X, y=None):
        if self.metric == 'auc':
            y_pred = estimator.predict_proba(X)
            score = roc_auc_score(y, y_pred) if y is not None else None
            return y_pred, score
        if self.metric == 'accuracy':
            y_pred = estimator.predict(X)
            score = accuracy_score(y, y_pred) if y is not None else None
            return y_pred, score
        else:
            print('There is no available metric. Change `metric`.')
            raise NotImplementedError
