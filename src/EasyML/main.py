import copy
import pickle
import warnings
from functools import partial

import lightgbm as lgb
import numpy as np
import optuna
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold
from tqdm import tqdm

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.ERROR)


# TODO: distillationの実装 DTで説明可能性を見たりする。
# TODO: regression, multiclass, multilabel (まずは実験的にマルチクラス、マルチラベルやってみる)
class Main:
    """
    Attributes:
    Methods:
    """
    def __init__(self, input='table', output='binary', algorithm='lgb', metric='auc', keep_input=True):
        """
        Args:
            input (str, optional (default='table')): input data type. 'table' or 'image'.
                # TODO: implement for image task
            output (str, optional (default='binary')): task type. 'binary', 'multiclass', 'multilabel' or  'regression'.
                # TODO: regression, multiclass, multilabel
            algorithm (str, optional (default='lgb')): estimator type. 'lgb', 'cnn' or 'rnn'
                # TODO: cnn, rnn
        """
        self.input = input
        self.output = output
        self.algorithm = algorithm
        self.metric = metric
        self.keep_input = keep_input

        self._estimator_list = []
        self.oof_pred = np.array([])
        self.score = {}
        self.X = np.array([])
        self.y = np.array([])
        self._X_train = np.array([])
        self._y_train = np.array([])
        self._X_val = np.array([])
        self._y_val = np.array([])
        self.X_train_list = []
        self.y_train_list = []
        self.X_val_list = []
        self.y_val_list = []

    def fit(self, X, y=None, cv=5, seed=0, stratify=True, tune_hparams=True, n_trials=128, **kwargs):
        """

        Args:
            X (np.ndarray, str)
            y (np.ndarray, optional,)
        Returns:
        """
        if self.keep_input:
            self.X = X
            self.y = y

        # TODO: regression と multilabelの層別化の追加。qcutを使う?
        if stratify and self.output in ('binary', 'multiclass'):
            kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
        else:
            kf = KFold(n_splits=cv, shuffle=True, random_state=seed)

        self.oof_pred = np.zeros_like(y)
        train_score_list = []
        val_score_list = []

        pbar = tqdm(kf.split(X, y), total=kf.get_n_splits(), desc='[train]')
        for train_idx, val_idx in pbar:
            X_train = X[train_idx]
            y_train = y[train_idx]
            X_val = X[val_idx]
            y_val = y[val_idx]
            # set attributes for other methods.
            self._X_train = X_train
            self._y_train = y_train
            self._X_val = X_val
            self._y_val = y_val
            if self.keep_input:
                self.X_train_list.append(X_train)
                self.y_train_list.append(y_train)
                self.X_val_list.append(X_val)
                self.y_val_list.append(y_val)

            if tune_hparams:
                study = optuna.create_study(direction='maximize', sampler=optuna.samplers.RandomSampler(seed=seed))
                study.optimize(partial(self._objective, X_train, y_train, X_val, y_val), n_trials=n_trials, n_jobs=-1)
                estimator = self._select_estimator(study.best_params)
            else:
                estimator = self._select_estimator()
            estimator.fit(X_train, y_train, **self._select_estimator_fit_params())
            _, train_score = self._predict_score(estimator, X_train, y_train)
            y_val_pred, val_score = self._predict_score(estimator, X_val, y_val)

            pbar.set_postfix(dict(train_score=train_score, val_score=val_score))
            self._estimator_list.append(copy.deepcopy(estimator))
            self.oof_pred[val_idx] = y_val_pred
            train_score_list.append(train_score)
            val_score_list.append(val_score)

        pbar.close()
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

    def save_input(self, path_dir='.', prefix=''):
        common_name = f'{path_dir}/{prefix}_' if prefix else f'{path_dir}/'
        if self.keep_input and self.X is not None:
            with open(f'{common_name}X.pkl', 'wb') as f:
                pickle.dump(self.X, f)
            with open(f'{common_name}y.pkl', 'wb') as f:
                pickle.dump(self.y, f)
            with open(f'{common_name}X_train_holds.pkl', 'wb') as f:
                pickle.dump(self.X_train_list, f)
            with open(f'{common_name}y_train_holds.pkl', 'wb') as f:
                pickle.dump(self.y_train_list, f)
            with open(f'{common_name}X_val_holds.pkl', 'wb') as f:
                pickle.dump(self.X_val_list, f)
            with open(f'{common_name}y_val_holds.pkl', 'wb') as f:
                pickle.dump(self.y_val_list, f)
        else:
            print('There is no input data. Try after changing `keep_input` to True and fitting this model.')

    def distillate(self, type='dt'):
        """Distillate model to DT or ...?
        """
        raise NotImplementedError
        # distillated_estimator = None
        # return distillated_estimator

    def _objective(self, X_train, y_train, X_val, y_val, trial):
        estimator = self._select_estimator(self._select_estimator_hparams(trial))
        estimator.fit(X_train, y_train, **self._select_estimator_fit_params())
        _, score = self._predict_score(estimator, X_val, y_val)
        return score

    def _select_estimator(self, hparams={}):
        if (self.input, self.output, self.algorithm) == ('table', 'binary', 'lgb'):
            return lgb.LGBMClassifier(**hparams)
        elif (self.input, self.output, self.algorithm) == ('table', 'regression', 'lgb'):
            raise NotImplementedError
        elif (self.input, self.output, self.algorithm) == ('table', 'multiclass', 'lgb'):
            raise NotImplementedError
        elif (self.input, self.output, self.algorithm) == ('table', 'multilabel', 'lgb'):
            raise NotImplementedError
        else:
            print('There is no available model for this task. Change `input`, `output`, or `algorithm`.')
            raise NotImplementedError

    def _select_estimator_hparams(self, trial):
        """
        Args:
            trial: use optuna to tune hyperparameters.
                1) categorical variables
                    suggest_categorical(name, choices)
                        e.g.
                        kernel = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf'])
                2) discrete variables
                    suggest_discrete_uniform(name, low, high, 離散値のステップ)
                        e.g.
                        subsample = trial.suggest_discrete_uniform('subsample', 0.1, 1.0, 0.1)
                3) integer variables
                    suggest = int(name, low, high)
                        e.g. n_estimators = trial.suggest_int('n_leaves', 16, 256)
                4) continuous variables
                    suggest_uniform(name, low, high)
                        e.g.
                        c = trial.suggest_loguniform('c', 1e-5, 1e2)
                5) continuous variables (from log uniform distribution)
                    uggest_loguniform(name, low, high)
                        e.g.
                        c = trial.suggest_loguniform('c', 1e-5, 1e2)
                6) float variables
                    suggest_float(name, low, high, step, log)
                        e.g.
                        trial.suggest_float('momentum', 0.0, 1.0)
                        trial.suggest_float('power_t', 0.2, 0.8, step=0.1)
                        trial.suggest_float('learning_rate_init',1e-5, 1e-3, log=True)

        Returns:
            (dict) : hiperparameters of estimator.
        """
        if (self.input, self.output, self.algorithm) == ('table', 'binary', 'lgb'):
            return {
                'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
                'num_leaves': trial.suggest_int('num_leaves', 8, 256),
                'min_child_samples': trial.suggest_int('min_child_samples', 2, 128),
                'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.3),
                'n_estimators': 1000
            }
        elif (self.input, self.output, self.algorithm) == ('table', 'regression', 'lgb'):
            raise NotImplementedError
        elif (self.input, self.output, self.algorithm) == ('table', 'multiclass', 'lgb'):
            raise NotImplementedError
        elif (self.input, self.output, self.algorithm) == ('table', 'multilabel', 'lgb'):
            raise NotImplementedError
        else:
            print('There is no available model for this task. Change `input`, `output`, or `algorithm`.')
            raise NotImplementedError

    def _select_estimator_fit_params(self):
        """
        Returns:
            (dict) : additional arguments of estimator's `fit` method.
        """
        if (self.input, self.output, self.algorithm) == ('table', 'binary', 'lgb'):
            return {
                'eval_set': [(self._X_val, self._y_val)],
                'early_stopping_rounds': 100,
                'verbose': False
            }
        elif (self.input, self.output, self.algorithm) == ('table', 'regression', 'lgb'):
            raise NotImplementedError
        elif (self.input, self.output, self.algorithm) == ('table', 'multiclass', 'lgb'):
            raise NotImplementedError
        elif (self.input, self.output, self.algorithm) == ('table', 'multilabel', 'lgb'):
            raise NotImplementedError
        else:
            print('There is no available model for this task. Change `input`, `output`, or `algorithm`.')
            raise NotImplementedError

    def _predict_score(self, estimator, X, y=None):
        if self.metric == 'auc':
            y_pred = estimator.predict_proba(X)[:, 1]  # output of predict_proba is a 2d array.
            score = roc_auc_score(y, y_pred) if y is not None else None
            return y_pred, score
        if self.metric == 'accuracy':
            y_pred = estimator.predict(X)
            score = accuracy_score(y, y_pred) if y is not None else None
            return y_pred, score
        else:
            print('There is no available metric. Change `metric`.')
            raise NotImplementedError
