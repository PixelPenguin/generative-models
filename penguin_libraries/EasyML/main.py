import pickle
import time
import warnings
from functools import partial

import lightgbm as lgb
import numpy as np
import optuna
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold

from . import utils

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.ERROR)


class Main:
    """

    Attributes:
        Main.available_task (list): available combinations of input, output, algorithm and metric.

    Methods:
        fit:
        predict:
        predict_proba:
        save_input: save data used to fit estimators.
        distillate: distillate model to DecisonTree by LIME.
        visualize: visualize model by boundary or contour in 2d coordinate system with dimention reduction.
    """
    available_task = [
        ('table', 'binary', 'lgb', 'auc'),
        ('table', 'binary', 'lgb', 'acc')
    ]

    def __init__(self, input='table', output='binary', algorithm='lgb', metric='auc', keep_input=True, random_state=0):
        """
        Args:
            input (str (default='table')): input data type. 'table'
                # TODO: implement for image task
            output (str (default='binary')): task type. 'binary'
                # TODO: regression, multiclass, multilabel
            algorithm (str (default='lgb')): estimator type. 'lgb'
                # TODO: cnn, rnn
            metric (str (default='auc')): metric type. 'auc' or 'acc'
                # TODO: rmse for regression
            keep_input (bool, (default=True)): keep input data as attributes.
            random_state (int): random seed.
        """
        if not (input, output, algorithm, metric) in Main.available_task:
            raise NotImplementedError
        self.input = input
        self.output = output
        self.algorithm = algorithm
        self.metric = metric
        self.keep_input = keep_input
        self.random_state = random_state

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

    # TODO: fit with minibatch for large data.
    def fit(self, X, y=None, cv=5, stratify=True, tune_hparams=True, n_trials=128, timeout=3600, verbose=True):
        """select proper ml model(s) then train them and tune hyperparameters.

        Args:
            X (np.ndarray, list): 2d array for input='table', 4d array or list of path/to/images for input='image'.
            y (np.ndarray): target variable.
            e.g.)
                if self.output == 'binary':
                    y.shape == (len(X),)  # each value is 0 or 1.
                elif self.output == 'regression':
                    y.shape == (len(X),)  # each value is float.
                elif self.output == 'multiclass':
                    y.shape == (len(X),)  # each value (integer) is zero or more and less than the num_classes.
                elif self.output == 'multilabel':
                    y.shape == (len(X), num_classes)  # one-hot encoding.
                    # e.g.) the first row y_pred[0, :] have the labels of the class 1 and 2.
                    # y_pred = array([
                    #     [0, 1, 1],
                    #     [0, 1, 0]
                    # ])
            cv (int): the number of cross validation.
            stratify (bool): flag to use `StratifiedKFold` instead of `KFold`.
            tune_hparams (bool): flag to tune hypterparameters by Optuna or not.
            n_trials (bool): if `tune_params` is True, hyperparameters are searched this number of times.
            timeout (int): stop study after the given number of second(s).
        """
        print(
            f'Training params: cv={cv}, stratify={stratify}, tune_hparams={tune_hparams}, '
            f'n_trials={n_trials}, timeout={timeout}'
        ) if verbose else None
        if self.keep_input:
            self.X = X
            self.y = y

        # TODO: stratify for regression and multilabel.
        if stratify and self.output in ('binary', 'multiclass'):
            kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        else:
            kf = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)

        self.oof_pred = np.zeros_like(y)
        train_score_list = []
        val_score_list = []

        for i, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            print(f'Fold #{i} | ', end='') if verbose else None
            fold_start = time.time()

            X_train, y_train, X_val, y_val = X[train_idx], y[train_idx], X[val_idx], y[val_idx]
            # set attributes for validation and tuning.
            self._X_train, self._y_train, self._X_val, self._y_val = X_train, y_train, X_val, y_val
            if self.keep_input:
                self.X_train_list.append(X_train)
                self.y_train_list.append(y_train)
                self.X_val_list.append(X_val)
                self.y_val_list.append(y_val)

            if tune_hparams:
                study = optuna.create_study(
                    direction='maximize', sampler=optuna.samplers.RandomSampler(seed=self.random_state)
                )
                study.optimize(
                    partial(self._objective, X_train, y_train, X_val, y_val),
                    n_trials=n_trials, timeout=timeout / cv, n_jobs=-1
                )
                estimator = self._select_estimator(study.best_params)
            else:
                estimator = self._select_estimator()
            estimator.fit(X_train, y_train, **self._select_estimator_fit_params())
            _, train_score = self._predict_score(estimator, X_train, y_train)
            y_val_pred, val_score = self._predict_score(estimator, X_val, y_val)

            self._estimator_list.append(utils.deepcopy_without_verbose(estimator))
            self.oof_pred[val_idx] = y_val_pred
            train_score_list.append(train_score)
            val_score_list.append(val_score)

            if verbose:
                print(f'Elapsed time: {time.time() - fold_start:<4.1f} [sec] | ', end='')
                print(f'Train {self.metric}: {train_score:<5.3f} | ', end='')
                print(f'Valid {self.metric}: {val_score:<5.3f}')

        self.score = {
            'train': (np.mean(train_score_list), np.std(train_score_list)),
            'val': (np.mean(val_score_list), np.std(val_score_list)),
        }
        return self

    def predict(self, X, y=None, threshold=0.5):
        """Predict classes or values like sklearn.

        Args:
            X (np.ndarray or list): 2d array for input=`table`, 4d array or path to images for input=`image`.
            threshold (float): threshold to be classified as class 1 for binary and multilabel classification.

        Returns:
            (np.ndarray): predicted y.

        Examples:
            y_pred = self.predict(X)
            if self.output == 'binary':
                y_pred.shape == (len(y),)  # each value is 0 or 1.
            elif self.output == 'regression':
                y_pred.shape == (len(y),)  # each value is float.
            elif self.output == 'multiclass':
                y_pred.shape == (len(y),)  # each value (integer) is zero or more and less than the num_classes.
            elif self.output == 'multilabel':
                y_pred.shape == y.shape  # one-hot encoding.
                # e.g.) the first row y_pred[0, :] have the labels of the class 1 and 2.
                # y_pred = array([
                #     [0, 1, 1],
                #     [0, 1, 0]
                # ])
        """
        if self.output == 'binary':
            y_pred_list = [estimator.predict_proba(X)[:, 1] for estimator in self._estimator_list]
            y_pred_proba = np.stack(y_pred_list).mean(axis=0)
            return np.where(y_pred_proba > threshold, 1, 0)
        elif self.output == 'regression':
            y_pred_list = [estimator.predict(X) for estimator in self._estimator_list]
            return np.stack(y_pred_list).mean(axis=0)

    def predict_proba(self, X, y=None):
        """Predict probability like sklearn.

        Args:
            X (np.ndarray or list): 2d array for input=`table`, 4d array or path to images for input=`image`.

        Returns:
            (np.ndarray): predicted probability of y.

        Examples:
            y_pred = self.predict_proba(X)
            if self.output == 'binary':
                y_pred.shape == (len(y), 2)
            elif self.output == 'multiclass':
                y_pred.shape == (len(y), len(np.unique(y)))
            elif self.output == 'multilabel':
                y_pred.shape == y.shape
        """
        y_pred_list = []
        for estimator in self._estimator_list:
            y_pred_list.append(estimator.predict_proba(X))
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

    # TODO:
    def distillate(self, type='dt'):
        """Distillate model to DecisionTree.
        """
        raise NotImplementedError

    # TODO:
    def visualize(self, reduction='umap'):
        """Visualize model by boundary or contour in 2d coodinate system.
        """
        raise NotImplementedError

    def _objective(self, X_train, y_train, X_val, y_val, trial):
        """Optimize function for Optuna.

        Returns:
            score (float): scalar to maximize.
        """
        estimator = self._select_estimator(self._select_estimator_hparams(trial))
        estimator.fit(X_train, y_train, **self._select_estimator_fit_params())
        _, score = self._predict_score(estimator, X_val, y_val)
        return score

    def _select_estimator(self, hparams={}):
        """Select proper estimator for the task.

        Returns:
            (Classifier, Regressor): sklearn api model of lightgbm, sklearn, or pytorch.
        """
        if (self.input, self.output, self.algorithm) == ('table', 'binary', 'lgb'):
            return lgb.LGBMClassifier(**hparams)
        elif (self.input, self.output, self.algorithm) == ('table', 'regression', 'lgb'):
            raise NotImplementedError
        elif (self.input, self.output, self.algorithm) == ('table', 'multiclass', 'lgb'):
            raise NotImplementedError
        elif (self.input, self.output, self.algorithm) == ('table', 'multilabel', 'lgb'):
            raise NotImplementedError

    def _select_estimator_hparams(self, trial):
        """Select hyperparameters of eash estimator for Optuna.

        Args:
            trial: use optuna to tune hyperparameters.

        Returns:
            (dict) : hiperparameters of estimator.

        Examples:
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
        """
        if (self.input, self.output, self.algorithm) == ('table', 'binary', 'lgb'):
            return {
                'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
                'num_leaves': trial.suggest_int('num_leaves', 8, 256),
                'min_child_samples': trial.suggest_int(
                    'min_child_samples',  # this hypterparamter has a great effect on the time to fit model(s).
                    max(int(np.log(len(self._X_train))), 8), min(int(np.sqrt(len(self._X_train))), 128)
                ),
                'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.3),
                'n_estimators': 1000,
                'verbose': -1,
                'random_state': self.random_state,
            }
        elif (self.input, self.output, self.algorithm) == ('table', 'regression', 'lgb'):
            raise NotImplementedError
        elif (self.input, self.output, self.algorithm) == ('table', 'multiclass', 'lgb'):
            raise NotImplementedError
        elif (self.input, self.output, self.algorithm) == ('table', 'multilabel', 'lgb'):
            raise NotImplementedError

    def _select_estimator_fit_params(self):
        """Select training parameters for the fit method.

        Returns:
            (dict) : additional arguments of estimator's `fit` method.
        """
        if (self.input, self.output, self.algorithm) == ('table', 'binary', 'lgb'):
            return {
                'eval_set': [(self._X_val, self._y_val)],
                'early_stopping_rounds': 100,
                'verbose': False,
            }
        elif (self.input, self.output, self.algorithm) == ('table', 'regression', 'lgb'):
            raise NotImplementedError
        elif (self.input, self.output, self.algorithm) == ('table', 'multiclass', 'lgb'):
            raise NotImplementedError
        elif (self.input, self.output, self.algorithm) == ('table', 'multilabel', 'lgb'):
            raise NotImplementedError

    def _predict_score(self, estimator, X, y=None):
        """Predict hold-out data for cross validation and calculate score for Optuna to tune model.

        Returns:
            y_pred (np.ndarray): predicted (probability of) hold-out y.
            score (float): calculated score of hold-out data.
        """
        if self.metric == 'auc':
            y_pred = estimator.predict_proba(X)[:, 1]  # output of predict_proba is a 2d array.
            score = roc_auc_score(y, y_pred) if y is not None else None
            return y_pred, score
        if self.metric == 'accuracy':
            y_pred = estimator.predict(X)
            score = accuracy_score(y, y_pred) if y is not None else None
            return y_pred, score
