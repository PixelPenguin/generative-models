import numpy as np
import pytest
from sklearn.datasets import load_diabetes, load_iris, load_wine  # TODO: add multilabel

from EasyML import main


@pytest.fixture(scope='module', autouse=True)
def toy_dataset():
    dataset = {
        'table': {
            'binary': {
                'X': np.array([]),
                'y': np.array([])
            },
            'regression': {
                'X': np.array([]),
                'y': np.array([])
            },
            'multiclass': {
                'X': np.array([]),
                'y': np.array([])
            },
            'multilabel': {
                'X': np.array([]),
                'y': np.array([])
            }
        },
        'image': {}
    }

    iris = load_iris()
    dataset['table']['binary']['X'] = iris.data
    dataset['table']['binary']['y'] = np.where(iris.target > 0, 1, 0)

    diabetes = load_diabetes()
    dataset['table']['regression']['X'] = diabetes.data
    dataset['table']['regression']['y'] = diabetes.target

    wine = load_wine()
    dataset['table']['multiclass']['X'] = wine.data
    dataset['table']['multiclass']['y'] = wine.target

    return dataset


@pytest.fixture(
    params=[
        ('table', 'binary', 'lgb', 'auc', 5, True),
        ('table', 'binary', 'lgb', 'auc', 5, False)
    ],
    autouse=True
)
def tp(request, toy_dataset):
    print(toy_dataset[request.param[0]][request.param[1]]['y'])
    return {
        'X': toy_dataset[request.param[0]][request.param[1]]['X'],
        'y': toy_dataset[request.param[0]][request.param[1]]['y'],
        'input': request.param[0],
        'output': request.param[1],
        'algorithm': request.param[2],
        'metric': request.param[3],
        'cv': request.param[4],
        'tune_hp': request.param[5],
    }


def test_main(tp):
    model = main.Main(input=tp['input'], output=tp['output'], algorithm=tp['algorithm'], metric=tp['metric'])
    model.fit(tp['X'], tp['y'], cv=tp['cv'], tune_hp=tp['tune_hp'], n_trials=1)
    assert (model.X == tp['X']).all()
    assert (model.y == tp['y']).all()
    assert len(model.X_train_list) == tp['cv']
    assert len(model.y_train_list) == tp['cv']
    assert len(model.X_val_list) == tp['cv']
    assert len(model.y_val_list) == tp['cv']
    assert len(model._estimator_list) == tp['cv']
    assert model.oof_pred.shape == tp['y'].shape
    assert set(model.score.keys()) == set(['train', 'val'])
    assert len(model.score['train']) == 2
    assert len(model.score['val']) == 2
    assert model.predict(tp['X']).shape == tp['y'].shape
    if tp['output'] in ['binary', 'multiclass']:
        assert model.predict_proba(tp['X']).shape == (len(tp['y']), len(np.unique(tp['y'])))
    elif tp['output'] in ['multilabel']:
        assert model.predict_proba(tp['X']).shape == tp['y'].shape
