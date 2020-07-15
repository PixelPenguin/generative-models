import pytorch_lightning as pl

from . import dataset
from .models.model import Model as Model


class Main:

    def __init__(self, hparams):
        """
        ネットワークや訓練情報を記述する。詳細はmodel.pyもしくはmodels/へ
        trainerとの関係
        """
        self.model = Model(hparams)
        self.dataloader = dataset.dataloader()

    def fit(self, X=None, **kwargs):
        """
        Args:
            X (DataLoader): default dataset and dataloader is defined in dataset.py
        """
        self.trainer = pl.Trainer(gpus=1)
        if X is None:
            return self.trainer.fit(self.model, self.dataloader['train'])
        else:
            return self.trainer.fit(self.model, X)

    def predict(self, X=None):
        if X is None:
            return self.trainer.test(self.dataloader['test'])
        else:
            return self.trainer.test(self.model, X)