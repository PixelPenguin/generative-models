# Useful algorithm libraries implemented in Python.

## EasyML

### Quick Start

```python
import numpy as np
from penguin-libraries import EasyML
from sklearn.datasets import load_iris

# binary classification.
iris = load_iris()
X = iris.data
y = np.where(iris.target > 0, 1, 0)

# default argments are optimized for binary classification.
model = EasyML(input='table', output='binary', algorithm='lgb', metric='auc')
# execute cross validation and hyperparameter tuning automatically.
model.fit(X, y)

# check score (mean ± std)
print(model.score)
>>> {
    'train': (0.99821875, 0.0023251344047172736),
    'val': (0.998, 0.0040000000000000036)
}
```


## EfficientGAN
