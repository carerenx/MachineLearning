from typing import Any

from sklearn.model_selection import train_test_split
import pandas as pd

class Dataprocess:

    train = pd.read_csv('res/train.csv')
    test = pd.read_csv('res/test.csv')
    X_train_c = train.drop(['ID', 'CLASS'], axis=1).values
    y_train_c = train['CLASS'].values
    X_test_c = test.drop(['ID'], axis=1).values
    X_train_d=train.drop(['ID', 'CLASS','T59','T109','T175','T208'], axis=1).values
    X_test_d = train.drop(['ID', 'CLASS', 'T59', 'T109', 'T175', 'T208'], axis=1).values
    x_train, x_test, y_train, y_test = train_test_split(X_train_c, y_train_c, test_size=0.3, random_state=420,stratify=y_train_c)


    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)

    def __getattribute__(self, name: str) -> Any:
        return super().__getattribute__(name)


