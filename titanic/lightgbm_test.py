import numpy as np
import pandas as pd

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train_x = train.drop(['Survived'], axis=1)
train_y = train['Survived']

test_x = test.copy()

train_x = train_x.drop(['PassengerId'], axis = 1)
test_x = test_x.drop(['PassengerId'], axis = 1)

train_x = train_x.drop(['Name', 'Ticket', 'Cabin'], axis = 1)
test_x = test_x.drop(['Name', 'Ticket', 'Cabin'], axis = 1)

from sklearn.preprocessing import LabelEncoder

for c in ['Sex', 'Embarked']:
    le = LabelEncoder()
    le.fit(train_x[c].fillna('NA'))

    train_x[c] = le.transform(train_x[c].fillna('NA'))
    test_x[c] = le.transform(test_x[c].fillna('NA'))

from sklearn.model_selection import KFold

kf = KFold(n_splits=4, shuffle=True, random_state=71)
tr_idx, va_idx = list(kf.split(train_x))[0]
tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

import lightgbm as lgb

dtrain = lgb.Dataset(tr_x, tr_y)
dvalid = lgb.Dataset(va_x, va_y)

params = {
    'objective': 'binary', 'seed': 71, 'verbose': 0,
     'metrics': 'binary_logloss'}
num_round = 100

categorical_features = [    'Sex', 'Embarked']
model = lgb.train(
    params, dtrain, num_boost_round=num_round,
    categorical_feature=categorical_features,
    valid_names=['train', 'valid'], valid_sets=[dtrain, dvalid])

va_pred = model.predict(va_x)

from sklearn.metrics import log_loss

score = log_loss(va_y, va_pred)
print(f'logloss: {score:.4f}')

prod = model.predict(test_x)
print(prod)
