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

import xgboost as xgb

dtrain = xgb.DMatrix(tr_x, label=tr_y)
dvalid = xgb.DMatrix(va_x, label=va_y)
dtest= xgb.DMatrix(test_x)

params = {'objective': 'binary:logistic', 'silent': 1, 'random_state': 71}
num_round = 50

watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
model = xgb.train(params, dtrain, num_round, evals=watchlist)

va_pred = model.predict(dvalid)

from sklearn.metrics import log_loss

score = log_loss(va_y, va_pred)
print(f'logloss: {score:.4f}')

prod = model.predict(dtest)
print(prod)
