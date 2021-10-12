import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# 前処理

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train_x = train.drop(['Survived'], axis=1)
train_y = train['Survived']

test_x = test.copy()

train_x = train_x.drop(['PassengerId'], axis = 1)
test_x = test_x.drop(['PassengerId'], axis = 1)

train_x = train_x.drop(['Name', 'Ticket', 'Cabin'], axis = 1)
test_x = test_x.drop(['Name', 'Ticket', 'Cabin'], axis = 1)

for c in ['Sex', 'Embarked']:
    le = LabelEncoder()
    le.fit(train_x[c].fillna('NA'))

    train_x[c] = le.transform(train_x[c].fillna('NA'))
    test_x[c] = le.transform(test_x[c].fillna('NA'))

# モデル生成と学習

from xgboost import XGBClassifier

model = XGBClassifier(n_estimators=20, random_state=71)
model.fit(train_x, train_y)

pred = model.predict_proba(test_x)[:,1]
pred_label = np.where(pred > 0.5, 1, 0)

submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': pred_label})
submission.to_csv('submission_first.csv', index=False)

# モデルの評価

from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import KFold

scores_accuracy = []
scores_logloss = []

kf = KFold(n_splits=4, shuffle=True, random_state=71)
for tr_idx, va_idx in kf.split(train_x):
    tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
    tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

    # モデル生成と学習
    model = XGBClassifier(n_estimators=20, random_state=71)
    model.fit(tr_x, tr_y)

    va_pred = model.predict_proba(va_x)[:, 1]
    logloss = log_loss(va_y, va_pred)
    accuracy = accuracy_score(va_y, va_pred > 0.5)

    scores_logloss.append(logloss)
    scores_accuracy.append(accuracy)

logloss = np.mean(scores_logloss)
accuracy = np.mean(scores_accuracy)
print(f'logloss: {logloss:.4f}, accuracy: {accuracy:.4f}')
