import pandas as pd

df = pd.read_csv('C:/Users/jenom/.vscode/.venv/poker-hand-training.csv')
X = df.iloc[:, :6]
y = df.iloc[:, -1]

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=500)

model.fit(X,y)

import pickle
filename = 'poker-mode13.sav'
pickle.dump(model, open(filename, 'wb'))

X = df.iloc[:, :8]
y = df.iloc[:, -1]

model = LogisticRegression(max_iter = 500)
model.fit(X,y)

filename = 'poker-mode14.sav'
pickle.dump(model, open(filename,'wb'))
