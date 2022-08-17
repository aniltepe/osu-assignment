from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
df = pd.read_csv('train.csv')
data_raw = df.to_numpy()
gender_raw = data_raw[:,4]
embark_raw = data_raw[:,11]
output = data_raw[:,1]
gender = np.empty(np.size(gender_raw, 0))
for i in range(len(gender_raw)): gender[i] = 1 if gender_raw[i] == 'male' else 0 if gender_raw[i] == 'female' else -1
embark = np.empty(np.size(embark_raw, 0))
for i in range(len(embark_raw)): embark[i] = 1 if embark_raw[i] == 'C' else 0.5 if embark_raw[i] == 'Q' else 0 if embark_raw[i] == 'S' else -1
concatenated = np.vstack((gender, embark, output))
for i in range(np.size(concatenated, 1) - 1, -1, -1):
    if concatenated[0, i] == -1.0 or concatenated[1, i] == -1.0:
        concatenated = np.delete(concatenated, i, 1)
X = concatenated[0:2, :].T
y = concatenated[2, :]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(X_train.tolist(), y_train.tolist())
print(neigh.score(X_test.tolist(), y_test.tolist()))
# print(neigh.kneighbors([[1.0, 1.0]]))


# print(neigh.predict_proba([[1.0, 1.0]]))


# X = concatenated.tolist()
# y = output.tolist()