from sklearn.neighbors import KNeighborsClassifier
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
for i in range(len(embark_raw) - 1, 0, -1):
    if embark[i] == -1:
        embark = np.delete(embark, i, 0)
        gender = np.delete(gender, i, 0)
        output = np.delete(output, i, 0)
concatenated = np.vstack((gender, embark)).T
X = concatenated.tolist()
y = output.tolist()
neigh = KNeighborsClassifier(n_neighbors=30)
neigh.fit(X, y)
print(neigh.predict_proba([[1.0, 1.0]]))