import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data_dict = pickle.load(open('./data.pickle', 'rb'))
# len(data_dict) ====> 2  (data/labels)


data = []
labels = []

for d, label in zip(data_dict['data'], data_dict['labels']):
    if len(d) != len(data_dict['data'][0]):
        if len(d) == 84:
            d1, d2 = d[:42], d[42:]
            data.append([float(x) for x in d1])
            data.append([float(x) for x in d2])
            labels.append(label)
            labels.append(label)
        else:
            raise ValueError('Toutes les listes doivent avoir la même taille.')
    else:
        data.append([float(x) for x in d])
        labels.append(label)


# print(len(data), len(labels))
# print(labels)


x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% des échantillons ont été classés correctement !'.format(score * 100))

f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()

