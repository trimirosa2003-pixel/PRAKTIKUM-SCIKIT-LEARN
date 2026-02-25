import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load data
data_dict = pickle.load(open(
    r'C:\Users\user\Documents\Rosa\python\buku\pelatihan\data.pickle',
    'rb'
))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Split data
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.1, shuffle=True, stratify=labels
)

# Train model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Test model
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly!'.format(score * 100))

# Save model
f = open(
    r'C:\Users\user\Documents\Rosa\python\buku\pelatihan\model.pickle',
    'wb'
)
pickle.dump({'model': model}, f)
f.close()