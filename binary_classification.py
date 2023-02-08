from sklearn import datasets
from sklearn.model_selection import train_test_split

data_br = datasets.load_breast_cancer()

X=data_br.data
Y=data_br.target

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense

X_train = X_train/255.0
X_test = X_test/255.0
model = Sequential()
model.add(Dense(30, activation = 'relu')),
model.add(Dense(40, activation = 'relu')),
model.add(Dense(70, activation = 'relu')),
model.add(Dense(100, activation = 'relu')),
model.add(Dense(2, activation = 'sigmoid'))


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10)

model.evaluate(X_test, y_test, verbose = 2)
predict = model.predict(X_test)

c = 0
a = 0
for i in range(len(y_test)):
  if np.argmax(predict[i]) == y_test[i]:
    result = 'right'
    c+=1
    a+=1
  else:
    result = 'wrong'
    a+=1
  print(f"Predicted value: {np.argmax(predict[i])}     Actual value: {y_test[i]}    {result}")
print(f"passed {c} out of {a} test cases")
