from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense

iris = datasets.load_iris()

X = iris.data
Y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 42)

X_train = X_train/255.0
X_test = X_test/255.0
model = Sequential()
model.add(Dense(30, activation = 'relu')),
model.add(Dense(40, activation = 'relu')),
model.add(Dense(60, activation = 'relu')),
model.add(Dense(80, activation = 'relu')),
model.add(Dense(100, activation = 'relu')),
model.add(Dense(3, activation = 'softmax'))

model.compile(
    optimizer = 'adam', loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)

model.fit(X_train, y_train, epochs = 30)