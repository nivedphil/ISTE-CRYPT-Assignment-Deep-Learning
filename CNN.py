import tensorflow as tf
from keras import datasets
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

(X_train, y_train),(X_test, y_test) = datasets.cifar10.load_data()

X_train = X_train/255.0
X_test = X_test/255.0
model = Sequential()
model.add(Conv2D(32,(3,3),activation = 'relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64,(3,3),activation = 'relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64,(3,3),activation = 'relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation = 'softmax'))

model.compile(optimizer = 'adam',loss = 'sparse_categorical_crossentropy',metrics = ['accuracy'])

model.fit(X_train,y_train,epochs = 10)