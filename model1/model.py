from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
import numpy as np
import pickle

x_train = pickle.load(open('x_train.pkl','rb'))
y_train = pickle.load(open('y_train.pkl','rb'))

model = Sequential()
model.add(Dense(128, input_shape=(len(x_train[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(y_train[0]), activation='softmax'))

sgd = SGD(learning_rate=0.01, decay=10-6, momentum=0.9, nesterov = True)
model.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics=['accuracy'])

model.fit(np.array(x_train), np.array(y_train), epochs=200, batch_size=5, verbose=1)

model.save('chatbot_model.h5')