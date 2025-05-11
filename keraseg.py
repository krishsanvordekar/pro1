import numpy as np
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

train_labels= []
train_samples= []

for i in range(50):
    random_younger = randint(13,64)
    train_samples.append(random_younger)
    train_labels.append(1)
    #youngers who expireinced the side effects

    random_older= randint(65,100)
    train_samples.append(random_older)
    train_labels.append(0)
    #olders who didnt experienced the side effect

for i in range(1000):
    random_younger = randint(13,64)
    train_samples.append(random_younger)
    train_labels.append(0)

    random_older= randint(65,100)
    train_samples.append(random_older)
    train_labels.append(1)

for i in train_samples:
    print(i)

train_labels=np.array(train_labels)
train_samples=np.array(train_samples)
train_labels, train_samples= shuffle(train_labels, train_samples)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_train_samples= scaler.fit_transform(train_samples.reshape(-1,1))

for i in scaled_train_samples:
    print(i)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential  # ✅ Correct
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy

model= Sequential([
    Dense(units=16, input_shape=(1,), activation='relu'),
    Dense(units=32, activation='relu'),
    Dense(units=2, activation='softmax')
])

model.summary()

model.compile(optimizer=Adam(learning_rate=0.0001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x=scaled_train_samples, y=train_labels,validation_split=0.1, batch_size=0, epochs=30, shuffle=True, verbose=2)





