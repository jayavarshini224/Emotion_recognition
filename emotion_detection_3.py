import sys
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import keras
from keras.layers import Conv2D, MaxPool2D, AveragePooling2D, Input, BatchNormalization, MaxPooling2D, Activation, Flatten, Dense, Dropout,GlobalAveragePooling2D,LeakyReLU,Lambda,concatenate,add
from keras.models import Model
from keras.utils import to_categorical
from keras.preprocessing import image


def preprocess_pixels(pixel_data):
  images = []
  for i in range(len(pixel_data)):
    img = np.fromstring(pixel_data[i], dtype='int', sep=' ')
    img = img.reshape(48,48,1)
    images.append(img)
  X = np.array(images)
  return X


def residual_nw(x):
  
    cardinality=32
  
    
    def add_common_layers(y):
        y=BatchNormalization()(y)
        y=LeakyReLU()(y)

        return y


    
    def grouped_convolution(y,nb_channels,_strides):
        
        if cardinality == 1:
            return Conv2D(nb_channels,kernel_size=(3,3),strides=_strides,padding='same')(y)
        
        assert not nb_channels % cardinality
        _d = nb_channels // cardinality

        groups = []
        for j in range(cardinality):
            group = Lambda(lambda z: z[:, :, :, j * _d:j * _d + _d])(y)
            groups.append(Conv2D(_d, kernel_size=(3, 3), strides=_strides, padding='same')(group))

        y = concatenate(groups)

        return y

    def residual_block(y, nb_channels_in, nb_channels_out, _strides=(1, 1), _project_shortcut=False):
        shortcut = y

        y = Conv2D(nb_channels_in, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
        y = add_common_layers(y)
        y = grouped_convolution(y, nb_channels_in, _strides=_strides)
        y = add_common_layers(y)


        y = Conv2D(nb_channels_out, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
        y = BatchNormalization()(y)

        if _project_shortcut or _strides != (1, 1):
            shortcut = Conv2D(nb_channels_out, kernel_size=(1, 1), strides=_strides, padding='same')(shortcut)
            shortcut = BatchNormalization()(shortcut)

        y = add([shortcut, y])
        y = LeakyReLU()(y)

        return y

    # conv1
    x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = add_common_layers(x)

    # conv2
    x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    for i in range(3):
        project_shortcut = True if i == 0 else False
        x = residual_block(x, 32, 64, _project_shortcut=project_shortcut)


    # conv3
    for i in range(4):
        # down-sampling is performed by conv3_1, conv4_1, and conv5_1 with a stride of 2
        strides = (2, 2) if i == 0 else (1, 1)
        x = residual_block(x,64, 128, _strides=strides)

    # conv4
    for i in range(6):
        strides = (2, 2) if i == 0 else (1, 1)
        x = residual_block(x,128,256, _strides=strides)

    # conv5
    for i in range(3):
        strides = (2, 2) if i == 0 else (1, 1)
        x = residual_block(x,256, 512, _strides=strides)

    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    x = Dense(200)(x)
    x = Dropout(0.6)(x)
    x = Dense(7,activation='softmax')(x)

    return x





#Main fn types
#label_dict = {0 : 'Angry', 1 : 'Disgust', 2 : 'Fear', 3 : 'Happiness', 4 : 'Sad', 5 : 'Surprise', 6 : 'Neutral'}

#read dataset
data=pd.read_csv('fer2013.csv')

#separate pixels and emotions
pixel_data = data['pixels']
label_data = data['emotion']

#randomly duplicates minority classes reducing the imbalance in the dataset
oversampler = RandomOverSampler(sampling_strategy='auto')
X_over, Y_over = oversampler.fit_resample(pixel_data.values.reshape(-1,1), label_data.values)
X_over_series = pd.Series(X_over.flatten())

#pixel to image form
X = preprocess_pixels(X_over_series)
#reshape emotion
Y = Y_over
Y = Y.reshape(Y.shape[0],1)


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, random_state = 45)

y_train = to_categorical(Y_train, num_classes=7)
y_test = to_categorical(Y_test, num_classes=7)


img_tensor=Input(shape=(48,48,1))
nw_op=residual_nw(img_tensor)
model=Model(inputs=[img_tensor],outputs=[nw_op])



adam = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

#Train model
model.fit(X_train, y_train,epochs=35,validation_data=(X_test, y_test))

fer_json = model.to_json()
with open("model2.json", "w") as json_file:
    json_file.write(fer_json)
model.save_weights("model2.h5")


