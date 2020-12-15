import os, shutil, random, glob
# import cv2
import numpy as np
import pandas as pd

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# CUDA_VISIBLE_DEVICES = 2
import tensorflow as tf
import keras
from keras.models import load_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import cifar10
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, BatchNormalization
import matplotlib.pyplot as plt

def AlexLoadModel():
#load Model
    with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
        new_model = load_model('models/saved-model-alexnet-03-0.80.hdf5')
        new_model.summary()
    return new_model

def loadData():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        # width_shift_range=[-200,200],
        rotation_range=90,
        brightness_range=[0.2,1.0],
        # height_shift_range=0.5,
        shear_range=0.2,
        zoom_range=0.2,
        # zca_whitening=True,
        horizontal_flip=True,
        vertical_flip=True,
        
        )

    test_datagen = ImageDataGenerator(rescale=1./255, brightness_range=[0.2,1.0],horizontal_flip=True,
        vertical_flip=True)

    train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(224, 224),
        batch_size=32,
        shuffle=True,
        seed = 42,
        class_mode='categorical')
    validation_generator = test_datagen.flow_from_directory(
        'data/validation',
        target_size=(224, 224),
        batch_size=32,
        shuffle=False,
        seed = 42,
        class_mode='categorical')

    
    return train_generator, validation_generator

from tensorflow.contrib import slim

def runAndTrainModel():
    resize = 224

    #define the model
    train_generator, validation_generator = loadData()
    print(train_generator.class_indices)
    print(validation_generator.class_indices)
    # AlexNet
    model = Sequential()
    #第一段
    model.add(Conv2D(filters=96, kernel_size=(11,11),
                    strides=(4,4), padding='valid',
                    input_shape=(resize,resize,3),
                    activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3,3),
                        strides=(2,2),
                        padding='valid'))
    #第二段
    model.add(Conv2D(filters=256, kernel_size=(5,5),
                    strides=(1,1), padding='same',
                    activation='relu'))
    
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3,3),
                        strides=(2,2),
                        padding='valid'))
    #第三段
    model.add(Conv2D(filters=384, kernel_size=(3,3),
                    strides=(1,1), padding='same',
                    activation='relu'))
  
    model.add(Conv2D(filters=384, kernel_size=(3,3),
                    strides=(1,1), padding='same',
    activation='relu'))
    model.add(Conv2D(filters=256, kernel_size=(3,3),
                    strides=(1,1), padding='same',
                    activation='relu'))
    model.add(MaxPooling2D(pool_size=(3,3),
                        strides=(2,2), padding='valid'))
    #第四段
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1000, activation='relu'))
    model.add(Dropout(0.5))

    # Output Layer
    model.add(Dense(2, name='output'))
    model.add(Activation('softmax'))
    #set the data for the model to fit on

    model.input

    # for layer in model.layers:
    #     slim.model_analyzer.analyze_vars([layer.output ], print_info=True)


    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata= tf.RunMetadata()

    #train the model, save the model every 10 epochs.
    model.compile(loss='categorical_crossentropy',
            optimizer='sgd',
            metrics=['accuracy']
            )
    # model.summary()
    # filepath = "models/saved-model-alexnet-{epoch:02d}-{val_acc:.2f}.hdf5"
    # checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    # callbacks_list = [checkpoint]
    # model.fit(train_data, train_label,
    #     batch_size = 64,
    #     epochs = 50,
    #     #   validation_split = 0.2,
    #     shuffle = True)

    model.fit_generator(
        train_generator,
        steps_per_epoch=5000,
        epochs=10,
        validation_data=validation_generator,
        validation_steps=1000,
        #callbacks=callbacks_list,
        shuffle=True,
        max_queue_size = 50,
        # use_multiprocessing = True,
        workers = 6
        )
    model.save('models/alexNet113.hdf5')
    return 0 
    
def loadAndEvalModel():
    model = AlexLoadModel()
    train_generator, validation_generator = loadData()
    print(train_generator.class_indices)
    print(validation_generator.class_indices)

    # import modelProfiler
    # layerBytes = modelProfiler.getLayerBytes(model,'alexnet')
    #modelProfiler.getFlopsFromArchitecture(model,'alexnet')
    # layerFlops = modelProfiler.getLayerFlops('models/saved-model-alexnet-03-0.80.hdf5','alexnet')


    predict = model.evaluate_generator(validation_generator,steps = 32)
    print(predict)
    # from sklearn.utils.extmath import softmax
    # results = softmax(predict)
    # index_max = np.argmax(results)
    return 0


if __name__ == '__main__':
    runAndTrainModel()
    #loadAndEvalModel()
    print("Task Complete")

    