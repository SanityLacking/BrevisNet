
from tensorflow._api.v2 import data
import branchingdnn
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
import itertools
import time
import json

#local imports
from branchingdnn.utils import *
from branchingdnn.dataset import prepare
from branchingdnn.branches import branches
from branchingdnn.eval import branchy_eval as eval
from branchingdnn.initNeptune import Neptune

def trainModel( model, dataset, epocs = 2,save = False):
    """ Train the model that is passed through. This function works for both single and multiple branch models.
    """
    logs = []
    train_ds, test_ds, validation_ds = prepare.prepareAlexNetDataset(dataset, batch_size=32)
    num_outputs = len(model.outputs) # the number of output layers for the purpose of providing labels

    model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(lr=0.001, momentum=0.9), metrics=['accuracy'])
    # model.compile(loss="sparse_categorical_crossentropy", optimizer=keras.optimizers.Adam(),metrics=["accuracy"])

    run_logdir = get_run_logdir(model.name)
    tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
    print("after reset:")
    test_scores = model.evaluate(test_ds, verbose=2)
    print("finish eval")
    printTestScores(test_scores,num_outputs)
    checkpoint = keras.callbacks.ModelCheckpoint("models/{}_new.hdf5".format(model.name), monitor='val_loss', verbose=1, mode='max')
    neptune_cbk = Neptune.getcallback()
    for j in range(epocs):
        print("epoc: {}".format(j))
        results = [j]           
        history = model.fit(train_ds, epochs=epocs, validation_data=validation_ds, callbacks=[tensorboard_cb,checkpoint,neptune_cbk])
        print(history)
        test_scores = model.evaluate(test_ds, verbose=2)
        print("overall loss: {}".format(test_scores[0]))
        if num_outputs > 1:
            for i in range(num_outputs):
                print("Output {}: Test loss: {}, Test accuracy {}".format(i, test_scores[i+1], test_scores[i+1+num_outputs]))
                results.append("Output {}: Test loss: {}, Test accuracy {}".format(i, test_scores[i+1], test_scores[i+1+num_outputs]))
        else:
            print("Test loss:", test_scores[0])
            print("Test accuracy:", test_scores[1])
        logs.append(results)
    if save:
        saveModel(model,"model_transfer_trained")

    return model


def trainModelTransfer(model, dataset, resetBranches = False, epocs = 2,save = False,transfer = True, saveName ="",customOptions="",tags = []):
    """Train the model that is passed using transfer learning. This function expects a model with trained main branches and untrained (or randomized) side branches.
    """
    logs = []
    num_outputs = len(model.outputs) # the number of output layers for the purpose of providing labels
    train_ds, test_ds, validation_ds = dataset
    # train_ds, test_ds, validation_ds = prepare.prepareMnistDataset(dataset, batch_size=32)

    #Freeze main branch layers
    #how to iterate through layers and find main branch ones?
    #simple fix for now: all branch nodes get branch in name.
    if transfer: 
        for i in range(len(model.layers)):
            print(model.layers[i].name)
            if "branch" in model.layers[i].name:
                print("setting branch layer training to true")
                model.layers[i].trainable = True
            else: 
                print("setting main layer training to false")
                model.layers[i].trainable = False               
    else:
        for i in range(len(model.layers)):
            print(model.layers[i].name)
            model.layers[i].trainable = True
            print("setting layer training to True")

    # model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=keras.optimizers.Adam(),metrics=["accuracy"])
    print(customOptions)
    if customOptions == "customLoss": 
        print("customOption: customLoss")
        model.compile(loss="categorical_crossentropy" , optimizer=tf.optimizers.SGD(lr=0.001), metrics=['accuracy'],run_eagerly=True)
    elif customOptions == "customLoss_onehot": 
        print("customOption: CrossE")
        model.compile( loss={"dense_2":keras.losses.CategoricalCrossentropy(from_logits=True)}, optimizer=tf.optimizers.SGD(lr=0.01,momentum=0.9), metrics=['accuracy'],run_eagerly=True)
    elif customOptions == "CrossE": 
        print("customOption: CrossE")
        model.compile( optimizer=tf.optimizers.SGD(lr=0.01,momentum=0.9), metrics=['accuracy'],run_eagerly=True)
    elif customOptions == "CrossE_Eadd":
        print("customOption: CrossE_Eadd")
        entropyAdd = entropyAddition_loss()
        model.compile( optimizer=tf.optimizers.SGD(lr=0.01,momentum=0.9,clipvalue=0.5), loss=[keras.losses.SparseCategoricalCrossentropy(),entropyAdd,entropyAdd,entropyAdd], metrics=['accuracy',confidenceScore, unconfidence],run_eagerly=True)
        # model.compile(optimizer=tf.optimizers.SGD(lr=0.001), loss=[crossE_test, entropyAdd, entropyAdd, entropyAdd], metrics=['accuracy',confidenceScore, unconfidence],run_eagerly=True)
    else:
        print("customOption: Other")
    # model.compile(loss=entropyAddition, optimizer=tf.optimizers.SGD(lr=0.001), metrics=['accuracy'],run_eagerly=True)
        model.compile(loss={"dense_2":keras.losses.SparseCategoricalCrossentropy(from_logits=True)} , optimizer=tf.optimizers.SGD(lr=0.001), metrics=['accuracy'],run_eagerly=True)

    run_logdir = get_run_logdir(model.name)
    tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
    # print("after reset:")
    # test_scores = model.evaluate(test_ds, verbose=2)
    # print("finish eval")
    # printTestScores(test_scores,num_outputs)

    if saveName =="":
        newModelName = "{}_branched.hdf5".format(model.name )
    else:
        newModelName = saveName
    checkpoint = keras.callbacks.ModelCheckpoint("models/{}.hdf5".format(newModelName), monitor='val_acc', verbose=1, mode='max')

    neptune_cbk = Neptune.getcallback(name = newModelName, tags =tags)
    # print("epoc: {}".format(j))
    # results = [j]           
    history =model.fit(train_ds,
            epochs=epocs,
            validation_data=validation_ds,
            validation_freq=1,
            # batch_size=1,
            callbacks=[tensorboard_cb,checkpoint,neptune_cbk])
                        # callbacks=[tensorboard_cb,checkpoint])
    print(history)
    test_scores = model.evaluate(test_ds, verbose=2)
    print("overall loss: {}".format(test_scores[0]))
    # if num_outputs > 1:
    #     for i in range(num_outputs):
    #         print("Output {}: Test loss: {}, Test accuracy {}".format(i, test_scores[i+1], test_scores[i+1+num_outputs]))
    #         results.append("Output {}: Test loss: {}, Test accuracy {}".format(i, test_scores[i+1], test_scores[i+1+num_outputs]))
    # else:
    #     print("Test loss:", test_scores[0])
    #     print("Test accuracy:", test_scores[1])
    # logs.append(results)
    return model
