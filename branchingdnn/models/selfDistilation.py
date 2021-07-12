

# import the necessary packages
import branchingdnn

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
import itertools
import time
import json
import math
import pydot
import os

from branchingdnn.utils import *
from branchingdnn.branches import branch
from branchingdnn.dataset import prepare
# import branchyNet
#class for building a seflDistilation branching model.

class SelfDistilation(branchingdnn.core):

     # initialize an implementation of alexnet branching that uses the selfdistil methodology
    def alexnet(numEpocs = 2, modelName="", saveName ="",transfer = True,customOptions=""):
        x = tf.keras.models.load_model("models/{}".format(modelName))

        x.summary()
        if saveName =="":
            saveName = modelName
        tf.keras.utils.plot_model(x, to_file="{}.png".format(saveName), show_shapes=True, show_layer_names=True)
        # funcModel = models.Model([input_layer], [prev_layer])
        # funcModel = self.addBranches(x,["dense","conv2d","max_pooling2d","batch_normalization","dense","dropout"],newBranch)
        funcModel = branch.add_distil(x,["max_pooling2d","max_pooling2d_1","dense"],branch.newBranch_distil,exact=True)
        #so to self distil, I have to pipe the loss from the main exit back to the branches.
        funcModel.summary()
        funcModel.save("models/{}".format(saveName))
        dataset = prepare.dataset_distil(tf.keras.datasets.cifar10.load_data(),32,5000,22500,(227,227))
        funcModel = branchingdnn.models.trainModelTransfer(funcModel, dataset, epocs = numEpocs, save = False, transfer = transfer, saveName = saveName,customOptions=customOptions)
        # funcModel.save("models/{}".format(saveName))
        # x = keras.Model(inputs=x.inputs, outputs=x.outputs, name="{}_normal".format(x.name))
        return x
    
    class BranchEndpoint(keras.layers.Layer):
        def __init__(self, name=None):
            super(BranchEndpoint, self).__init__(name=name)
            self.loss_fn = keras.losses.SparseCategoricalCrossentropy()
            self.loss_coefficient = 1
            self.feature_loss_coefficient = 1
    #         self.loss_fn = keras.losses.sparse_categorical_crossentropy()

        def call(self, prediction, targets, additional_loss=None, student_features=None, teaching_features=None, sample_weights=None):
            # Compute the training-time loss value and add it
            # to the layer using `self.add_loss()`.
            print(prediction, targets, additional_loss)
            #loss functions are (True, Prediction)
            loss = self.loss_fn(targets, prediction, sample_weights)
            print(loss)
            #if loss is a list of additional loss objects
            if isinstance(additional_loss,list):
                for i in range(len(additional_loss)):
                    loss += self.loss_fn(targets, additional_loss[i], sample_weights) * self.loss_coefficient
            elif additional_loss is not None:
                loss += self.loss_fn(targets, additional_loss, sample_weights) * self.loss_coefficient
                
            #feature distillation
            if teaching_features is not None and student_features is not None:
                diff = tf.norm(tf.math.abs(student_features - teaching_features)) * self.feature_loss_coefficient
                loss += self.loss_fn(targets, additional_loss, sample_weights)
            #TODO might be faster to concatenate all elements together and then perform the loss once on all the elements.
            
            self.add_loss(loss)

            return tf.nn.softmax(prediction)
