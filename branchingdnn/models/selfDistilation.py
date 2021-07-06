

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

    def printStuff():
        print("helloworld")
        return 

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
    
