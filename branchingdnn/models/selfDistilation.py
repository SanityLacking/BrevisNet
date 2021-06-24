

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
from branchingdnn import branches
# import branchyNet
#class for building a seflDistilation branching model.

class SelfDistilation(branchingdnn.core):

    def printStuff():
        print("helloworld")
        return 

    # initialize an implementation of alexnet branching that uses the selfdistil methodology
    def Alex_SelfDistil(numEpocs = 2, modelName="", saveName ="",transfer = True,customOptions=""):
        x = tf.keras.models.load_model("models/{}".format(modelName))

        x.summary()
        if saveName =="":
            saveName = modelName
        tf.keras.utils.plot_model(x, to_file="{}.png".format(saveName), show_shapes=True, show_layer_names=True)
        # funcModel = models.Model([input_layer], [prev_layer])
        # funcModel = self.addBranches(x,["dense","conv2d","max_pooling2d","batch_normalization","dense","dropout"],newBranch)
        funcModel = branches.add(x,["max_pooling2d","max_pooling2d_1","dense"],branches.newBranch_flatten,exact=True)
        
        funcModel.summary()
        funcModel.save("models/{}".format(saveName))
        
        funcModel = branchingdnn.models.trainModelTransfer(funcModel,tf.keras.datasets.cifar10.load_data(), epocs = numEpocs, save = False, transfer = transfer, saveName = saveName,customOptions=customOptions)
        # funcModel.save("models/{}".format(saveName))
        # x = keras.Model(inputs=x.inputs, outputs=x.outputs, name="{}_normal".format(x.name))
        return x
    
