###evaluate the completed models. ####
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
import itertools
import sys

# from keras.models import load_model
# from keras.utils import CustomObjectScope
# from keras.initializers import glorot_uniform

import math
import pydot
import os
#os.environ["PATH"] += os.pathsep + "C:\Program Files\Graphviz\bin"
#from tensorflow.keras.utils import plot_model
from utils import *

from Alexnet_kaggle_v2 import * 
from branchyNet import BranchyNet


<<<<<<< HEAD
=======

>>>>>>> dc183174b963244a9bddc35ca7c3e87bd4d6ef3f

if __name__ == "__main__":
    opts = [opt for opt in sys.argv[1:] if opt.startswith("-")]
    args = [arg for arg in sys.argv[1:] if not arg.startswith("-")]
    
    # print("model to eval: {}".format(args[0]))

    #which model to eval
    #either list a valid model name, aka mnist/alexnet, etc
    #or specify a model filepath

    #which dataset to eval on?
    #check the model name for one of the valid model types and use the default dataset for that.
    

    #load the model
    branchy = BranchyNet()
    branchy.ALEXNET = True
    #load the dataset
    x = tf.keras.models.load_model("models/alexnet_branch_pooling.hdf5")

<<<<<<< HEAD
    y = branchy.BranchEntropyConfusionMatrix(x, tf.keras.datasets.cifar10.load_data())
=======
>>>>>>> dc183174b963244a9bddc35ca7c3e87bd4d6ef3f

    import modelProfiler
    # layerBytes = modelProfiler.getLayerBytes(x,'alexnet_branch_pooling')
    #modelProfiler.getFlopsFromArchitecture(model,'alexnet')
    layerFlops = modelProfiler.getLayerFlops_old('models/alexnet_branch_pooling.hdf5','alexnet_branch_pooling')

    #print the model structure summary
    # x.summary()
    #eval the model
    # branchy.eval_branches(x, tf.keras.datasets.cifar10.load_data(),"accuracy")
    #print the results

    pass