# import the necessary packages
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
import itertools

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
# ALEXNET = False
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)
root_logdir = os.path.join(os.curdir, "logs\\fit\\")


# tf.debugging.experimental.enable_dump_debug_info("logs/", tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1)

def newBranchCustom(prevLayer, outputs=[]):
    """ example of a custom branching layer, used as a drop in replacement of "newBranch"
    """                 
    branchLayer = layers.Dense(10, name=tf.compat.v1.get_default_graph().unique_name("branch_output"))(prevLayer)
    outputs.append(layers.Softmax(name=tf.compat.v1.get_default_graph().unique_name("branch_softmax"))(branchLayer))

    return outputs




if __name__ == "__main__":
    branchy = BranchyNet()
    branchy.ALEXNET = True
    # branchy.ALEXNET = False
    # x = branchy.Run_mnistNormal(1)
    # x = branchy.Run_mnistTransfer(1)

    #### build alexnet model
    # x = branchy.Run_alexNet( 5, modelName="alexNetv4_new.hdf5", saveName = "alexNetv4_branched_new")

    x = branchy.Run_alexNet( 10, modelName="alexNetv5.hdf5", saveName = "alexNetv5_customLoss_3",transfer = False)
 


    # x = branchy.Run_alexNet( 10, modelName="alexNetv5.hdf5", saveName = "alexNetv5_customLoss_3",transfer = False, custom=True)

    x = tf.keras.models.load_model("models/alexNetv5_customLoss_3.hdf5")
    y = branchy.GetResultsCSV(x, tf.keras.datasets.cifar10.load_data(),"custloss_3")


    # x = tf.keras.models.load_model("models/alexNetv5_customLoss_2.hdf5")
    # y = branchy.GetResultsCSV(x, tf.keras.datasets.cifar10.load_data(),"custloss_2")

    # x = branchy.Run_inceptionv3( 3, modelName="inception_finetuned.hdf5", saveName = "inception_branched",transfer = False)
    # x = branchy.Run_resnet50v2( 3, modelName="resnet50_finetuned.hdf5", saveName = "resnet50_branched",transfer = False)

    # x = branchy.Run_mnistNet( 5, modelName="mnistNormal.hdf5", saveName = "mnistNormal_branched",transfer = True)
    
    """
    Various model versions:
        alexNetv5 : up to date version of testing, trained using the augmented, not self standardized images. base for most other versions that I tried out
        
        models with alt in the name are models I made trying to track down what is going on with the missing 0 class from branches
        alexNetv5_alt6: model with branches on dense layers, this model actually lost a second class completely as well, class 1. 
    """

    # x = tf.keras.models.load_model("models/mnist_transfer_trained_21-01-04_125846.hdf5")
    # x.summary()
    # branchy.eval_branches(x,branchy.loadTrainingData(),1,"accuracy")
    # branchy.eval_branches(x,branchy.loadTrainingData(),1,"entropy")
    # branchy.find_mistakes(x,branchy.loadTrainingData(),1)
    
    # x = tf.keras.models.load_model("models/alexnet_branched_new_trained.hdf5")
    # x.summary()
    # branchy.entropyMatrix(x,tf.keras.datasets.cifar10.load_data())


    # branchy.eval_branches(x,tf.keras.datasets.cifar10.load_data(),1,"entropy")
    # branchy.find_mistakes(x,tf.keras.datasets.cifar10.load_data(),1)



    # x.summary()
    # branchy.eval_branches(x,tf.keras.datasets.cifar10.load_data(),1,)



    ####Make a new model
    # x = branchy.Run_alexNet(50, saveName = "alexnext_branched_fullModel_trained",transfer = False)
    # x.summary()

    # branchy.datasetStats(tf.keras.datasets.cifar10.load_data())

    # x = tf.keras.models.load_model("models/alexnext_branched_fullModel_trained_branched_branched.hdf5")
    # x.summary()
    # branchy.eval_branches(x,tf.keras.datasets.cifar10.load_data())
    # """ 
    # x = tf.keras.models.load_model("models/alexnet_branch_pooling.hdf5")
    # x.summary()
    # branchy.eval_branches(x,tf.keras.datasets.cifar10.load_data())
    # x = tf.keras.models.load_model("models/alexnet_branched_new_trained.hdf5")
    # x.summary()
    # branchy.eval_branches(x,tf.keras.datasets.cifar10.load_data())
    # """
    # x = branchy.Run_train_model(x,tf.keras.datasets.cifar10.load_data(),10)
    # x.save("models/alexnet_branched_new_trained.hdf5")

    # x = branchy.Run_alexNet(1)

    # x = branchy.mnistBranchy()
    

    # x = branchy.loadModel("models/mnist_trained_20-12-15_112434.hdf5")
    # x = tf.keras.models.load_model("models/mnist2_transfer_trained_.tf")

    # x.save("models/mnistNormal2_trained.hdf5")
    # saveModel(x,"mnist2_transfer_trained_final",includeDate=False)
    pass

