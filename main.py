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
# from utils import *

# from Alexnet_kaggle_v2 import * 
import branchingdnn as branching


# ALEXNET = False
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)
root_logdir = os.path.join(os.curdir, "logs\\fit\\")


# tf.debugging.experimental.enable_dump_debug_info("logs/", tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1)






if __name__ == "__main__":
    # print(branching.utils.calcEntropy([0,.3,.3]))
    # print(branching.core.printStuff())
    
    #### build alexnet model
    # x = branching.Run_alexNet( 5, modelName="alexNetv4_new.hdf5", saveName = "alexNetv4_branched_new")

    # x = branching.Run_alexNet( 10, modelName="alexNetv5.hdf5", saveName = "alexNetv5_crossE",transfer = False,customOptions="CrossE")
    

    # x = branching.core.Run_alexNet( 20, modelName="alexNetv6.hdf5", saveName = "alexNetv6_vanilla_T",transfer = True)
    x = branching.models.SelfDistilation.alexnet( 10, modelName="alexNetv6.hdf5", saveName = "alexNetv6_feat_distill",transfer = True,customOptions="CrossE")
    
    # x = tf.keras.models.load_model("models/alexNetv6.hdf5")
    # y = branching.core.evalModel(x, tf.keras.datasets.cifar10.load_data(),"natural")
  
    # x = tf.keras.models.load_model("models/alexNetv5_crossE.hdf5")
    # y = branching.GetResultsCSV(x, tf.keras.datasets.cifar10.load_data(),"_crossE")
  
    # x = tf.keras.models.load_model("models/alexNetv5_crossE_Eadd.hdf5")
    # y = branching.GetResultsCSV(x, tf.keras.datasets.cifar10.load_data(),"_crossE_Eadd")
    


    # x = branching.Run_alexNet( 10, modelName="alexNetv5.hdf5", saveName = "alexNetv5_customLoss_3",transfer = False, custom=True)

    


    # x = tf.keras.models.load_model("models/alexNetv5_customLoss_2.hdf5")
    # y = branching.GetResultsCSV(x, tf.keras.datasets.cifar10.load_data(),"custloss_2")

    # x = branching.Run_inceptionv3( 3, modelName="inception_finetuned.hdf5", saveName = "inception_branched",transfer = False)
    # x = branching.Run_resnet50v2( 3, modelName="resnet50_finetuned.hdf5", saveName = "resnet50_branched",transfer = False)

    # x = branching.Run_mnistNet( 5, modelName="mnistNormal.hdf5", saveName = "mnistNormal_branched",transfer = True)
    
    """
    Various model versions:
        alexNetv5 : up to date version of testing, trained using the augmented, not self standardized images. base for most other versions that I tried out
        
        models with alt in the name are models I made trying to track down what is going on with the missing 0 class from branches
        alexNetv5_alt6: model with branches on dense layers, this model actually lost a second class completely as well, class 1. 
    """

    # x = tf.keras.models.load_model("models/mnist_transfer_trained_21-01-04_125846.hdf5")
    # x.summary()
    # branching.eval_branches(x,branching.loadTrainingData(),1,"accuracy")
    # branching.eval_branches(x,branching.loadTrainingData(),1,"entropy")
    # branching.find_mistakes(x,branching.loadTrainingData(),1)
    
    # x = tf.keras.models.load_model("models/alexnet_branched_new_trained.hdf5")
    # x.summary()
    # branching.entropyMatrix(x,tf.keras.datasets.cifar10.load_data())


    # branching.eval_branches(x,tf.keras.datasets.cifar10.load_data(),1,"entropy")
    # branching.find_mistakes(x,tf.keras.datasets.cifar10.load_data(),1)



    # x.summary()
    # branching.eval_branches(x,tf.keras.datasets.cifar10.load_data(),1,)



    ####Make a new model
    # x = branching.Run_alexNet(50, saveName = "alexnext_branched_fullModel_trained",transfer = False)
    # x.summary()

    # branching.datasetStats(tf.keras.datasets.cifar10.load_data())

    # x = tf.keras.models.load_model("models/alexnext_branched_fullModel_trained_branched_branched.hdf5")
    # x.summary()
    # branching.eval_branches(x,tf.keras.datasets.cifar10.load_data())
    # """ 
    # x = tf.keras.models.load_model("models/alexnet_branch_pooling.hdf5")
    # x.summary()
    # branching.eval_branches(x,tf.keras.datasets.cifar10.load_data())
    # x = tf.keras.models.load_model("models/alexnet_branched_new_trained.hdf5")
    # x.summary()
    # branching.eval_branches(x,tf.keras.datasets.cifar10.load_data())
    # """
    # x = branching.Run_train_model(x,tf.keras.datasets.cifar10.load_data(),10)
    # x.save("models/alexnet_branched_new_trained.hdf5")

    # x = branching.Run_alexNet(1)

    # x = branching.mnistbranching()
    

    # x = branching.loadModel("models/mnist_trained_20-12-15_112434.hdf5")
    # x = tf.keras.models.load_model("models/mnist2_transfer_trained_.tf")

    # x.save("models/mnistNormal2_trained.hdf5")
    # saveModel(x,"mnist2_transfer_trained_final",includeDate=False)
    pass

