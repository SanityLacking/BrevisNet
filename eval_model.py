###evaluate the completed models. ####
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
import itertools
import sys
import math
import pydot
import os
#os.environ["PATH"] += os.pathsep + "C:\Program Files\Graphviz\bin"
#from tensorflow.keras.utils import plot_model
from utils import *

from Alexnet_kaggle_v2 import * 
from branchyNet import BranchyNet

def evalBranchMatrix_old(model, input, labels=""):
    num_outputs = len(model.outputs) # the number of output layers for the purpose of providing labels
    model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(lr=0.001), metrics=['accuracy'])
    print(type(input))
    if labels == "":
        if type(input)=="tensorflow.python.data.ops.dataset_ops.BatchDataset":
            print("yes")
            pass
        else: 
            print("no")
    iterator = iter(input)
    item = iterator.get_next()
    pred=[]
    labels=[]
    for i in range(100):
        pred.append(model.predict(item[0]))
        labels.append(item[1])
    
    results = throughputMatrix(pred, labels, num_outputs)
    print(results)
    print(pd.DataFrame(results).T)

    return

if __name__ == "__main__":
    opts = [opt for opt in sys.argv[1:] if opt.startswith("-")]
    args = [arg for arg in sys.argv[1:] if not arg.startswith("-")]
    
    # print("model to eval: {}".format(args[0]))

    #which model to eval
    #either list a valid model name, aka mnist/alexnet, etc
    #or specify a model filepath

    #which dataset to eval on?
    #check the model name for one of the valid model types and use the default dataset for that.
    
    print("evalModel")

    #load the model
    branchy = BranchyNet()
    branchy.ALEXNET = True
    #load the dataset
    # x = tf.keras.models.load_model("models/alexnet_branched_new_trained.hdf5")
    # normal one
    # x = tf.keras.models.load_model("models/alexnet_branch_pooling.hdf5")

    # test model
    x = tf.keras.models.load_model("models/alexNetv5_alt8_branched.hdf5")
    x.summary()
    print(x.outputs)


    y = branchy.GetResultsCSV(x, tf.keras.datasets.cifar10.load_data(),"_alt8_1")
    # y = branchy.GetResultsCSV(x,keras.datasets.mnist.load_data(), "_mnist")
    

    # import modelProfiler
    # layerBytes = modelProfiler.getLayerBytes(x,'alexnet_branch_pooling')
    #modelProfiler.getFlopsFromArchitecture(model,'alexnet')
    # layerFlops = modelProfiler.getLayerFlops_old('models/alexnet_branch_pooling.hdf5','alexnet_branch_pooling')

    # y = branchy.BranchEntropyConfusionMatrix(x, tf.keras.datasets.cifar10.load_data())

    # y = branchy.BranchEntropyMatrix(x, tf.keras.datasets.cifar10.load_data())


    #print the model structure summary
    # x.summary()
    #eval the model
    # branchy.eval_branches(x, tf.keras.datasets.cifar10.load_data())
    # output_names = [i.name for i in x.outputs]
    # print(output_names)
    # y = branchy.evalBranchMatrix(x, tf.keras.datasets.cifar10.load_data())
    #print the results

    pass