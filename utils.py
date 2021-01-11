import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
import itertools
import glob
import os
# from keras.models import load_model
# from keras.utils import CustomObjectScope
# from keras.initializers import glorot_uniform

import math
import pydot
import os
#os.environ["PATH"] += os.pathsep + "C:\Program Files\Graphviz\bin"
#from tensorflow.keras.utils import plot_model

MODEL_DIR = "models/"

#Visualize Model
def visualize_model(model,name=""):
    # tf.keras.utils.plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    if name == "":
        name = "model_plot.png"
    else: 
        name = name + ".png"
    #plot_model(model, to_file=name, show_shapes=True, show_layer_names=True)

def fullprint(*args, **kwargs):
        from pprint import pprint
        import numpy
        opt = numpy.get_printoptions()
        numpy.set_printoptions(threshold=numpy.inf)
        pprint(*args, **kwargs)
        numpy.set_printoptions(**opt)



def calcEntropy(y_hat):
        #entropy is the sum of y * log(y) for all possible labels.
        results =[]
        entropy = 0
        for i in range(len(y_hat)):
            entropy += y_hat[i] * math.log(y_hat(i))
            print(entropy)

        return results

from scipy.special import (comb, chndtr, entr, rel_entr, xlogy, ive)
def entropy(pk, qk=None, base=None):
    #taken from branchynet github
    """Calculate the entropy of a distribution for given probability values.

    If only probabilities `pk` are given, the entropy is calculated as
    ``S = -sum(pk * log(pk), axis=0)``.

    If `qk` is not None, then compute the Kullback-Leibler divergence
    ``S = sum(pk * log(pk / qk), axis=0)``.

    This routine will normalize `pk` and `qk` if they don't sum to 1.

    Parameters
    ----------
    pk : sequence
        Defines the (discrete) distribution. ``pk[i]`` is the (possibly
        unnormalized) probability of event ``i``.
    qk : sequence, optional
        Sequence against which the relative entropy is computed. Should be in
        the same format as `pk`.
    base : float, optional
        The logarithmic base to use, defaults to ``e`` (natural logarithm).

    Returns
    -------
    S : float
        The calculated entropy.

    """
    pk = np.asarray(pk)
    print(pk)
    print(1.0*pk)
    print(np.sum(pk,axis=0))
    pk = 1.0*pk / np.sum(pk, axis=0)
    print(pk)
    if qk is None:
        vec = entr(pk)
    else:
        qk = np.asarray(qk)
        if len(qk) != len(pk):
            raise ValueError("qk and pk must have same length.")
        qk = 1.0*qk / np.sum(qk, axis=0)
        vec = rel_entr(pk, qk)
    print(vec)
    S = np.sum(vec, axis=0)
    if base is not None:
        S /= math.log(base)
    return S



def saveModel(model,name,overwrite = True, includeDate= True, folder ="models", fileFormat = "hdf5"):
    from datetime import datetime
    import os
    now = datetime.now() # current date and time
    stringName =""
    date =""
    if not os.path.exists(folder):
        try:
            os.mkdir(folder)
        except FileExistsError:
            pass
    try:
        if includeDate:
            date =now.strftime("%y-%m-%d_%H%M%S")

        stringName = "{}{}_{}.{}".format(folder+"\\",name,date,fileFormat)
        model.save(stringName, save_format="fileFormat")
        print("saved Model:{}".format(stringName))
    except OSError:
        pass

    return stringName

def printTestScores(test_scores,num_outputs):
    print("overall loss: {}".format(test_scores[0]))
    if num_outputs > 1:
        for i in range(num_outputs):
            print("Output {}: Test loss: {}, Test accuracy {}".format(i, test_scores[i+1], test_scores[i+1+num_outputs]))
    else:
        print("Test loss:", test_scores[0])
        print("Test accuracy:", test_scores[1])


#https://github.com/keras-team/keras/issues/341
def reset_model_weights(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model): #if you're using a model as a layer
            reset_weights(layer) #apply function recursively
            continue

        #where are the initializers?
        if hasattr(layer, 'cell'):
            init_container = layer.cell
        else:
            init_container = layer

        for key, initializer in init_container.__dict__.items():
            if "initializer" not in key: #is this item an initializer?
                  continue #if no, skip it

            # find the corresponding variable, like the kernel or the bias
            if key == 'recurrent_initializer': #special case check
                var = getattr(init_container, 'recurrent_kernel')
            else:
                var = getattr(init_container, key.replace("_initializer", ""))

            var.assign(initializer(var.shape, var.dtype))
            #use the initializer

def reset_layer_weights(layer):
    """ reset the weights for a specific layer.
    """

    #where are the initializers?
    if hasattr(layer, 'cell'):
        init_container = layer.cell
    else:
        init_container = layer

    for key, initializer in init_container.__dict__.items():
        if "initializer" not in key: #is this item an initializer?
                continue #if no, skip it

        # find the corresponding variable, like the kernel or the bias
        if key == 'recurrent_initializer': #special case check
            var = getattr(init_container, 'recurrent_kernel')
        else:
            var = getattr(init_container, key.replace("_initializer", ""))

        var.assign(initializer(var.shape, var.dtype))
        #use the initializer

def reset_branch_weights(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model): #if you're using a model as a layer
            reset_branch_weights(layer) #apply function recursively
            continue
        if "branch" in layer.name:
            print("reseting weights for {}".format(layer.name))
             #where are the initializers?
            if hasattr(layer, 'cell'):
                init_container = layer.cell
            else:
                init_container = layer

            for key, initializer in init_container.__dict__.items():
                if "initializer" not in key: #is this item an initializer?
                    continue #if no, skip it

                # find the corresponding variable, like the kernel or the bias
                if key == 'recurrent_initializer': #special case check
                    var = getattr(init_container, 'recurrent_kernel')
                else:
                    var = getattr(init_container, key.replace("_initializer", ""))

                var.assign(initializer(var.shape, var.dtype))
                #use the initializer
        else: 
            pass

def newBranch_flatten(prevLayer):
    """ Add a new branch to a model connecting at the output of prevLayer. 
        NOTE: use the substring "branch" in all names for branch nodes. this is used as an identifier of the branching layers as opposed to the main branch layers for training
    """ 
    branchLayer = layers.Flatten(name=tf.compat.v1.get_default_graph().unique_name("branch_flatten"))(prevLayer)
    branchLayer = layers.Dense(124, activation="relu",name=tf.compat.v1.get_default_graph().unique_name("branch124"))(branchLayer)
    branchLayer = layers.Dense(64, activation="relu",name=tf.compat.v1.get_default_graph().unique_name("branch64"))(branchLayer)
    branchLayer = layers.Dense(10, name=tf.compat.v1.get_default_graph().unique_name("branch_output"))(branchLayer)
    output = (layers.Softmax(name=tf.compat.v1.get_default_graph().unique_name("branch_softmax"))(branchLayer))

    return output


def newBranch(prevLayer):
    """ Add a new branch to a model connecting at the output of prevLayer. 
        NOTE: use the substring "branch" in all names for branch nodes. this is used as an identifier of the branching layers as opposed to the main branch layers for training
    """ 
    branchLayer = layers.Dense(124, activation="relu",name=tf.compat.v1.get_default_graph().unique_name("branch124"))(prevLayer)
    branchLayer = layers.Dense(64, activation="relu",name=tf.compat.v1.get_default_graph().unique_name("branch64"))(branchLayer)
    branchLayer = layers.Dense(10, name=tf.compat.v1.get_default_graph().unique_name("branch_output"))(branchLayer)
    output = (layers.Softmax(name=tf.compat.v1.get_default_graph().unique_name("branch_softmax"))(branchLayer))

    return output


def newestModelPath(modelNames):
    """ returns the path of the newest model with modelname in the filename.
        does not check the actual model name allocated in the file.
        janky, but works
    """
    # if modelNames is not list:
        # modelNames = [modelNames]
    full_list = os.scandir(MODEL_DIR)
    # print(modelNames)
    # print(full_list)
    items = []
    for i in full_list:
        if modelNames in i.name:
            # print(i.name)
            # print(os.path.getmtime(i.path))
            items.append(i)
    
    items.sort(key=lambda x: os.path.getmtime(x.path), reverse=True)
    result = items[0].path

    return result