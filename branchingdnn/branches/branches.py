# import the necessary packages
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
import itertools
import time
import json
from utils import *

class branches:
    #add a branch
    def add(model, identifier =[""], customBranch = [],exact = True):
        """ add branches to the provided model, aka modifying an existing model to include branches.
            identifier: takes a list of names of layers to branch on is blank, branches will be added to all layers except the input and final layer. Can be a list of layer numbers, following the numbering format of model.layers[]
            If identifier is not blank, a branch will be added to each layer with identifier in its name. (identifier = "dense", all dense layers will be branched.)
            Warning! individual layers are defined according to how TF defines them. this means that for layers that would be normally grouped, they will be treated as individual layers (conv2d, pooling, flatten, etc)
            customBranch: optional function that can be passed to provide a custom branch to be inserted. Check "newBranch" function for default shape of branches and how to build custom branching function. Can be provided as a list and each branch will iterate through provided customBranches, repeating last the last branch until function completes
        """
        
        # model = keras.Model([model.input], [model_old.output], name="{}_branched".format(model_old.name))
        # model.summary()

        # outputs = [model.outputs]
        # outputs.append(newBranch(model.layers[6].output))
        # new_model = keras.Model([model.input], outputs, name="{}_branched".format(model.name))
        # new_model.summary()
        outputs = []
        for i in model.outputs:
            outputs.append(i)

        old_output = outputs
        # outputs.append(i in model.outputs) #get model outputs that already exist 

        if type(identifier) != list:
            identifier = [identifier]

        if type(customBranch) != list:
            customBranch = [customBranch]
        if len(customBranch) == 0:
            customBranch = [newBranch]
        branches = 0
        # print(customBranch)
        if len(identifier) > 0:
            print(">0")
            if type(identifier[0]) == int:
                print("int")
                for i in identifier: 
                    print(model.layers[i].name)
                    try:
                        outputs.append(customBranch[min(branches, len(customBranch))-1](model.layers[i].output))
                        branches=branches+1
                        # outputs = newBranch(model.layers[i].output,outputs)
                    except:
                        pass
            else:
                print("abc")
                for i in range(len(model.layers)):
                    print(model.layers[i].name)
                    if exact == True:
                        if model.layers[i].name in identifier:
                            print("add Branch")
                            # print(customBranch[min(i, len(customBranch))-1])
                            # print(min(i, len(customBranch))-1)
                            outputs.append(customBranch[min(branches, len(customBranch))-1](model.layers[i].output))
                            branches=branches+1
                            # outputs = newBranch(model.layers[i].output,outputs)
                    else:
                        if any(id in model.layers[i].name for id in identifier):
                            print("add Branch")
                            # print(customBranch[min(i, len(customBranch))-1])
                            # print(min(i, len(customBranch))-1)
                            outputs.append(customBranch[min(branches, len(customBranch))-1](model.layers[i].output))
                            branches=branches+1
                            # outputs = newBranch(model.layers[i].output,outputs)
        else: #if identifier is blank or empty
            print("nothing")
            for i in range(1-len(model.layers)-1):
                print(model.layers[i].name)
                # if "dense" in model.layers[i].name:
                # outputs = newBranch(model.layers[i].output,outputs)
                outputs = customBranch[min(branches, len(customBranch))-1](model.layers[i].output,outputs)
                branches=branches+1
            # for j in range(len(model.layers[i].inbound_nodes)):
            #     print(dir(model.layers[i].inbound_nodes[j]))
            #     print("inboundNode: " + model.layers[i].inbound_nodes[j].name)
            #     print("outboundNode: " + model.layers[i].outbound_nodes[j].name)
        print(outputs)
        print(model.input)
        # input_layer = layers.Input(batch_shape=model.layers[0].input_shape)
        model = models.Model([model.input], [outputs], name="{}_branched".format(model.name))
        return model
    def newBranchCustom(prevLayer, outputs=[]):
        """ example of a custom branching layer, used as a drop in replacement of "newBranch"
        """                 
        branchLayer = layers.Dense(10, name=tf.compat.v1.get_default_graph().unique_name("branch_output"))(prevLayer)
        outputs.append(layers.Softmax(name=tf.compat.v1.get_default_graph().unique_name("branch_softmax"))(branchLayer))

        return outputs

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

        
    def newBranch_flatten_alt(prevLayer):
        """ Add a new branch to a model connecting at the output of prevLayer. 
            NOTE: use the substring "branch" in all names for branch nodes. this is used as an identifier of the branching layers as opposed to the main branch layers for training
        """ 
        branchLayer = layers.Flatten(name=tf.compat.v1.get_default_graph().unique_name("branch_flatten"))(prevLayer)
        branchLayer = layers.Dense(4096, activation="relu",name=tf.compat.v1.get_default_graph().unique_name("branch4096"))(branchLayer)
        branchLayer = keras.layers.Dropout(0.5,name=tf.compat.v1.get_default_graph().unique_name("branch_dropout"))(branchLayer)
        branchLayer = layers.Dense(2048, activation="relu",name=tf.compat.v1.get_default_graph().unique_name("branch2048"))(branchLayer)
        branchLayer = keras.layers.Dropout(0.5,name=tf.compat.v1.get_default_graph().unique_name("branch_dropout"))(branchLayer)
        branchLayer = layers.Dense(10, name=tf.compat.v1.get_default_graph().unique_name("branch_output"))(branchLayer)
        output = (layers.Softmax(name=tf.compat.v1.get_default_graph().unique_name("branch_softmax"))(branchLayer))

        return output

    def newBranch_resnet(prevLayer):
        """ Add a new branch to a model connecting at the output of prevLayer. 
            NOTE: use the substring "branch" in all names for branch nodes. this is used as an identifier of the branching layers as opposed to the main branch layers for training
        """ 
        branchLayer = layers.Flatten(name=tf.compat.v1.get_default_graph().unique_name("branch_flatten"))(prevLayer)
        # branchLayer = keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(227,227,3))(branchLayer)
        # branchLayer = keras.layers.BatchNormalization()(branchLayer)
        # branchLayer = keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2))(branchLayer)
        # branchLayer = layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(227,227,3))(branchLayer)
        # branchLayer = layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(227,227,3))(branchLayer)
        branchLayer = layers.Dense(2048, name=tf.compat.v1.get_default_graph().unique_name("branch_2048"))(branchLayer)
        branchLayer = layers.Dense(1000, name=tf.compat.v1.get_default_graph().unique_name("branch_output"))(branchLayer)
        output = (layers.Softmax(name=tf.compat.v1.get_default_graph().unique_name("branch_softmax"))(branchLayer))

        return output

    def newBranch_flatten100(prevLayer):
        """ Add a new branch to a model connecting at the output of prevLayer. 
            NOTE: use the substring "branch" in all names for branch nodes. this is used as an identifier of the branching layers as opposed to the main branch layers for training
        """ 
        branchLayer = layers.Flatten(name=tf.compat.v1.get_default_graph().unique_name("branch_flatten"))(prevLayer)
        branchLayer = layers.Dense(256, activation="relu",name=tf.compat.v1.get_default_graph().unique_name("branch124"))(branchLayer)
        branchLayer = layers.Dense(124, activation="relu",name=tf.compat.v1.get_default_graph().unique_name("branch64"))(branchLayer)
        branchLayer = layers.Dense(100, name=tf.compat.v1.get_default_graph().unique_name("branch_output"))(branchLayer)
        output = (layers.Softmax(name=tf.compat.v1.get_default_graph().unique_name("branch_softmax"))(branchLayer))

        return output

    def newBranch_oneLayer(prevLayer):
        """ Add a new branch to a model connecting at the output of prevLayer. 
            NOTE: use the substring "branch" in all names for branch nodes. this is used as an identifier of the branching layers as opposed to the main branch layers for training
        """ 
        # branchLayer = layers.Flatten(name=tf.compat.v1.get_default_graph().unique_name("branch_flatten"))(prevLayer)
        branchLayer = layers.Dense(10, name=tf.compat.v1.get_default_graph().unique_name("branch_output"))(prevLayer)
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
