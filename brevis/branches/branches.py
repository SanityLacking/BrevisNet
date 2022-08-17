""" Branch model for use in Brevis Branching Models """
# import the necessary packages
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
import itertools
import time
import json

from tensorflow.python.util.tf_export import KERAS_API_NAME


import brevis
from brevis.utils import *

class branch:
    #add a branch
    def add(model,  custom_branch = [], identifier =[""],exact = True, target_input= True, compact = False, num_outputs=10):
        """ add branches to the provided model, aka modifying an existing model to include branches.
            identifier: takes a list of names of layers to branch on is blank, branches will be added to all layers except the input and final layer. Can be a list of layer numbers, following the numbering format of model.layers[]
            If identifier is not blank, a branch will be added to each layer with identifier in its name. (identifier = "dense", all dense layers will be branched.)
            Warning! individual layers are defined according to how TF defines them. this means that for layers that would be normally grouped, they will be treated as individual layers (conv2d, pooling, flatten, etc)
            custom_branch: optional function that can be passed to provide a custom branch to be inserted. Check "newBranch" function for default shape of branches and how to build custom branching function. Can be provided as a list and each branch will iterate through provided custom_branches, repeating last the last branch until function completes
        """
        outputs = []
        for i in model.outputs:
            outputs.append(i)
        
        inputs = []
        ready = False
        
        targets= None
        
        for i in model.inputs:
            if i.name == "targets":
                ready = True
            inputs.append(i)
        if target_input:
            print("targets already present? ",ready)

            if not ready:
                print("added targets")
                targets = keras.Input(shape=(num_outputs,), name="targets")
                inputs.append(targets) #shape is (1,) for sparse_categorical_crossentropy
            else:
                targets = model.get_layer('targets').output

        #add targets as an input to the model so it can be used for the custom losses.
        new_model = brevis.BranchModel(inputs=inputs, outputs=outputs,name = model.name, transfer=model.transfer, custom_objects=model.custom_objects)

        if type(identifier) != list:
            identifier = [identifier]

        if type(custom_branch) != list:
            custom_branch = [custom_branch]
        if len(custom_branch) == 0:
            return new_model    
        branches = 0
        if len(identifier) > 0:
            print("Matching Branchpoint by id number")
            if type(identifier[0]) == int:
                for i in identifier: 
                    try:
                        outputs.append(custom_branch[min(branches, len(custom_branch))-1](model.layers[i].output,targets = targets))
                        branches=branches+1
                    except:
                        pass
            else:
                print("Matching Branchpoint by name")
                for i in range(len(model.layers)):
                    if exact == True:
                        if model.layers[i].name in identifier:
                            print("add Branch to branch point ",model.layers[i].name)
                            outputs.append(custom_branch[min(branches, len(custom_branch)-1)](model.layers[i].output,targets = targets))
                            branches=branches+1
                    else:
                        if any(id in model.layers[i].name for id in identifier):
                            print("add Branch to branch point ",model.layers[i].name)
                            outputs.append(custom_branch[min(branches, len(custom_branch)-1)](model.layers[i].output,targets = targets))
                            branches=branches+1
        else: #if identifier is blank or empty
            for i in range(1-len(model.layers)-1):
                outputs = custom_branch[min(branches, len(custom_branch))-1](model.layers[i].output,outputs,targets = targets)
                branches=branches+1
        print(new_model.input)
        print(outputs)
        new_model = brevis.BranchModel([new_model.input], [outputs], name = new_model.name, custom_objects=new_model.custom_objects)
        return new_model
    
    
    
    def add_loop(model, custom_branch = [], identifier =[""], exact = True, target_input= True, num_outputs=10):
        """ add branches to the provided model, aka modifying an existing model to include branches.
            identifier: takes a list of names of layers to branch on is blank, branches will be added to all layers except the input and final layer. Can be a list of layer numbers, following the numbering format of model.layers[]
            If identifier is not blank, a branch will be added to each layer with identifier in its name. (identifier = "dense", all dense layers will be branched.)
            Warning! individual layers are defined according to how TF defines them. this means that for layers that would be normally grouped, they will be treated as individual layers (conv2d, pooling, flatten, etc)
            custom_branch: optional function that can be passed to provide a custom branch to be inserted. Check "newBranch" function for default shape of branches and how to build custom branching function. Can be provided as a list and each branch will iterate through provided custom_branches, repeating last the last branch until function completes
        """
        layers = [l for l in model.layers]
        outputs = []
        for i in model.outputs:
            outputs.append(i)
        inputs = []
        ready = False
        targets= None
        for i in model.inputs:
            if i.name == "targets":
                ready = True
            inputs.append(i)
        if target_input:
            print("targets already present? ",ready)
            if not ready:
                print("added targets")
                targets = keras.Input(shape=(num_outputs,), name="targets")
                inputs.append(targets) #shape is (1,) for sparse_categorical_crossentropy
            else:
                targets = model.get_layer('targets').output

        new_model = brevis.BranchModel(inputs=inputs, outputs=outputs,name = model.name, transfer=model.transfer, custom_objects=model.custom_objects)
        old_output = outputs
        if type(identifier) != list:
            identifier = [identifier]
        if type(custom_branch) != list:
            custom_branch = [custom_branch]
        if len(custom_branch) == 0:
            custom_branch = [branch.newBranch_flatten]
        branches = 0
        # print(custom_branch)
        if len(identifier) > 0:
            # print("Matching Branchpoint by id number")
            if type(identifier[0]) == int:
                for i in identifier: 
                    try:
                        outputs.append(custom_branch[min(branches, len(custom_branch))-1](new_model.layers[i].output,targets = targets))
                        branches=branches+1
                    except:
                        pass
            else:
                print("Matching Branchpoint by name, exact: ",exact)
                x = layers[0].output
                for i in range(1,len(layers)):
                # for i, layer in enumerate(new_model.layers):
                # for i in range(len(new_model.layers)):
                    if exact == True:
                        
                        if layers[i].name in identifier:
                            print("add Branch to branch point ",layers[i].name)
                            x = layers[i](x)
                            new_branch = custom_branch[min(branches, len(custom_branch)-1)](x,targets = targets)
                            x = branch.branch_finished(0,name=tf.compat.v1.get_default_graph().unique_name("branch_finished"))(x,new_branch)
                            outputs.append(new_branch)
                            branches=branches+1
                        else:        
                            x = layers[i](x)
                    else:
                        if any(id in new_model.layers[i].name for id in identifier):
                            print("add Branch to branch point ",layers[i].name)
                            x = layers[i](x)
                            new_branch = custom_branch[min(branches, len(custom_branch)-1)](x,targets = targets)
                            x = branch.branch_finished(0,name=tf.compat.v1.get_default_graph().unique_name("branch_finished"))(x,new_branch)
                            outputs.append(new_branch)
                            branches=branches+1
                        else:
                            x = layers[i](x)
                            
        else: #if identifier is blank or empty
            # print("nothing")
            for i in range(1-len(new_model.layers)-1):
                # print(new_model.layers[i].name)
                # if "dense" in new_model.layers[i].name:
                # outputs = newBranch(new_model.layers[i].output,outputs)
                outputs = custom_branch[min(branches, len(custom_branch))-1](new_model.layers[i].output,outputs,targets = targets)
                branches=branches+1
            # for j in range(len(new_model.layers[i].inbound_nodes)):
            #     print(dir(new_model.layers[i].inbound_nodes[j]))
            #     print("inboundNode: " + new_model.layers[i].inbound_nodes[j].name)
            #     print("outboundNode: " + new_model.layers[i].outbound_nodes[j].name)
        # print(outputs)
        # print(new_model.input)
        # outputs.pop(0)
        # print(outputs)
        # input_layer = layers.Input(batch_shape=new_model.layers[0].input_shape)
        outputs.append(x)
        new_model = brevis.BranchModel([inputs], [outputs], name = new_model.name, transfer = new_model.transfer, custom_objects=new_model.custom_objects)
        new_model.branch_active=model.branch_active        
        # new_model.summary()

        return new_model
   












