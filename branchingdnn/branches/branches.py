# import the necessary packages
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
import itertools
import time
import json

import branchingdnn
from branchingdnn.utils import *

class branch:
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
            customBranch = [branch.newBranch_flatten]
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


    class LogisticEndpoint(keras.layers.Layer):
        def __init__(self, name=None):
            super(branch.LogisticEndpoint, self).__init__(name=name)
            self.loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
            self.accuracy_fn = keras.metrics.BinaryAccuracy()

        def call(self, targets, logits, sample_weights=None):
            # Compute the training-time loss value and add it
            # to the layer using `self.add_loss()`.
            loss = self.loss_fn(targets, logits, sample_weights)
            self.add_loss(loss)

            # Log accuracy as a metric and add it
            # to the layer using `self.add_metric()`.
            acc = self.accuracy_fn(targets, logits, sample_weights)
            self.add_metric(acc, name="accuracy")

            # Return the inference-time prediction tensor (for `.predict()`).
            return tf.nn.softmax(logits)         
                
    class BranchEndpoint(keras.layers.Layer):
        def __init__(self, name=None):
            super(branch.BranchEndpoint, self).__init__(name=name)
            # self.loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
            self.loss_coefficient = 1
            self.feature_loss_coefficient = 1
            self.kl = tf.keras.losses.KLDivergence()
            self.loss_fn = keras.losses.sparse_categorical_crossentropy

        def call(self, inputs, labels, teacher_sm=None, sample_weights=None):
            # Compute the training-time loss value and add it
            # to the layer using `self.add_loss()`.
            #loss functions are (True, Prediction)
            softmax = tf.nn.softmax(inputs)
            
            #loss 1. normal loss, predictions vs labels
            normal_loss = self.loss_fn(labels, softmax, sample_weights)
            # self.add_loss(normal_loss) 
            # currently turned off, the normal loss is still managed by the compile function. use this if no default loss is defined.
            #loss 2. KL divergence loss, aka the difference between the student and teacher's softmax
            if teacher_sm is not None:
                kl_loss = self.kl(softmax,teacher_sm)
                self.add_loss(kl_loss)
                self.add_metric(kl_loss, name=self.name+"_KL")
            #NOTE
            # The total loss is different from parts_loss because it includes the regularization term.
            # In other words, loss is computed as loss = parts_loss + k*R, where R is the regularization term 
            # (typically the L1 or L2 norm of the model's weights) and k a hyperparameter that controls the 
            # contribution of the regularization loss in the total loss.
            return softmax
        
        
        
    class FeatureDistillation(keras.layers.Layer):
        def __init__(self, name=None):
            super(branch.FeatureDistillation, self).__init__(name=name)
            self.loss_coefficient = 1
            self.feature_loss_coefficient = 0.3
        def call(self, inputs, teaching_features, sample_weights=None):
            #loss 3. Feature distillation of the difference between features of teaching layer and student layer.
            l2_loss = self.feature_loss_coefficient * tf.reduce_sum(tf.square(inputs - teaching_features))
            self.add_loss(l2_loss)
            self.add_metric(l2_loss, name=self.name+"_distill") # metric so this loss value can be monitored.
            return inputs


    def add_distil(model, identifier =[""], customBranch = [],exact = True):
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


        inputs = []
        for i in model.inputs:
            inputs.append(i)
        inputs.append(keras.Input(shape=(1,), name="targets")) #shape is (1,) for sparse_categorical_crossentropy
        #add targets as an input to the model so it can be used for the custom losses.
        #   input size is the size of the     
        #add target input 
        model = keras.Model(inputs=inputs, outputs=outputs)

        model.summary()


        targets = model.get_layer('targets').output
        print("targets:", targets)
        teacher_softmax = outputs[0]
        print("teacher_softmax:", teacher_softmax)
        teaching_features = model.get_layer('dense_1').output
        print("teaching Feature:", teaching_features)
        #get the loss from the main exit and combine it with the loss of the 
        old_output = outputs
        # outputs.append(i in model.outputs) #get model outputs that already exist 

        if type(identifier) != list:
            identifier = [identifier]

        if type(customBranch) != list:
            customBranch = [customBranch]
        if len(customBranch) == 0:
            customBranch = [branch.newBranch_distil]
        branches = 0
        # print(customBranch)
        if len(identifier) > 0:
            print(">0")
            if type(identifier[0]) == int:
                print("int")
                for i in identifier: 
                    print(model.layers[i].name)
                    try:
                        outputs.append(customBranch[min(branches, len(customBranch))-1](model.layers[i].output,targets = targets, teacher_sm = teacher_softmax, teaching_features = teaching_features))
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
                            outputs.append(customBranch[min(branches, len(customBranch))-1](model.layers[i].output,targets = targets, teacher_sm = teacher_softmax, teaching_features = teaching_features))
                            branches=branches+1
                            # outputs = newBranch(model.layers[i].output,outputs)
                    else:
                        if any(id in model.layers[i].name for id in identifier):
                            print("add Branch")
                            # print(customBranch[min(i, len(customBranch))-1])
                            # print(min(i, len(customBranch))-1)
                            outputs.append(customBranch[min(branches, len(customBranch))-1](model.layers[i].output,targets = targets, teacher_sm = teacher_softmax, teaching_features = teaching_features))
                            branches=branches+1
                            # outputs = newBranch(model.layers[i].output,outputs)
        else: #if identifier is blank or empty
            print("nothing")
            for i in range(1-len(model.layers)-1):
                print(model.layers[i].name)
                # if "dense" in model.layers[i].name:
                # outputs = newBranch(model.layers[i].output,outputs)
                outputs = customBranch[min(branches, len(customBranch))-1](model.layers[i].output,targets = targets, teacher_sm = teacher_softmax, teaching_features = teaching_features)
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


    # need a bottleneck layer to squeeze the feature hints down to a viable size.
    def newBranch_distil(prevLayer, targets, teacher_sm, teaching_features):
        print("targets::::",targets)
        print("teacher_sm::::",teacher_sm)
        print("teaching_features::::",teaching_features)
        # if teaching_features is not None:
        #     bottle_neck = branch.bottleneck(prevLayer,teaching_features)
        #     branchLayer = branch.FeatureDistillation(name=tf.compat.v1.get_default_graph().unique_name("branch_teaching"))(bottle_neck,teaching_features)    
        #     branchLayer = layers.Flatten(name=tf.compat.v1.get_default_graph().unique_name("branch_flatten"))(branchLayer)
        # else:
        #     print("no teaching feature Provided, bottleneck and teaching loss skipped")
        branchLayer = layers.Flatten(name=tf.compat.v1.get_default_graph().unique_name("branch_flatten"))(prevLayer)

        branchLayer = layers.Dense(124, activation="relu",name=tf.compat.v1.get_default_graph().unique_name("branch124"))(branchLayer)
        branchLayer = layers.Dense(64, activation="relu",name=tf.compat.v1.get_default_graph().unique_name("branch64"))(branchLayer)
        branchLayer = layers.Dense(10, name=tf.compat.v1.get_default_graph().unique_name("branch_output"))(branchLayer)
        output = branch.BranchEndpoint(name=tf.compat.v1.get_default_graph().unique_name("branch_softmax"))(branchLayer, targets, teacher_sm)
        # output = (layers.Softmax(name=tf.compat.v1.get_default_graph().unique_name("branch_softmax"))(branchLayer))

        return output

    #build from LinfengZhang self distillation bottleneck code
    #  Idea is to squeeze the teaching feature set to the same size as the previousLayer,
    #  this is then provided as a part of the overall loss of the branch classifier.
    #bottleneck expands the previous layer to match the feature size of the teaching layer so comparision can be made.
    # I wonder if the expansion should be just done for the teaching process, not actually passed to the rest of the branch. 
    # so the bottleneck's expanded features are not actually passed on, just used to see if they are on the right track.

    def bottleneck(prevLayer, featureLayer):
        base_width = 64
        groups = 1
        stride = 1  
        print(prevLayer)
        print(featureLayer)
        if len(featureLayer.shape)>2: #mutli-dimensional layer output, aka not a dense fullyconnected layer.
            filters = featureLayer.shape[3]    
        else:
            filters = featureLayer.shape[1]
        planes = featureLayer.shape[1]
        
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        bottleneck = layers.Conv2D(filters, (1,1),activation='relu')(prevLayer)
        # self.bn1 = norm_layer(width)
        bottleneck = layers.BatchNormalization()(bottleneck)
        # self.conv2 = conv3x3(width, width, stride, groups, dilation)
        bottleneck = layers.Conv2D(filters,(3,3),activation='relu')(bottleneck)
        # self.bn2 = norm_layer(width)
        bottleneck = layers.BatchNormalization()(bottleneck)
        # self.conv3 = conv1x1(width, planes * self.expansion)
        bottleneck = layers.Conv2D(filters, (1,1),activation='relu')(bottleneck)
        # self.bn3 = norm_layer(planes * self.expansion)
        bottleneck = layers.BatchNormalization()(bottleneck)
        # self.relu = nn.ReLU(inplace=True)
        bottleneck = layers.ReLU()(bottleneck)
        # self.downsample = downsample
        # self.stride = stride



        return bottleneck

