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

    class EvidenceEndpoint(keras.layers.Layer):
        def __init__(self, name=None, **kwargs):
            super(branch.EvidenceEndpoint, self).__init__(name=name)
            self.num_outputs = 10
            self.loss_coefficient = 1
            self.feature_loss_coefficient = 1
            self.kl = tf.keras.losses.KLDivergence()
            self.global_step = tf.Variable(initial_value=0.0, name='global_step', trainable=False)
            self.crossE = tf.keras.losses.SparseCategoricalCrossentropy()
            self.loss_fn = utils.loss_function()
            self.temperature = 10
            self.alpha = .1
            self.lmb = 0.005
        def build(self, input_shape):
            self.kernel = self.add_weight("kernel", shape=[int(input_shape[-1]),
                                                    self.num_outputs])

        def call(self, inputs, labels):
            outputs = tf.matmul(inputs,self.kernel)
            # Compute the training-time loss value and add it
            # to the layer using `self.add_loss()`.
            #loss functions are (True, Prediction)
            # softmax = tf.nn.softmax(inputs)
            # print("inputs",input)
            #loss 1. normal loss, predictions vs labels
            evidence, normal_loss = self.loss_fn(labels, outputs)

            # softmax = tf.nn.softmax(outputs)
            # evidence2, normal_loss2 = self.loss_fn(labels, softmax)
            # print(self.weights, self.lmb)
            # l2_loss = tf.nn.l2_loss(self.weights)# * self.lmb
            # print("l2_loss",l2_loss)
            # print(evidence.shape)
            # print(normal_loss)
            total_loss =tf.reduce_mean(normal_loss)# + l2_loss
            # print(total_loss)
            total_evidence = tf.reduce_sum(evidence,1, keepdims=True) 
            # print(total_evidence)
            pred = tf.argmax(outputs, 1)
            # truth = tf.argmax(labels, 1)
            # print("labels",labels)
            truth = tf.cast(labels,tf.int64)
            # print("evid", evidence)
            # print("loss", normal_loss)
            # print("truth",truth)
            # print("pred",pred)
            match = (tf.equal(tf.reshape(truth,(1,32)),pred))
            # match = tf.where(tf.math.equal(pred, tf.cast(tf.reshape(truth,(32,1)),'int64')))
            match = tf.cast(match,tf.float32)
            # match = tf.reshape(tf.cast(tf.equal(pred, truth), tf.float32),(-1,1))
            # print("match",match)
            mean_avg = tf.reduce_mean(total_evidence)
            # print("mean_Avg")
            # print("match",match)
            mean_succ = tf.reduce_sum(tf.reduce_sum(evidence,1, keepdims=True)* match) / tf.reduce_sum(match+1e-20)
            # print("mean_fail")
            # side_x = tf.reduce_sum(tf.reduce_sum(evidence,1, keepdims=True)*(1-match))
            # print("side_x",side_x)
            # side_y = (tf.reduce_sum(tf.abs(1-match))+1e-20)
            # print("side_y",side_y)
            # div = tf.reduce_sum(side_x/side_y)
            # print("div",div)
            mean_fail = tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(evidence,1, keepdims=True)*(1-match)) / (tf.reduce_sum(tf.abs(1-match))+1e-20) )
            # print('end')
            self.add_metric(normal_loss, name=self.name+"_normal_loss")
            # self.add_metric(normal_loss2, name=self.name+"_softmax_loss")
            # self.add_metric(evidence, name=self.name+"_evidence")
            # self.add_metric(mean_avg, name=self.name+"_mean_ev_avg")
            # self.add_metric(mean_succ, name=self.name+"_mean_ev_succ")
            # self.add_metric(mean_fail, name=self.name+"_mean_ev_fail")
            # print('metrics')
            self.add_loss(normal_loss)
            
            # CrossE_loss= self.crossE(labels,softmax)
            # self.add_loss(CrossE_loss)

            # print("pred",inputs)
            # print("loss")
            #NOTE
            # The total loss is different from parts_loss because it includes the regularization term.
            # In other words, loss is computed as loss = parts_loss + k*R, where R is the regularization term 
            # (typically the L1 or L2 norm of the model's weights) and k a hyperparameter that controls the 
            # contribution of the regularization loss in the total loss.
            return outputs

    class LogisticEndpoint(keras.layers.Layer):
        def __init__(self, name=None, **kwargs):
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
        def __init__(self, name=None, **kwargs):
            super(branch.BranchEndpoint, self).__init__(name=name)
            # self.loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
            self.loss_coefficient = 1
            self.feature_loss_coefficient = 1
            self.kl = tf.keras.losses.KLDivergence()
            self.loss_fn = keras.losses.sparse_categorical_crossentropy
            self.temperature = 10
            self.alpha = .1

        def call(self, inputs, labels, teacher_sm=None, sample_weights=None):
            # Compute the training-time loss value and add it
            # to the layer using `self.add_loss()`.
            #loss functions are (True, Prediction)
            softmax = tf.nn.softmax(inputs)
            
            #loss 1. normal loss, predictions vs labels
            normal_loss = self.loss_fn(labels, softmax, sample_weights)
            total_loss =tf.reduce_mean(normal_loss)
            # print("normal loss",normal_loss)
            # currently turned off, the normal loss is still managed by the compile function. use this if no default loss is defined.
            #loss 2. KL divergence loss, aka the difference between the student and teacher's softmax
            if teacher_sm is not None:
                kl_loss = self.kl( tf.nn.softmax(softmax / self.temperature, axis = 1 ),
                tf.nn.softmax(teacher_sm /self.temperature,axis=1))
                # print("KL_LOSS", kl_loss)
                # self.add_loss(kl_loss)
                total_loss += self.alpha * total_loss + (1- self.alpha) * kl_loss
                self.add_metric(kl_loss, name=self.name+"_KL")
                        
            self.add_loss(total_loss)
            self.add_metric(tf.reduce_sum(self.losses), name=self.name+"_losses")
            #NOTE
            # The total loss is different from parts_loss because it includes the regularization term.
            # In other words, loss is computed as loss = parts_loss + k*R, where R is the regularization term 
            # (typically the L1 or L2 norm of the model's weights) and k a hyperparameter that controls the 
            # contribution of the regularization loss in the total loss.
            return softmax
        
        
        
    class FeatureDistillation(keras.layers.Layer):
        def __init__(self, name=None, **kwargs):
            super(branch.FeatureDistillation, self).__init__(name=name)
            self.loss_coefficient = 1
            self.feature_loss_coefficient = 0.3
        def call(self, inputs, teaching_features, sample_weights=None):
            #loss 3. Feature distillation of the difference between features of teaching layer and student layer.
            # print("input",inputs)
            # print("teaching",teaching_features)
            l2_loss = self.feature_loss_coefficient * tf.reduce_sum(tf.square(inputs - teaching_features))
            self.add_loss(l2_loss)
            self.add_metric(l2_loss, name=self.name+"_distill") # metric so this loss value can be monitored.
            return inputs

    class FeatureDistillation_clear(keras.layers.Layer):
        def __init__(self, name=None, **kwargs):
            super(branch.FeatureDistillation_clear, self).__init__(name=name)
            self.loss_coefficient = 1
            self.feature_loss_coefficient = 0.3
        def call(self, inputs, teaching_features, original_inputs, sample_weights=None):
            #loss 3. Feature distillation of the difference between features of teaching layer and student layer.
            # print("input",inputs)
            # print("teaching",teaching_features)
            l2_loss = self.feature_loss_coefficient * tf.reduce_sum(tf.square(inputs - teaching_features))
            self.add_loss(l2_loss)
            self.add_metric(l2_loss, name=self.name+"_distill") # metric so this loss value can be monitored.
            return original_inputs


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
        ready = False
        for i in model.inputs:
            if i.name == "targets":
                ready = True
            inputs.append(i)
        if not ready:
            inputs.append(keras.Input(shape=(10,), name="targets")) #shape is (1,) for sparse_categorical_crossentropy
        #add targets as an input to the model so it can be used for the custom losses.
        #   input size is the size of the     
        #add target input 
        model = keras.Model(inputs=inputs, outputs=outputs)

        model.summary()
        # outputs = []

        targets = model.get_layer('targets').output
        print("targets:", targets)
        # teacher_softmax = outputs[0]
        teaching_features = None

        # print("teacher_softmax:", teacher_softmax)
        # teaching_features = [model.get_layer('max_pooling2d_1').output, model.get_layer('max_pooling2d_2').output, model.get_layer('max_pooling2d_2').output]
        # print("teaching Feature:", teaching_features)
        teacher_softmax = [None]
        if type(teaching_features) != list:
            teaching_features = [teaching_features]
        #get the loss from the main exit and combine it with the loss of the 
        old_output = outputs
        # outputs.append(i in model.outputs) #get model outputs that already exist 
        teaching_features= [None]
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

                        outputs.append(customBranch[min(branches, len(customBranch))-1](model.layers[i].output,targets = targets,
                                 teacher_sm = teacher_softmax, teaching_features = teaching_features[min(branches, len(teaching_features))-1]))
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
                            # print("test",teaching_features[min(branches, len(teaching_features))-1])

                            outputs.append(customBranch[min(branches, len(customBranch))-1](model.layers[i].output,targets = targets, teacher_sm = teacher_softmax, teaching_features = teaching_features[min(branches, len(teaching_features))-1]))
                            branches=branches+1
                            # outputs = newBranch(model.layers[i].output,outputs)
                    else:
                        if any(id in model.layers[i].name for id in identifier):
                            print("add Branch")
                            # print(customBranch[min(i, len(customBranch))-1])
                            # print(min(i, len(customBranch))-1)
                            outputs.append(customBranch[min(branches, len(customBranch))-1](model.layers[i].output,targets = targets, teacher_sm = teacher_softmax, teaching_features = teaching_features[min(branches, len(teaching_features))-1]))
                            branches=branches+1
                            # outputs = newBranch(model.layers[i].output,outputs)
        else: #if identifier is blank or empty
            print("nothing")
            for i in range(1-len(model.layers)-1):
                print(model.layers[i].name)
                # if "dense" in model.layers[i].name:
                # outputs = newBranch(model.layers[i].output,outputs)
                outputs = customBranch[min(branches, len(customBranch))-1](model.layers[i].output,targets = targets, teacher_sm = teacher_softmax, teaching_features = teaching_features[min(branches, len(teaching_features))-1])
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
    def newBranch_distil(prevLayer, targets, teacher_sm, teaching_features=None):
        print("targets::::",targets)
        print("teacher_sm::::",teacher_sm)
        print("teaching_features::::",teaching_features)
        if prevLayer.shape[1] == 4096:
            #don't add a feature distil to the last branch
            teaching_features = None
        if teaching_features is not None:
            branchLayer = branch.bottleneck(prevLayer,teaching_features)
            branchLayer = branch.FeatureDistillation(name=tf.compat.v1.get_default_graph().unique_name("branch_teaching"))(branchLayer,teaching_features)    
            branchLayer = layers.Flatten(name=tf.compat.v1.get_default_graph().unique_name("branch_flatten"))(branchLayer)
        else:
            # print("no teaching feature Provided, bottleneck and teaching loss skipped")
            branchLayer = branch.bottleneck2(prevLayer)
            branchLayer = layers.Flatten(name=tf.compat.v1.get_default_graph().unique_name("branch_flatten"))(prevLayer)

        branchLayer = layers.Dense(124, activation="relu",name=tf.compat.v1.get_default_graph().unique_name("branch124"))(branchLayer)
        branchLayer = layers.Dense(64, activation="relu",name=tf.compat.v1.get_default_graph().unique_name("branch64"))(branchLayer)
        branchLayer = layers.Dense(10, name=tf.compat.v1.get_default_graph().unique_name("branch_output"))(branchLayer)
        output = branch.BranchEndpoint(name=tf.compat.v1.get_default_graph().unique_name("branch_softmax"))(branchLayer, targets, teacher_sm)
        # output = (layers.Softmax(name=tf.compat.v1.get_default_graph().unique_name("branch_softmax"))(branchLayer))

        return output

        # need a bottleneck layer to squeeze the feature hints down to a viable size.
    def newBranch_bottleneck(prevLayer, targets=None, teacher_sm=None, teaching_features=None):
        print("targets::::",targets)
        print("teacher_sm::::",teacher_sm)
        print("teaching_features::::",teaching_features)
        if prevLayer.shape[1] == 4096:
            #don't add a feature distil to the last branch
            teaching_features = None
        if teaching_features is not None:
            branchLayer = branch.bottleneck(prevLayer,teaching_features)
            branchLayer = branch.FeatureDistillation(name=tf.compat.v1.get_default_graph().unique_name("branch_teaching"))(branchLayer,teaching_features)    
            branchLayer = layers.Flatten(name=tf.compat.v1.get_default_graph().unique_name("branch_flatten"))(branchLayer)
        else:
            # branchLayer = branch.bottleneck2(prevLayer)
            print("no teaching feature Provided, bottleneck and teaching loss skipped")
            branchLayer = layers.Flatten(name=tf.compat.v1.get_default_graph().unique_name("branch_flatten"))(prevLayer)

        branchLayer = layers.Dense(124, activation="relu",name=tf.compat.v1.get_default_graph().unique_name("branch124"))(branchLayer)
        branchLayer = layers.Dense(64, activation="relu",name=tf.compat.v1.get_default_graph().unique_name("branch64"))(branchLayer)
        # branchLayer = layers.Dense(10, name=tf.compat.v1.get_default_graph().unique_name("branch_output"))(branchLayer)
        # if targets is None and teacher_sm is None:
            # output = (layers.Softmax(name=tf.compat.v1.get_default_graph().unique_name("branch_softmax"))(branchLayer))
        # else:
            # output = branch.BranchEndpoint(name=tf.compat.v1.get_default_graph().unique_name("branch_softmax"))(branchLayer, targets, teacher_sm)
        # output = (layers.Softmax(name=tf.compat.v1.get_default_graph().unique_name("branch_softmax"))(branchLayer))
        output = branch.EvidenceEndpoint(name=tf.compat.v1.get_default_graph().unique_name("branch_output"))(branchLayer, targets)

        return output

    def newBranch_bottleneck2(prevLayer):
        branchLayer = branch.bottleneck2(prevLayer)
        print("no teaching feature Provided, bottleneck and teaching loss skipped")
        branchLayer = layers.Flatten(name=tf.compat.v1.get_default_graph().unique_name("branch_flatten"))(branchLayer)

        branchLayer = layers.Dense(124, activation="relu",name=tf.compat.v1.get_default_graph().unique_name("branch124"))(branchLayer)
        branchLayer = layers.Dense(64, activation="relu",name=tf.compat.v1.get_default_graph().unique_name("branch64"))(branchLayer)
        branchLayer = layers.Dense(10, name=tf.compat.v1.get_default_graph().unique_name("branch_output"))(branchLayer)
        output = (layers.Softmax(name=tf.compat.v1.get_default_graph().unique_name("branch_softmax"))(branchLayer))

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
            teacher_size = featureLayer.shape[3]    
        else:
            teacher_size = featureLayer.shape[1]
        planes = featureLayer.shape[1]
        
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1

        print("prev layer", prevLayer)
        print("prev shape", len(prevLayer.shape))
        print("conv shape",teacher_size)
        filt = int(math.sqrt(prevLayer.shape[1]))
        prev_Size= 1
        pool_times = 0
        for i, dim in enumerate(prevLayer.shape):
            if dim is not None:
                prev_Size = prev_Size * dim 
            
        if prev_Size ==69984:
            pool_times = 2
        print("prev size is:",prev_Size)
        bottleneck = prevLayer
        # if len(prevLayer.shape) > 2 :
        #     #reshape the input
        #     bottleneck = layers.Reshape((filt, filt))(prevLayer)
        # else:
        #     bottleneck = layers.Conv2D(256, kernel_size=(1,1),activation='relu', name=tf.compat.v1.get_default_graph().unique_name("branch_bottleneck"))(prevLayer)
        # stride = input / output
        pool_times = 0
        if prevLayer.shape[1] == 27:
            pool_times = 2
        elif prevLayer.shape[1] == 13:
            pool_times = 0
        for i in range(pool_times):
            bottleneck = layers.MaxPool2D(pool_size=(3,3), strides=(2,2),name=tf.compat.v1.get_default_graph().unique_name("branch_MaxPool2D"))(bottleneck)
        # bottleneck = layers.MaxPool2D(pool_size=(3,3), strides=(2,2),name=tf.compat.v1.get_default_graph().unique_name("branch_MaxPool2D"))(bottleneck)
        if pool_times >0:
            bottleneck = keras.layers.Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same",name=tf.compat.v1.get_default_graph().unique_name("branch_conv"))(bottleneck)
        # bottleneck = layers.Flatten(name=tf.compat.v1.get_default_graph().unique_name("branch_flatten"))(bottleneck)
        
        # bottleneck = layers.Reshape((int(bottleneck.shape[1]),1),name=tf.compat.v1.get_default_graph().unique_name("branch_flatten"))(bottleneck)
        # print(bottleneck)

        # bottleneck = layers.MaxPool1D(pool_size=stride, strides=stride, padding='same',name=tf.compat.v1.get_default_graph().unique_name("branch_MaxPool1D_{}".format(stride)))(bottleneck)
        # bottleneck = layers.Dense(teacher_size, name=tf.compat.v1.get_default_graph().unique_name("branch_bottleneck"))(bottleneck)
        # bottleneck = branch.FeatureDistillation(name=tf.compat.v1.get_default_graph().unique_name("branch_teaching"))(bottleneck,featureLayer)    
        
        
        ### Attempt at conv2d  bottleneck 
        # if len(prevLayer.shape) == 2 :
        #     #reshape the input
        #     bottleneck = layers.Reshape((filt, filt))(prevLayer)
        # else:
        #     bottleneck = layers.Conv2D(256, kernel_size=(1,1),activation='relu', name=tf.compat.v1.get_default_graph().unique_name("branch_bottleneck"))(prevLayer)
        # # self.bn1 = norm_layer(width)
        # bottleneck = layers.BatchNormalization(name=tf.compat.v1.get_default_graph().unique_name("branch_norm"))(bottleneck)
        # self.conv2 = conv3x3(width, width, stride, groups, dilation)
        # bottleneck = layers.Conv2D(filters,(3,3),activation='relu')(bottleneck)
        # # self.bn2 = norm_layer(width)
        # bottleneck = layers.BatchNormalization()(bottleneck)
        # # self.conv3 = conv1x1(width, planes * self.expansion)
        # bottleneck = layers.Conv2D(filters, (1,1),activation='relu')(bottleneck)
        # # self.bn3 = norm_layer(planes * self.expansion)
        # bottleneck = layers.BatchNormalization()(bottleneck)
        # # self.relu = nn.ReLU(inplace=True)
        # bottleneck = layers.ReLU()(bottleneck)
        # self.downsample = downsample
        # self.stride = stride



        return bottleneck


    def bottleneck2(prevLayer):
        pool_times = 2
        bottleneck = prevLayer

        for i in range(pool_times):
            bottleneck = layers.MaxPool2D(pool_size=(3,3), strides=(2,2),name=tf.compat.v1.get_default_graph().unique_name("branch_MaxPool2D"))(bottleneck)

        return bottleneck
     # need a bottleneck layer to squeeze the feature hints down to a viable size.
    

    def newBranch_distil_1(prevLayer, targets, teacher_sm, teaching_features):
        print("targets::::",targets)
        print("teacher_sm::::",teacher_sm)
        print("teaching_features::::",teaching_features)
        if prevLayer.shape[1] == 4096:
            #don't add a feature distil to the last branch
            teaching_features = None
        if teaching_features is not None:
            prev_Size= 1
            pool_times = 0
            for i, dim in enumerate(prevLayer.shape):
                if dim is not None:
                    prev_Size = prev_Size * dim 
            if prev_Size ==69984:
                pool_times = 2
            print("prev size is:",prev_Size)
            bottleneck = prevLayer
            pool_times = 0
            if prevLayer.shape[1] == 27:
                pool_times = 2
            elif prevLayer.shape[1] == 13:
                pool_times = 1
            for i in range(pool_times):
                bottleneck = layers.MaxPool2D(pool_size=(3,3), strides=(2,2),name=tf.compat.v1.get_default_graph().unique_name("branch_MaxPool2D"))(bottleneck)
            # bottleneck = layers.MaxPool2D(pool_size=(3,3), strides=(2,2),name=tf.compat.v1.get_default_graph().unique_name("branch_MaxPool2D"))(bottleneck)
            # if pool_times >0:
                # bottleneck = keras.layers.Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same",name=tf.compat.v1.get_default_graph().unique_name("branch_conv"))(bottleneck)
        
            branchLayer = branch.FeatureDistillation(name=tf.compat.v1.get_default_graph().unique_name("branch_teaching"))(branchLayer,teaching_features)    
            branchLayer = layers.Flatten(name=tf.compat.v1.get_default_graph().unique_name("branch_flatten"))(branchLayer)
        else:
            print("no teaching feature Provided, bottleneck and teaching loss skipped")
            branchLayer = layers.Flatten(name=tf.compat.v1.get_default_graph().unique_name("branch_flatten"))(prevLayer)

        branchLayer = layers.Dense(124, activation="relu",name=tf.compat.v1.get_default_graph().unique_name("branch124"))(branchLayer)
        branchLayer = layers.Dense(64, activation="relu",name=tf.compat.v1.get_default_graph().unique_name("branch64"))(branchLayer)
        branchLayer = layers.Dense(10, name=tf.compat.v1.get_default_graph().unique_name("branch_output"))(branchLayer)
        output = branch.BranchEndpoint(name=tf.compat.v1.get_default_graph().unique_name("branch_softmax"))(branchLayer, targets, teacher_sm)
        # output = (layers.Softmax(name=tf.compat.v1.get_default_graph().unique_name("branch_softmax"))(branchLayer))

        return output

    def newBranch_compress_old(prevLayer):

        fire2_squeeze = layers.Convolution2D(
        16, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='branch_2_squeeze',
        data_format="channels_last")(prevLayer)
        fire2_expand1 = layers.Convolution2D(
        64, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='branch_2_expand1',
        data_format="channels_last")(fire2_squeeze)
        fire2_expand2 = layers.Convolution2D(
        64, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='branch_2_expand2',
        data_format="channels_last")(fire2_squeeze)
        merge2 = layers.Concatenate(axis=1)([fire2_expand1, fire2_expand2])

        fire3_squeeze = layers.Convolution2D(
        16, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='branch_3_squeeze',
        data_format="channels_last")(merge2)
        fire3_expand1 = layers.Convolution2D(
        64, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='branch_3_expand1',
        data_format="channels_last")(fire3_squeeze)
        fire3_expand2 = layers.Convolution2D(
        64, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='branch_3_expand2',
        data_format="channels_last")(fire3_squeeze)
        merge3 = layers.Concatenate(axis=1)([fire3_expand1, fire3_expand2])

        fire4_squeeze = layers.Convolution2D(
        32, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='branch_4_squeeze',
        data_format="channels_last")(merge3)
        fire4_expand1 = layers.Convolution2D(
        128, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='branch_4_expand1',
        data_format="channels_last")(fire4_squeeze)
        fire4_expand2 = layers.Convolution2D(
        128, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='branch_4_expand2',
        data_format="channels_last")(fire4_squeeze)
        merge4 = layers.Concatenate(axis=1)([fire4_expand1, fire4_expand2])
        maxpool4 = layers.MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), name='maxpool4',
        data_format="channels_last")(merge4)

        fire5_squeeze = layers.Convolution2D(
        32, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='branch_5_squeeze',
        data_format="channels_last")(maxpool4)
        fire5_expand1 = layers.Convolution2D(
        128, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='branch_5_expand1',
        data_format="channels_last")(fire5_squeeze)
        fire5_expand2 = layers.Convolution2D(
        128, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='branch_5_expand2',
        data_format="channels_last")(fire5_squeeze)
        merge5 = layers.Concatenate(axis=1)([fire5_expand1, fire5_expand2])

        fire6_squeeze = layers.Convolution2D(
        48, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='branch_6_squeeze',
        data_format="channels_last")(merge5)
        fire6_expand1 = layers.Convolution2D(
        192, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='branch_6_expand1',
        data_format="channels_last")(fire6_squeeze)
        fire6_expand2 = layers.Convolution2D(
        192, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='branch_6_expand2',
        data_format="channels_last")(fire6_squeeze)
        merge6 = layers.Concatenate(axis=1)([fire6_expand1, fire6_expand2])

        fire7_squeeze = layers.Convolution2D(
        48, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='branch_7_squeeze',
        data_format="channels_last")(merge6)
        fire7_expand1 = layers.Convolution2D(
        192, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='branch_7_expand1',
        data_format="channels_last")(fire7_squeeze)
        fire7_expand2 = layers.Convolution2D(
        192, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='branch_7_expand2',
        data_format="channels_last")(fire7_squeeze)
        merge7 = layers.Concatenate(axis=1)([fire7_expand1, fire7_expand2])

        fire8_squeeze = layers.Convolution2D(
        64, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='branch_8_squeeze',
        data_format="channels_last")(merge7)
        fire8_expand1 = layers.Convolution2D(
        256, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='branch_8_expand1',
        data_format="channels_last")(fire8_squeeze)
        fire8_expand2 = layers.Convolution2D(
        256, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='branch_8_expand2',
        data_format="channels_last")(fire8_squeeze)
        merge8 = layers.Concatenate(axis=1)([fire8_expand1, fire8_expand2])

        maxpool8 = layers.MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), name='maxpool8',
        data_format="channels_last")(merge8)
        fire9_squeeze = layers.Convolution2D(
        64, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='branch_9_squeeze',
        data_format="channels_last")(maxpool8)
        fire9_expand1 = layers.Convolution2D(
        256, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='branch_9_expand1',
        data_format="channels_last")(fire9_squeeze)
        fire9_expand2 = layers.Convolution2D(
        256, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='branch_9_expand2',
        data_format="channels_last")(fire9_squeeze)
        merge9 = layers.Concatenate(axis=1)([fire9_expand1, fire9_expand2])

        fire9_dropout = layers.Dropout(0.5, name='branch_9_dropout')(merge9)
        conv10 = layers.Convolution2D(
        10, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='valid', name='branch_conv10'
        )(fire9_dropout)

        global_avgpool10 = layers.GlobalAveragePooling2D(name='branch_global_avgpool10')(conv10)
        softmax = layers.Activation("softmax", name=tf.compat.v1.get_default_graph().unique_name("branch_softmax"))(global_avgpool10)
        # output = (layers.Softmax(name=tf.compat.v1.get_default_graph().unique_name("branch_softmax"))(branchLayer))

        return softmax

    def newBranch_compress(prevLayer):
        fire2_squeeze = layers.Convolution2D(
        16, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='branch_2_squeeze',
        data_format="channels_last")(prevLayer)
        fire2_expand1 = layers.Convolution2D(
        64, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='branch_2_expand1',
        data_format="channels_last")(fire2_squeeze)
        fire2_expand2 = layers.Convolution2D(
        64, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='branch_2_expand2',
        data_format="channels_last")(fire2_squeeze)
        merge2 = layers.Concatenate(axis=1,name="branch_merge2")([fire2_expand1, fire2_expand2])

        fire3_squeeze = layers.Convolution2D(
        16, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='branch_3_squeeze',
        data_format="channels_last")(merge2)
        fire3_expand1 = layers.Convolution2D(
        64, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='branch_3_expand1',
        data_format="channels_last")(fire3_squeeze)
        fire3_expand2 = layers.Convolution2D(
        64, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='branch_3_expand2',
        data_format="channels_last")(fire3_squeeze)
        merge3 = layers.Concatenate(axis=1,name="branch_merge3")([fire3_expand1, fire3_expand2])

        fire4_squeeze = layers.Convolution2D(
        32, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='branch_4_squeeze',
        data_format="channels_last")(merge3)
        fire4_expand1 = layers.Convolution2D(
        128, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='branch_4_expand1',
        data_format="channels_last")(fire4_squeeze)
        fire4_expand2 = layers.Convolution2D(
        128, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='branch_4_expand2',
        data_format="channels_last")(fire4_squeeze)
        merge4 = layers.Concatenate(axis=1)([fire4_expand1, fire4_expand2])
        maxpool4 = layers.MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), name='branch_maxpool4',
        data_format="channels_last")(merge4)

        fire5_squeeze = layers.Convolution2D(
        32, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='branch_5_squeeze',
        data_format="channels_last")(maxpool4)
        fire5_expand1 = layers.Convolution2D(
        128, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='branch_5_expand1',
        data_format="channels_last")(fire5_squeeze)
        fire5_expand2 = layers.Convolution2D(
        128, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='branch_5_expand2',
        data_format="channels_last")(fire5_squeeze)
        merge5 = layers.Concatenate(axis=1, name="branch_merge5")([fire5_expand1, fire5_expand2])

        fire6_squeeze = layers.Convolution2D(
        48, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='branch_6_squeeze',
        data_format="channels_last")(merge5)
        fire6_expand1 = layers.Convolution2D(
        192, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='branch_6_expand1',
        data_format="channels_last")(fire6_squeeze)
        fire6_expand2 = layers.Convolution2D(
        192, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='branch_6_expand2',
        data_format="channels_last")(fire6_squeeze)
        merge6 = layers.Concatenate(axis=1, name="branch_merge6")([fire6_expand1, fire6_expand2])

        fire7_squeeze = layers.Convolution2D(
        48, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='branch_7_squeeze',
        data_format="channels_last")(prevLayer)
        fire7_expand1 = layers.Convolution2D(
        192, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='branch_7_expand1',
        data_format="channels_last")(fire7_squeeze)
        fire7_expand2 = layers.Convolution2D(
        192, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='branch_7_expand2',
        data_format="channels_last")(fire7_squeeze)
        merge7 = layers.Concatenate(axis=1,name="branch_merge7")([fire7_expand1, fire7_expand2])

        fire8_squeeze = layers.Convolution2D(
        64, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='branch_8_squeeze',
        data_format="channels_last")(merge7)
        fire8_expand1 = layers.Convolution2D(
        256, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='branch_8_expand1',
        data_format="channels_last")(fire8_squeeze)
        fire8_expand2 = layers.Convolution2D(
        256, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='branch_8_expand2',
        data_format="channels_last")(fire8_squeeze)
        merge8 = layers.Concatenate(axis=1, name="branch_merge8")([fire8_expand1, fire8_expand2])

        maxpool8 = layers.MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), name='branch_maxpool8',
        data_format="channels_last")(merge8)
        fire9_squeeze = layers.Convolution2D(
        64, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='branch_9_squeeze',
        data_format="channels_last")(maxpool8)
        fire9_expand1 = layers.Convolution2D(
        256, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='branch_9_expand1',
        data_format="channels_last")(fire9_squeeze)
        fire9_expand2 = layers.Convolution2D(
        256, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='branch_9_expand2',
        data_format="channels_last")(fire9_squeeze)
        merge9 = layers.Concatenate(axis=1,name="branch_merge9")([fire9_expand1, fire9_expand2])

        fire9_dropout = layers.Dropout(0.5, name='branch_9_dropout')(merge9)
        conv10 = layers.Convolution2D(
        10, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='valid', name='branch_conv10'
        )(fire9_dropout)

        global_avgpool10 = layers.GlobalAveragePooling2D(name='branch_global_avgpool10')(conv10)
        softmax = layers.Activation("softmax", name=tf.compat.v1.get_default_graph().unique_name("branch_softmax"))(global_avgpool10)
        # output = (layers.Softmax(name=tf.compat.v1.get_default_graph().unique_name("branch_softmax"))(branchLayer))

        return softmax
