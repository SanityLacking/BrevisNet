# import the necessary packages
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
import itertools
import time
import json

# from keras.models import load_model
# from keras.utils import CustomObjectScope
# from keras.initializers import glorot_uniform

import math
import pydot
import os
#os.environ["PATH"] += os.pathsep + "C:\Program Files\Graphviz\bin"
#from tensorflow.keras.utils import plot_model
from utils import *
from branchyEval import branchyEval as eval
from Alexnet_kaggle_v2 import * 

# ALEXNET = False
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)
root_logdir = os.path.join(os.curdir, "logs\\fit\\")



class BranchyNet:
    def loadTrainingData(self):
        """ load minist dataset and configure it. 
        """
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        x_train = x_train.reshape(60000, 784).astype("float32") / 255
        x_test = x_test.reshape(10000, 784).astype("float32") / 255

        return (x_train, y_train), (x_test, y_test)


    ALEXNET = False
    KNOWN_MODELS = [
        {"name":"mnist","dataset": loadTrainingData},
        {"name":"alexnet","dataset":tf.keras.datasets.cifar10.load_data},
    ]
    def prepareMnistDataset(self,dataset,batch_size=32):
        import csv
        (train_images, train_labels), (test_images, test_labels) = dataset
        train_images = train_images.reshape(60000, 784).astype("float32") / 255
        test_images = test_images.reshape(10000, 784).astype("float32") / 255

        validation_images, validation_labels = train_images[:12000], train_labels[:12000]
        train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
        validation_ds = tf.data.Dataset.from_tensor_slices((validation_images, validation_labels))

        train_ds_size = len(list(train_ds))
        test_ds_size = len(list(test_ds))
        validation_ds_size = len(list(validation_ds))
        train_ds = (train_ds
            # .map(augment_images)
            .shuffle(buffer_size=int(train_ds_size),reshuffle_each_iteration=True)
            .batch(batch_size=batch_size, drop_remainder=True))
        test_ds = (test_ds
            # .map(augment_images)
            .shuffle(buffer_size=int(test_ds_size)) ##why would you shuffle the test set?
            .batch(batch_size=batch_size, drop_remainder=True))

        validation_ds = (validation_ds
            # .map(augment_images)
            .shuffle(buffer_size=int(validation_ds_size))
            .batch(batch_size=batch_size, drop_remainder=True))
        return train_ds, test_ds, validation_ds

    def prepareAlexNetDataset_alt(self,dataset,batch_size=32):
        import csv
        (train_images, train_labels), (test_images, test_labels) = dataset
        with open('results/altTrain_labels2.csv', newline='') as f:
            reader = csv.reader(f,quoting=csv.QUOTE_NONNUMERIC)
            alt_trainLabels = list(reader)
        with open('results/altTest_labels2.csv', newline='') as f:
            reader = csv.reader(f,quoting=csv.QUOTE_NONNUMERIC)
            alt_testLabels = list(reader)

        altTraining = tf.data.Dataset.from_tensor_slices((train_images,alt_trainLabels))

        validation_images, validation_labels = train_images[:5000], alt_trainLabels[:5000]
        train_ds = tf.data.Dataset.from_tensor_slices((train_images, alt_trainLabels))
        test_ds = tf.data.Dataset.from_tensor_slices((test_images, alt_testLabels))
        validation_ds = tf.data.Dataset.from_tensor_slices((validation_images, validation_labels))

        train_ds_size = len(list(train_ds))
        test_ds_size = len(list(test_ds))
        validation_ds_size = len(list(validation_ds))
        train_ds = (train_ds
            .map(augment_images)
            .shuffle(buffer_size=int(train_ds_size),reshuffle_each_iteration=True)
            .batch(batch_size=batch_size, drop_remainder=True))
        test_ds = (test_ds
            .map(augment_images)
            .shuffle(buffer_size=int(test_ds_size)) ##why would you shuffle the test set?
            .batch(batch_size=batch_size, drop_remainder=True))

        validation_ds = (validation_ds
            .map(augment_images)
            .shuffle(buffer_size=int(validation_ds_size))
            .batch(batch_size=batch_size, drop_remainder=True))
        return train_ds, test_ds, validation_ds


    def prepareAlexNetDataset(self, dataset, batch_size =32):
        (train_images, train_labels), (test_images, test_labels) = dataset

        validation_images, validation_labels = train_images[:5000], train_labels[:5000]
        train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
        validation_ds = tf.data.Dataset.from_tensor_slices((validation_images, validation_labels))

        train_ds_size = len(list(train_ds))
        test_ds_size = len(list(test_ds))
        validation_ds_size = len(list(validation_ds))


        print("trainSize {}".format(train_ds_size))
        print("testSize {}".format(test_ds_size))

        train_ds = (train_ds
            .map(augment_images)
            .shuffle(buffer_size=int(train_ds_size))
            # .shuffle(buffer_size=int(train_ds_size),reshuffle_each_iteration=True)
            .batch(batch_size=batch_size, drop_remainder=True))
        test_ds = (test_ds
            .map(augment_images)
            # .shuffle(buffer_size=int(train_ds_size)) ##why would you shuffle the test set?
            .batch(batch_size=batch_size, drop_remainder=True))

        validation_ds = (validation_ds
            .map(augment_images)
            # .shuffle(buffer_size=int(train_ds_size))
            .batch(batch_size=batch_size, drop_remainder=True))
        return train_ds, test_ds, validation_ds
    
    def prepareAlexNetDataset_old(self, batchsize=32):
        # tf.debugging.experimental.enable_dump_debug_info(logdir, tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1)
        (train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()

        CLASS_NAMES= ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        # validation_images, validation_labels = train_images[:5000], alt_trainLabels[:5000]
        # train_ds = tf.data.Dataset.from_tensor_slices((train_images, alt_trainLabels))
        # test_ds = tf.data.Dataset.from_tensor_slices((test_images, alt_testLabels))

        ###normal method
        validation_images, validation_labels = train_images[:5000], train_labels[:5000]
        train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
        validation_ds = tf.data.Dataset.from_tensor_slices((validation_images, validation_labels))

        def augment_images(image, label):
            # Normalize images to have a mean of 0 and standard deviation of 1
            # image = tf.image.per_image_standardization(image)
            # Resize images from 32x32 to 277x277
            image = tf.image.resize(image, (227,227))
            return image, label
        def augment_images2(image):
            # Normalize images to have a mean of 0 and standard deviation of 1
            # image = tf.image.per_image_standardization(image)
            # Resize images from 32x32 to 277x277
            image = tf.image.resize(image, (227,227))
            return image

        train_ds_size = len(list(train_ds))
        test_ds_size = len(list(test_ds))
        validation_ds_size = len(list(validation_ds))

        print("trainSize {}".format(train_ds_size))
        print("testSize {}".format(test_ds_size))

        train_ds = (train_ds
                        .map(augment_images)
                        .shuffle(buffer_size=train_ds_size)
                        .batch(batch_size=batchsize, drop_remainder=True))

        test_ds = (test_ds
                        .map(augment_images)
                        #   .shuffle(buffer_size=train_ds_size)
                        .batch(batch_size=batchsize, drop_remainder=True))

        validation_ds = (validation_ds
                        .map(augment_images)
                        #   .shuffle(buffer_size=validation_ds_size)
                        .batch(batch_size=batchsize, drop_remainder=True))

        print("testSize2 {}".format(len(list(test_ds))))
        return train_ds, test_ds, validation_ds

    def addBranches(self,model, identifier =[""], customBranch = [],exact = True):
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




    def trainModel(self, model, dataset, epocs = 2,save = False):
        """ Train the model that is passed through. This function works for both single and multiple branch models.
        """
        logs = []
        (x_train, y_train), (x_test, y_test) = dataset
        print("ALEXNET {}".format(self.ALEXNET))
        if self.ALEXNET:
           train_ds, test_ds, validation_ds = self.prepareAlexNetDataset(dataset, batch_size=32)
        else: 
            # can still use tf.data.Dataset for mnist and numpy models
            # I found a bug where the model couldn't run on the input unless the dataset is batched. so make sure to batch it.
            val_size = int(len(train_images) * 0.2)  #atm I'm making validation sets that are a fifth of the test set. 
            x_val = train_images[-val_size:]
            y_val = train_labels[-val_size:]
            train_images = train_images[:-val_size]
            train_labels = train_labels[:-val_size]
            
            train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
            train_ds = train_ds.shuffle(buffer_size=1024).batch(64)
            # Reserve 10,000 samples for validation
           
            test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
            test_ds = test_ds.batch(64)

            # Prepare the validation dataset
            validation_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
            validation_ds = validation_ds.batch(64)

        num_outputs = len(model.outputs) # the number of output layers for the purpose of providing labels

        if self.ALEXNET: 
            model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(lr=0.001, momentum=0.9), metrics=['accuracy'])
        else:
            model.compile(loss="sparse_categorical_crossentropy", optimizer=keras.optimizers.Adam(),metrics=["accuracy"])

        run_logdir = get_run_logdir(model.name)
        tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
        print("after reset:")
        test_scores = model.evaluate(test_ds, verbose=2)
        print("finish eval")
        printTestScores(test_scores,num_outputs)
        checkpoint = keras.callbacks.ModelCheckpoint("models/{}_new.hdf5".format(model.name), monitor='val_loss', verbose=1, mode='max')

        for j in range(epocs):
            print("epoc: {}".format(j))
            results = [j]           
            history = model.fit(train_ds, epochs=epocs, validation_data=validation_ds, callbacks=[tensorboard_cb,checkpoint])
            print(history)
            test_scores = model.evaluate(test_ds, verbose=2)
            print("overall loss: {}".format(test_scores[0]))
            if num_outputs > 1:
                for i in range(num_outputs):
                    print("Output {}: Test loss: {}, Test accuracy {}".format(i, test_scores[i+1], test_scores[i+1+num_outputs]))
                    results.append("Output {}: Test loss: {}, Test accuracy {}".format(i, test_scores[i+1], test_scores[i+1+num_outputs]))
            else:
                print("Test loss:", test_scores[0])
                print("Test accuracy:", test_scores[1])
            logs.append(results)
        if save:
            saveModel(model,"model_transfer_trained")

        return model
  

    def trainModelTransfer(self, model, dataset, resetBranches = False, epocs = 2,save = False,transfer = True, saveName =""):
        """Train the model that is passed using transfer learning. This function expects a model with trained main branches and untrained (or randomized) side branches.
        """
        logs = []
        num_outputs = len(model.outputs) # the number of output layers for the purpose of providing labels
        print("ALEXNET {}".format(self.ALEXNET))
        if self.ALEXNET:
           train_ds, test_ds, validation_ds = self.prepareAlexNetDataset_old()
        else: 
            train_ds, test_ds, validation_ds = self.prepareMnistDataset(dataset, batch_size=32)

       
        #Freeze main branch layers
        #how to iterate through layers and find main branch ones?
        #simple fix for now: all branch nodes get branch in name.
        if transfer: 
            for i in range(len(model.layers)):
                print(model.layers[i].name)
                if "branch" in model.layers[i].name:
                    print("setting branch layer training to true")
                    model.layers[i].trainable = True
                else: 
                    print("setting main layer training to false")
                    model.layers[i].trainable = False               
        else:
            for i in range(len(model.layers)):
                print(model.layers[i].name)
                model.layers[i].trainable = True
                print("setting layer training to True")

        # model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=keras.optimizers.Adam(),metrics=["accuracy"])
        if self.ALEXNET: 
            model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(lr=0.001), metrics=['accuracy'])
        else:
            model.compile(loss="sparse_categorical_crossentropy", optimizer=keras.optimizers.Adam(),metrics=["accuracy"])

        run_logdir = get_run_logdir(model.name)
        tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
        print("after reset:")

        test_scores = model.evaluate(test_ds, verbose=2)
        print("finish eval")
        printTestScores(test_scores,num_outputs)
        if saveName =="":
            newModelName = "{}_branched.hdf5".format(model.name )
        else:
            newModelName = saveName
        checkpoint = keras.callbacks.ModelCheckpoint("models/{}.hdf5".format(newModelName), monitor='val_acc', verbose=1, mode='max')

        for j in range(epocs):
            print("epoc: {}".format(j))
            results = [j]           
            history =model.fit(train_ds,
                    epochs=50,
                    validation_data=validation_ds,
                    validation_freq=1,
                    callbacks=[tensorboard_cb,checkpoint])

                                
                                # batch_size=32,
                                # validation_data=validation_ds,
                                # validation_steps=10,
                                # callbacks=[tensorboard_cb,checkpoint])
            print(history)
            test_scores = model.evaluate(test_ds, verbose=2)
            print("overall loss: {}".format(test_scores[0]))
            if num_outputs > 1:
                for i in range(num_outputs):
                    print("Output {}: Test loss: {}, Test accuracy {}".format(i, test_scores[i+1], test_scores[i+1+num_outputs]))
                    results.append("Output {}: Test loss: {}, Test accuracy {}".format(i, test_scores[i+1], test_scores[i+1+num_outputs]))
            else:
                print("Test loss:", test_scores[0])
                print("Test accuracy:", test_scores[1])
            logs.append(results)
        return model

    def datasetStats(self, dataset):
        (train_images, train_labels), (test_images, test_labels) = dataset
        print("ALEXNET {}".format(self.ALEXNET))
        if self.ALEXNET:
            train_ds, test_ds, validation_ds = self.prepareAlexNetDataset(dataset, batch_size = 1)
        else: 
            # can still use tf.data.Dataset for mnist and numpy models
            # I found a bug where the model couldn't run on the input unless the dataset is batched. so make sure to batch it.
            val_size = int(len(train_images) * 0.2)  #atm I'm making validation sets that are a fifth of the test set. 
            x_val = train_images[-val_size:]
            y_val = train_labels[-val_size:]
            train_images = train_images[:-val_size]
            train_labels = train_labels[:-val_size]
            
            train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
            train_ds = train_ds.shuffle(buffer_size=1024).batch(64)
            # Reserve 10,000 samples for validation
           
            test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
            test_ds = test_ds.batch(64)

            # Prepare the validation dataset
            validation_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
            validation_ds = validation_ds.batch(64)
        ds_iter = iter(test_ds)
        results = []
        for i in range(len(list(validation_ds))):
            one = ds_iter.get_next()
            results.append(one[1].numpy())
        # print(one)
            # print(one[1])
        results = np.vstack(results)
        unique, counts = np.unique(results, return_counts=True)        
        print(dict(zip(unique, counts)))
        return

    def checkModelName(self, modelName):
        """ check to see if the model's name exists in the known models list
            if so, return the object of the model's details
            else, return None
        """
        print("modelName:{}".format(modelName))
        result = None
        try:
            for model in self.KNOWN_MODELS:
                print(model)
                if model["name"] == modelName:
                    result = model
        except Exception as e:
            print(e)
            raise
            # print("could not find model name: {} in known models".format(modelName))
            
        return result



    ###### RUN MODEL SHORTCUTS ######

        
    def Run_alexNet(self, numEpocs = 2, modelName="", saveName ="",transfer = True):
        x = tf.keras.models.load_model("models/{}".format(modelName))

        x.summary()
        if saveName =="":
            saveName = modelName

        # funcModel = models.Model([input_layer], [prev_layer])
        # funcModel = self.addBranches(x,["dense","conv2d","max_pooling2d","batch_normalization","dense","dropout"],newBranch)
        funcModel = self.addBranches(x,["max_pooling2d","max_pooling2d_1","dense"],newBranch_flatten,exact=True)
        # funcModel = self.addBranches(x,["dense","dense_1"],newBranch_oneLayer,exact=True)
        funcModel.summary()
        funcModel.save("models/{}".format(saveName))

        funcModel = self.trainModelTransfer(funcModel,tf.keras.datasets.cifar10.load_data(), epocs = numEpocs, save = False, transfer = transfer, saveName = saveName)
        # funcModel.save("models/{}".format(saveName))
        # x = keras.Model(inputs=x.inputs, outputs=x.outputs, name="{}_normal".format(x.name))
        return x

    def Run_mnistNet(self, numEpocs = 5, modelName="", saveName ="",transfer = True):
        x = tf.keras.models.load_model("models/{}".format(modelName))
        x.summary()
        if saveName =="":
            saveName = modelName
        # funcModel = models.Model([input_layer], [prev_layer])
        # funcModel = self.addBranches(x,["dense","conv2d","max_pooling2d","batch_normalization","dense","dropout"],newBranch)
        funcModel = self.addBranches(x,["dense","dense_2","dense_3"],newBranch,exact=True)
        funcModel.summary()
        if saveName == "":
            funcModel.save("models/{}_branched.hdf5".format(modelName))
        else: 
            funcModel.save("models/{}".format(saveName))

        funcModel = self.trainModelTransfer(funcModel,keras.datasets.mnist.load_data(),epocs = numEpocs, transfer = transfer, saveName = saveName)
        if saveName == "":
            funcModel.save("models/{}_branched.hdf5".format(modelName))
        else: 
            funcModel.save("models/{}".format(saveName))

        # x = keras.Model(inputs=x.inputs, outputs=x.outputs, name="{}_normal".format(x.name))
        return x
    def Run_train_model(self, model_name, dataset=None, numEpocs =2):
        """ generic training function. takes a model or model name and trains the model on the dataset for the specificed epocs.
        """
        x=None
        modelDetails = None
        print(model_name)
        if type(model_name) == type(""):
            #if the model_name is a valid filepath:
            if os.path.isfile(model_name):
                try:
                    x = tf.keras.models.load_model(model_name)
                except Exception as e:
                    print(e)
                    print("model {} could not be loaded".format(model_name))
            else:
                modelDetails = self.checkModelName(model_name) 
                print(modelDetails)
                #load newest version of model of known type.
                try:
                    x = tf.keras.models.load_model(newestModelPath(modelDetails["name"]))
                except Exception as e:
                    print(e)
                    print("could not load the newest model of known type: {}".format(modelDetails["name"]))
                    raise


        # elif type(model_name) == type(Model):
        x = model_name                        

        x.summary()

        #load dataset
        if dataset ==None:
            #check name of model, if recognized against known models, use the default dataset.
            modelDetails = self.checkModelName(model_name)
            if modelDetails:
                try:
                    dataset = modelDetails["dataset"](self)
                except expression as identifier:
                    print("dataset was not able to be identified")
            else: 
                print("model doesn't match any known models, dataset could not be loaded.")
                raise
            #else 


        #run training
        x = self.branchy.trainModelTransfer(x,dataset,epocs = numEpocs, save = False)


        return x
    
    def eval_branches(self, model, dataset, count = 1, options="accuracy"):
        """ evaulate func for checking how well a branched model is performing.
            function may be moved to eval_model.py in the future.
        """ 
        num_outputs = len(model.outputs) # the number of output layers for the purpose of providing labels

        (train_images, train_labels), (test_images, test_labels) = dataset
        
        print("ALEXNET {}".format(self.ALEXNET))
        if self.ALEXNET:
            train_ds, test_ds, validation_ds = self.prepareAlexNetDataset(dataset,32)                       
        else: 
            test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
            test_ds_size = len(list(test_ds))
            test_ds =  (test_ds
                .batch(batch_size=64, drop_remainder=True))
        
        if self.ALEXNET: 
            model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(lr=0.001), metrics=['accuracy'])
        else:
            model.compile(loss="sparse_categorical_crossentropy", optimizer=keras.optimizers.Adam(),metrics=["accuracy"])

        run_logdir = get_run_logdir(model.name)
        tensorboard_cb = keras.callbacks.TensorBoard(run_logdir +"/eval")

        if options == "accuracy":
            test_scores = model.evaluate(test_ds, verbose=2)
            printTestScores(test_scores,num_outputs)
        elif options == "entropy":
            if self.ALEXNET:
                train_ds, test_ds, validation_ds = self.prepareAlexNetDataset(dataset,1)                       
            else: 
                test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
                test_ds_size = len(list(test_ds))
                test_ds =  (test_ds
                    # .map(augment_images)
                    # .shuffle(buffer_size=int(test_ds_size))
                    .batch(batch_size=1, drop_remainder=True))
        
            iterator = iter(test_ds)
            item = iterator.get_next()
            results = model.predict(item[0])

            for output in results:
                for result in output: 
                    print(result)
                    Pclass = np.argmax(result)
                    print("predicted class:{}, actual class: {}".format(Pclass, item[1]))

                    for softmax in result:
                        entropy = calcEntropy(softmax)
                        print("entropy:{}".format(entropy))
            print("answer: {}".format(item[1]))
            # results = calcEntropy(y_hat)
            # print(results)
            pass
        elif options == "throughput":
            if self.ALEXNET:
                train_ds, test_ds, validation_ds = self.prepareAlexNetDataset(dataset,1)                       
            else: 
                test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
                test_ds_size = len(list(test_ds))
                test_ds =  (test_ds
                    # .map(augment_images)
                    # .shuffle(buffer_size=int(test_ds_size))
                    .batch(batch_size=1, drop_remainder=True))
        
            iterator = iter(test_ds)
            item = iterator.get_next()
            pred=[]
            labels=[]
            for i in range(len(test_ds)):
                pred.append(model.predict(item[0]))
                labels.append(item[1])
            
            results = throughputMatrix(pred, labels, numOutput)
            print(results)
            print(pd.DataFrame(results).T)
            pass


        return 

    def predict(self, model, dataset, thresholds=[]):
        """ run the model using the provided confidence thresholds            
        """ 
        num_outputs = len(model.outputs) # the number of output layers for the purpose of providing labels

        train_ds, test_ds, validation_ds = self.prepareAlexNetDataset(dataset,32)                       
        model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(lr=0.001), metrics=['accuracy'])

        run_logdir = get_run_logdir(model.name)
        tensorboard_cb = keras.callbacks.TensorBoard(run_logdir +"/eval")
        predictions = []
        labels = []
        labelClasses = [0,1,2,3,4,5,6,7,8,9]



        """ For the first version testing purpose of this function, the whole model is run for each 
            input item. in future versions the model will be run sequentially from branch to branch, exiting run when 
            an accepted confidence score per item is achieved.

        """

        iterator = iter(test_ds)
        for j in range(len(test_ds)):
        # for j in range(10):
            # print("prediction: {} of {}".format(j,len(test_ds)),end='\r')
            item = iterator.get_next()
            prediction = model.predict(item[0])
            predictions.append(prediction)
            # print(prediction)
            labels.append(item[1].numpy().tolist()) #put a copy of the label (second element in the item tuple) in the labels list.
        labels = [expandlabels(x,num_outputs)for x in labels]
        predEntropy =[]
        predClasses =[]
        print("predictions complete, analyizing")


        for i,output in enumerate(predictions):
            for k, pred in enumerate(output):
                pred_classes=[]
                pred_entropy = []
                print("image: {} of {}".format(i,len(predictions)),end='\r')
                for l, branch in enumerate(pred):
                    Pclass = np.argmax(branch[0])
                    pred_classes.append(Pclass) 
                    pred_entropy.append(calcEntropy(branch[0]))                       
                predClasses.append(pred_classes)
                predEntropy.append(pred_entropy)
        
        results = np.equal(predictions, labels)
        labels = np.array(labels)
        transpose_results = np.transpose(results) #truths
        transpose_labels = np.transpose(labels)

        mAcc = exitAccuracy(transpose_results[0],transpose_labels[0], labelClasses)
        # results = eval.KneeGraphClasses(predClasses, labels,predEntropy, num_outputs,labelClasses,output_names)
        return

    def old_entropyMatrix(self, model, dataset):
        """
            calculate the entropy of the branches for the test set.
        """
        num_outputs = len(model.outputs) # the number of output layers for the purpose of providing labels
        (train_images, train_labels), (test_images, test_labels) = dataset
        
        
        print("ALEXNET {}".format(self.ALEXNET))
        if self.ALEXNET:
            train_ds, test_ds, validation_ds = self.prepareAlexNetDataset(dataset,1)
        else: 
            test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
            test_ds_size = len(list(test_ds))
            test_ds =  (test_ds
                .batch(batch_size=1, drop_remainder=True))
        
        predictions = []
        labels = []
        if self.ALEXNET: 
            model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(lr=0.001), metrics=['accuracy'])
            iterator = iter(test_ds)
            indices = []
            for j in range(5):
                item = iterator.get_next()
                # for j in [33,2,3,4]:
                # img = test_images[j].reshape(1,784)
                prediction = model.predict(item[0])
                # print(prediction)
                predictions.append(prediction)
                labels.append(item[1].numpy().tolist())
                # print(item[1])
            predClasses =[]
            predEntropy =[]
            labelClasses = []
            for i,output in enumerate(predictions):
                print(output)
                for k, pred in enumerate(output):
                    pred_classes=[]
                    pred_entropy = []
                    label_classes = []
                    # print(i,end='\r')
                    for l, branch in enumerate(pred):
                        print(l)
                        Pclass = np.argmax(branch[0])
                        pred_classes.append(Pclass)     
                        pred_entropy.append(calcEntropy(branch[0]))                   
                        print("{}, class {}".format(branch[0], Pclass))
                    predClasses.append(pred_classes)
                    predEntropy.append(pred_entropy)
                    # Pprob = exit[0][Pclass]
                    # print("image:{}, exit:{}, pred Class:{}, probability: {} actual Class:{}".format(j,i, Pclass,Pprob, item[1]))
                    # if Pclass != item[1]:
                        # indices.append(j)
                    # entropy = calcEntropy(elem)
                    # print("entropy:{}".format(entropy))                
            # labels = [item for sublist in labels for item in sublist]
           
            labels = list(map(expandlabels,labels))

            print(predClasses)
            print(labels)
            print(predEntropy)
            # matrix = []cmd
            # for i, prediction in enumerate(predClasses):
            #     for j, branch in enumerate(prediction):
            #         if branch == labels[i]:

            #             matrix.append()


        return 
    
    def GetResultsCSV(self,model,dataset,suffix=""):
        num_outputs = len(model.outputs) # the number of output layers for the purpose of providing labels
        if self.ALEXNET:
            train_ds, test_ds, validation_ds = self.prepareAlexNetDataset_old(1)
        else:
            train_ds, test_ds, validation_ds = self.prepareMnistDataset(dataset,1)
        
        predictions = []
        labels = []
        model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(lr=0.001), metrics=['accuracy'])

        # test_scores = model.evaluate(test_ds, verbose=2)
        # print(test_scores)

        iterator = iter(test_ds)
        print(len(test_ds))
        for j in range(len(test_ds)):
        # for j in range(12):

            print("prediction: {} of {}".format(j,len(test_ds)),end='\r')

            item = iterator.get_next()
            prediction = model.predict(item[0])
            # print("predictions {}".format(prediction))
            predictions.append(prediction)
            # print(prediction)
            labels.append(item[1].numpy().tolist())
        # print("labels")
        # print(labels)
        if self.ALEXNET:
            labels = [expandlabels(x,num_outputs)for x in labels]
        else:
            for i, val in enumerate(labels):
                print(i)
                labels[i]= [val]* num_outputs

        predEntropy =[]
        predClasses =[]
        predRaw=[]
        print("predictions complete, analyizing") 
        for i,output in enumerate(predictions):
            for k, pred in enumerate(output):
                pred_classes=[]
                pred_entropy = []
                pred_Raw=[]
                print("image: {} of {}".format(i,len(predictions)),end='\r')
                for l, branch in enumerate(pred):
                    pred_Raw.append(branch[0])
                    Pclass = np.argmax(branch[0])
                    pred_classes.append(Pclass) 
                    # if labels[i][0] == 0:
                        # print("class {}".format(Pclass))
                        # print("label {}".format(labels[i]))
                    # print(branch)
                    pred_entropy.append(calcEntropy(branch[0]))  
                    # print("entropy {}".format(pred_entropy))                     
                predRaw.append(pred_Raw)
                predClasses.append(pred_classes)
                predEntropy.append(pred_entropy)
                
        # print(predClasses)
        # print(predEntropy)
        # print(labels)
        # labels = list(map(expandlabels,labels,num_outputs))
        labelClasses = [0,1,2,3,4,5,6,7,8,9]
        predClasses = pd.DataFrame(predClasses)
        labels = pd.DataFrame(labels)
        predEntropy = pd.DataFrame(predEntropy)
        
        PredRaw = pd.DataFrame(predRaw)
        PredRaw.to_csv("results/predRaw_temp.csv", sep=',', mode='w',index=False)

        predClasses.to_csv("results/predClasses{}.csv".format(suffix), sep=',', mode='w',index=False)
        labels.to_csv("results/labels{}.csv".format(suffix), sep=',', mode='w',index=False)
        predEntropy.to_csv("results/predEntropy{}.csv".format(suffix), sep=',', mode='w',index=False)


        # results = KneeGraph(predClasses, labels,predEntropy, num_outputs,labelClasses,output_names)
        # results.to_csv("logs_entropy/{}_{}_entropyStats.csv".format(model.name,time.strftime("%Y%m%d_%H%M%S")), sep=',', mode='a')
        return

    def BranchKneeGraph(self,model,dataset):
        num_outputs = len(model.outputs) # the number of output layers for the purpose of providing labels
        
        output_names = [i.name for i in model.outputs]
        train_ds, test_ds, validation_ds = self.prepareAlexNetDataset(dataset,1)
        
        predictions = []
        labels = []
        model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(lr=0.001), metrics=['accuracy'])
        iterator = iter(test_ds)
        indices = []
        for j in range(len(test_ds)):
        # for j in range(10):

            print("prediction: {} of {}".format(j,len(test_ds)),end='\r')

            item = iterator.get_next()
            prediction = model.predict(item[0])
            predictions.append(prediction)
            # print(prediction)
            labels.append(item[1].numpy().tolist())
        labels = [expandlabels(x,num_outputs)for x in labels]
        predEntropy =[]
        predClasses =[]
        print("predictions complete, analyizing")
        for i,output in enumerate(predictions):
            # print(output)
            for k, pred in enumerate(output):
                pred_classes=[]
                pred_entropy = []
                print("image: {} of {}".format(i,len(predictions)),end='\r')
                for l, branch in enumerate(pred):
                    Pclass = np.argmax(branch[0])
                    pred_classes.append(Pclass) 
                    pred_entropy.append(calcEntropy(branch[0]))                       
                predClasses.append(pred_classes)
                predEntropy.append(pred_entropy)
                
        # print(predClasses)
        # print(predEntropy)
        # print(labels)
        # labels = list(map(expandlabels,labels,num_outputs))
        labelClasses = [0,1,2,3,4,5,6,7,8,9]
        results = eval.KneeGraph(predClasses, labels,predEntropy, num_outputs,labelClasses,output_names)
        # f = open("logs_entropy/{}_{}_entropyStats.txt".format(model.name,time.strftime("%Y%m%d_%H%M%S")), "w")
        # f.write(json.dumps(results))
        results.to_csv("logs_entropy/{}_{}_entropyStats.csv".format(model.name,time.strftime("%Y%m%d_%H%M%S")), sep=',', mode='a')
        # f.close()
        # print(results)
        # print(pd.DataFrame(results).T)
        return

    def BranchKneeGraphClasses(self,model,dataset):
        num_outputs = len(model.outputs) # the number of output layers for the purpose of providing labels
        
        output_names = [i.name for i in model.outputs]
        train_ds, test_ds, validation_ds = self.prepareAlexNetDataset(dataset,1)
        
        predictions = []
        labels = []
        model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(lr=0.001), metrics=['accuracy'])
        iterator = iter(test_ds)
        indices = []
        for j in range(len(test_ds)):
        # for j in range(10):
            print("prediction: {} of {}".format(j,len(test_ds)),end='\r')
            item = iterator.get_next()
            prediction = model.predict(item[0])
            predictions.append(prediction)
            # print(prediction)
            labels.append(item[1].numpy().tolist())
        labels = [expandlabels(x,num_outputs)for x in labels]
        predEntropy =[]
        predClasses =[]
        print("predictions complete, analyizing")
        for i,output in enumerate(predictions):
            # print(output)
            for k, pred in enumerate(output):
                pred_classes=[]
                pred_entropy = []
                print("image: {} of {}".format(i,len(predictions)),end='\r')
                for l, branch in enumerate(pred):
                    Pclass = np.argmax(branch[0])
                    pred_classes.append(Pclass) 
                    pred_entropy.append(calcEntropy(branch[0]))                       
                predClasses.append(pred_classes)
                predEntropy.append(pred_entropy)
                
        # print(predClasses)
        # print(predEntropy)
        # print(labels)
        # labels = list(map(expandlabels,labels,num_outputs))
        labelClasses = [0,1,2,3,4,5,6,7,8,9]
        results = eval.KneeGraphClasses(predClasses, labels,predEntropy, num_outputs,labelClasses,output_names)
        # f = open("logs_entropy/{}_{}_entropyStats.txt".format(model.name,time.strftime("%Y%m%d_%H%M%S")), "w")
        # f.write(json.dumps(results))
        results.to_csv("logs_entropy/{}_{}_PredictedClassesStats.csv".format(model.name,time.strftime("%Y%m%d_%H%M%S")), sep=',', mode='a')
        # f.close()
        # print(results)
        # print(pd.DataFrame(results).T)
        return

    def BranchKneeGraphPredictedClasses(self,model,dataset):
        num_outputs = len(model.outputs) # the number of output layers for the purpose of providing labels
        
        output_names = [i.name for i in model.outputs]
        (train_images, train_labels), (test_images, test_labels) = dataset
        train_ds, test_ds, validation_ds = self.prepareAlexNetDataset(dataset,1)
        
        predictions = []
        labels = []
        model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(lr=0.001), metrics=['accuracy'])
        iterator = iter(test_ds)
        indices = []
        # for j in range(len(test_ds)):
        for j in range(100):
            print("prediction: {} of {}".format(j,len(test_ds)),end='\r')
            item = iterator.get_next()
            prediction = model.predict(item[0])
            predictions.append(prediction)
            # print(prediction)
            labels.append(item[1].numpy().tolist())
        labels = [expandlabels(x,num_outputs)for x in labels]
        predEntropy =[]
        predClasses =[]
        print("predictions complete, analyizing")
        for i,output in enumerate(predictions):
            # print(output)
            for k, pred in enumerate(output):
                pred_classes=[]
                pred_entropy = []
                print("image: {} of {}".format(i,len(predictions)),end='\r')
                for l, branch in enumerate(pred):
                    Pclass = np.argmax(branch[0])
                    pred_classes.append(Pclass) 
                    pred_entropy.append(calcEntropy(branch[0]))                       
                predClasses.append(pred_classes)
                predEntropy.append(pred_entropy)
                
        # print(predClasses)
        # print(predEntropy)
        # print(labels)
        # labels = list(map(expandlabels,labels,num_outputs))
        labelClasses = [0,1,2,3,4,5,6,7,8,9]
        results = eval.KneeGraphPredictedClasses(predClasses, labels,predEntropy, num_outputs,labelClasses,output_names)
        # f = open("logs_entropy/{}_{}_entropyStats.txt".format(model.name,time.strftime("%Y%m%d_%H%M%S")), "w")
        # f.write(json.dumps(results))
        # results.to_csv("logs_entropy/{}_{}_entropyClassesStats.csv".format(model.name,time.strftime("%Y%m%d_%H%M%S")), sep=',', mode='a')
        # f.close()
        # print(results)
        # print(pd.DataFrame(results).T)
        return


    def BranchEntropyConfusionMatrix(self, model, dataset):
        num_outputs = len(model.outputs) # the number of output layers for the purpose of providing labels
        
        (train_images, train_labels), (test_images, test_labels) = dataset
        train_ds, test_ds, validation_ds = self.prepareAlexNetDataset(dataset,1)
        
        predictions = []
        labels = []
        model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(lr=0.001), metrics=['accuracy'])
        iterator = iter(test_ds)
        indices = []
        for j in range(len(test_ds)):
        # for j in range(5):
            print("prediction: {} of {}".format(j,len(test_ds)),end='\r')
            item = iterator.get_next()
            prediction = model.predict(item[0])
            predictions.append(prediction)
            # print(prediction)
            labels.append(item[1].numpy().tolist())
        labels = [expandlabels(x,num_outputs)for x in labels]
        predEntropy =[]
        predClasses =[]
        print("predictions complete, analyizing")
        for i,output in enumerate(predictions):
            # print(output)
            for k, pred in enumerate(output):
                pred_classes=[]
                pred_entropy = []
                print("image: {} of {}".format(i,len(predictions)),end='\r')
                for l, branch in enumerate(pred):
                    Pclass = np.argmax(branch[0])
                    pred_classes.append(Pclass) 
                    pred_entropy.append(calcEntropy(branch[0]))                       
                predClasses.append(pred_classes)
                predEntropy.append(pred_entropy)
                
        # print(predClasses)
        # print(predEntropy)
        # print(labels)
        # labels = list(map(expandlabels,labels,num_outputs))
        labelClasses = [0,1,2,3,4,5,6,7,8,9]
        results = eval.entropyConfusionMatrix(predClasses, labels,predEntropy, num_outputs,labelClasses,output_names)
        print(results)
        # print(pd.DataFrame(results).T)
        return

    def BranchEntropyMatrix(self, model, dataset):
        num_outputs = len(model.outputs) # the number of output layers for the purpose of providing labels
        
        output_names = [i.name for i in model.outputs]
        (train_images, train_labels), (test_images, test_labels) = dataset
        train_ds, test_ds, validation_ds = self.prepareAlexNetDataset(dataset,1)
        
        predictions = []
        labels = []
        model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(lr=0.001), metrics=['accuracy'])
        iterator = iter(test_ds)
        indices = []
        # for j in range(len(test_ds)):
        for j in range(5):

            print("prediction: {} of {}".format(j,len(test_ds)),end='\r')

            item = iterator.get_next()
            prediction = model.predict(item[0])
            predictions.append(prediction)
            # print(prediction)
            labels.append(item[1].numpy().tolist())
        labels = [expandlabels(x,num_outputs)for x in labels]
        predEntropy =[]
        predClasses =[]
        print("predictions complete, analyizing")
        for i,output in enumerate(predictions):
            # print(output)
            for k, pred in enumerate(output):
                pred_classes=[]
                pred_entropy = []
                print("image: {} of {}".format(i,len(predictions)),end='\r')
                for l, branch in enumerate(pred):
                    Pclass = np.argmax(branch[0])
                    pred_classes.append(Pclass) 
                    pred_entropy.append(calcEntropy(branch[0]))                       
                predClasses.append(pred_classes)
                predEntropy.append(pred_entropy)
                
        print(predClasses)
        # print(predEntropy)
        # print(labels)
        # labels = list(map(expandlabels,labels,num_outputs))
        labelClasses = [0,1,2,3,4,5,6,7,8,9]
        results = eval.entropyMatrix(predEntropy, labels, num_outputs,labelClasses,output_names)
        print(results)
        # print(pd.DataFrame(results).T)
        return

    def evalBranchMatrix(self, model, dataset):
        num_outputs = len(model.outputs) # the number of output layers for the purpose of providing labels
        
        output_names = [i.name for i in model.outputs]
        train_ds, test_ds, validation_ds = self.prepareAlexNetDataset(dataset,1)
        
        predictions = []
        labels = []
        model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(lr=0.001), metrics=['accuracy'])
        iterator = iter(test_ds)
        indices = []
        # for j in range(len(test_ds)):
        for j in range(len(test_ds)):
            print("prediction: {} of {}".format(j,len(test_ds)),end='\r')
            item = iterator.get_next()
            prediction = model.predict(item[0])
            predictions.append(prediction)
            # print(prediction)
            labels.append(item[1].numpy().tolist())
        labels = [expandlabels(x,num_outputs)for x in labels]
        predClasses =[]
        print("predictions complete, analyizing")
        for i,output in enumerate(predictions):
            # print(output)
            for k, pred in enumerate(output):
                pred_classes=[]               
                print("image: {} of {}".format(i,len(predictions)),end='\r')
                for l, branch in enumerate(pred):
                    # print(l)
                    Pclass = np.argmax(branch[0])
                    pred_classes.append(Pclass)     
                predClasses.append(pred_classes)
                


        # labels = list(map(expandlabels,labels,num_outputs))
        labelClasses = [0,1,2,3,4,5,6,7,8,9]
        results = eval.throughputMatrix(predClasses, labels, num_outputs,labelClasses,output_names)
        print(results)
        print(pd.DataFrame(results).T)
        return




    def find_mistakes(self, model, dataset, count = 1):
        """
            find rows that the model gets wrong
        """
        num_outputs = len(model.outputs) # the number of output layers for the purpose of providing labels
        (train_images, train_labels), (test_images, test_labels) = dataset
        
        
        print("ALEXNET {}".format(self.ALEXNET))
        if self.ALEXNET:
            train_ds, test_ds, validation_ds = self.prepareAlexNetDataset(dataset,1)
        else: 
            test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
            test_ds_size = len(list(test_ds))
            test_ds =  (test_ds
                # .map(augment_images)
                .batch(batch_size=1, drop_remainder=True))
        
        if self.ALEXNET: 
            model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(lr=0.001), metrics=['accuracy'])
            iterator = iter(test_ds)
            indices = []

            for j in range(5):
                item = iterator.get_next()

                # for j in [33,2,3,4]:
                # img = test_images[j].reshape(1,784)
                prediction = model.predict(item[0])
                print(np.array(prediction))
                # print(prediction)
                for i,exit in enumerate(prediction[0]):
                    for k, elem in enumerate(exit):
                        # print(i,end='\r')
                        Pclass = np.argmax(exit)
                        Pprob = exit[0][Pclass]
                        print("image:{}, exit:{}, pred Class:{}, probability: {} actual Class:{}".format(j,i, Pclass,Pprob, item[1]))
                        if Pclass != item[1]:
                            indices.append(j)
                        entropy = calcEntropy(elem)
                        print("entropy:{}".format(entropy))
                    
            print(indices)
        else:
            model.compile(loss="sparse_categorical_crossentropy", optimizer=keras.optimizers.Adam(),metrics=["accuracy"])
            iterator = iter(test_ds)
            indices = []
            # item = iterator.get_next()

            for j in [33,2,3,4]:
                img = test_images[j].reshape(1,784)
                prediction = model.predict(img)

                # print(prediction)
                for i,elem in enumerate(prediction[0][1]):
                    # print(i,end='\r')
                    Pclass = np.argmax(elem)
                    if Pclass != test_labels[j]:
                        print("test image: {}, pred Class:{}, actual Class:{}".format(j, Pclass,test_labels[j]))
                        indices.append(i)
                        entropy = calcEntropy(elem)
                        print("entropy:{}".format(entropy))
                print(indices)    
        pass


        return 