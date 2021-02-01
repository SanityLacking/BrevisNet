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


    def mainBranch(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(784,)),
            tf.keras.layers.Dense(512, activation=tf.nn.relu),
            
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        ])
        return model

        
    def prepareAlexNetDataset(self, dataset, batch_size =32):
        (train_images, train_labels), (test_images, test_labels) = dataset

        validation_images, validation_labels = train_images[:5000], train_labels[:5000]
        train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
        validation_ds = tf.data.Dataset.from_tensor_slices((validation_images, validation_labels))

        train_ds_size = len(list(train_ds))
        train_ds_size = len(list(test_ds))
        validation_ds_size = len(list(validation_ds))

        train_ds = (train_ds
            .map(augment_images)
            .shuffle(buffer_size=int(train_ds_size),reshuffle_each_iteration=True)
            .batch(batch_size=1, drop_remainder=True))

        test_ds = (test_ds
            .map(augment_images)
            # .shuffle(buffer_size=int(train_ds_size)) ##why would you shuffle the test set?
            .batch(batch_size=1, drop_remainder=True))

        validation_ds = (validation_ds
            .map(augment_images)
            # .shuffle(buffer_size=int(train_ds_size))
            .batch(batch_size=1, drop_remainder=True))

        return train_ds, test_ds, validation_ds

    

    def mnistNormal(self):
        outputs =[]
        inputs = keras.Input(shape=(784,))
        x = layers.Flatten(input_shape=(28,28))(inputs)
        x = layers.Dense(512, activation="relu")(x)
        x= layers.Dropout(0.2)(x)
        #exit 2
        x = layers.Dense(512, activation="relu")(x)
        x= layers.Dropout(0.2)(x)
        #exit 3
        x = layers.Dense(512, activation="relu")(x)
        x= layers.Dropout(0.2)(x)
        #exit 4
        x = layers.Dense(512, activation="relu")(x)
        x= layers.Dropout(0.2)(x)
        #exit 5
        x = layers.Dense(512, activation="relu")(x)
        x= layers.Dropout(0.2)(x)
        #exit 1 The main branch exit is refered to as "exit 1" or "main exit" to avoid confusion when adding addtional exits
        output1 = layers.Dense(10, name="output1")(x)
        softmax = layers.Softmax()(output1)

        outputs.append(softmax)
        print(len(outputs))
        model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model_normal")
        model.summary()
        #visualize_model(model,"mnist_normal")
        print(len(model.outputs))

        return model

    def mnistBranchy(self):

        outputs =[]
        inputs = keras.Input(shape=(784,))
        x = layers.Flatten(input_shape=(28,28))(inputs)
        x = layers.Dense(512, activation="relu")(x)
        x= layers.Dropout(0.2)(x)        
        #exit 2
        outputs = newBranch(x,outputs)
        # outputs.append(layers.Dense(10, name="output2")(x))

        x = layers.Dense(512, activation="relu")(x)
        x= layers.Dropout(0.2)(x)
        #exit 3
        # outputs.append(layers.Dense(10, name="output3")(x))
        outputs = newBranch(x,outputs)
        x = layers.Dense(512, activation="relu")(x)
        x= layers.Dropout(0.2)(x)
        #exit 4
        # outputs.append(layers.Dense(10, name="output4")(x))
        outputs = newBranch(x,outputs)
        x = layers.Dense(512, activation="relu")(x)
        x= layers.Dropout(0.2)(x)
        #exit 5
        # outputs.append(layers.Dense(10, name="output5")(x))
        outputs = newBranch(x,outputs)
        x = layers.Dense(512, activation="relu")(x)
        x= layers.Dropout(0.2)(x)
        #exit 1 The main branch exit is refered to as "exit 1" or "main exit" to avoid confusion when adding addtional exits
        output1 = layers.Dense(10, name="output1")(x)
        softmax = layers.Softmax()(output1)
        # x = layers.Dense(64, activation="relu")(x)
        # output2 = layers.Dense(10, name="output2")(x)
        outputs.append(softmax)
        model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model_branched")
        model.summary()
        visualize_model(model,"mnist_branched")

        return model
    def mnistAddBranches(self,model):
        """add branches to the mnist model, aka modifying an existing model to include branches."""
        print(model.inputs)
        inputs = model.inputs
        outputs = []
        print(model.outputs)
        outputs.append(model.outputs)
        for i in range(len(model.layers)):
            print(model.layers[i].name)
            if "dense" in model.layers[i].name:

                outputs = newBranch(model.layers[i].output,outputs)
            # for j in range(len(model.layers[i].inbound_nodes)):
            #     print(dir(model.layers[i].inbound_nodes[j]))
            #     print("inboundNode: " + model.layers[i].inbound_nodes[j].name)
            #     print("outboundNode: " + model.layers[i].outbound_nodes[j].name)
        model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model_branched")
        return model

    def addBranches(self,model, identifier =[""], customBranch = []):
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


    def fullprint(*args, **kwargs):
        from pprint import pprint
        import numpy
        opt = numpy.get_printoptions()
        numpy.set_printoptions(threshold=numpy.inf)
        pprint(*args, **kwargs)
        numpy.set_printoptions(**opt)

    # def loadModel(self,modelName):
    #     with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
    #         KerasModel = load_model(modelName, {'optimizer':'adam',
    #           'loss':'sparse_categorical_crossentropy',
    #           'metrics':['accuracy']})
    #         KerasModel.summary()
    #         config = KerasModel.get_config()   
    #         return KerasModel


   


    def trainModel(self, model, dataset, epocs = 2,save = False):
        """
        Train the model that is passed through. This function works for both single and multiple branch models.
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
  

    def trainModelTransfer(self, model, dataset, resetBranches = False, epocs = 2,save = False,transfer = True):
        """
        Train the model that is passed using transfer learning. This function expects a model with trained main branches and untrained (or randomized) side branches.
        """
        logs = []
        num_outputs = len(model.outputs) # the number of output layers for the purpose of providing labels

        (train_images, train_labels), (test_images, test_labels) = dataset
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
        checkpoint = keras.callbacks.ModelCheckpoint("models/{}_branched.hdf5".format(model.name), monitor='val_loss', verbose=1, mode='max')

        for j in range(epocs):
            print("epoc: {}".format(j))
            results = [j]           
            history = model.fit(train_ds, batch_size=64, epochs=epocs, validation_data=validation_ds, validation_steps=10, callbacks=[tensorboard_cb,checkpoint])
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


    def Run_mnistNormal(self, numEpocs = 2):
        """ load a mnist model, add branches to it and train using transfer learning function
        """
        x = self.mnistNormal()
        x = self.trainModel(x,self.loadTrainingData(), epocs = numEpocs,save = True)
        return x


    def Run_mnistTransfer(self, numEpocs = 2):
        """ load a mnist model, add branches to it and train using transfer learning function
        """
        x = tf.keras.models.load_model("models/mnist_trained_.hdf5")
        x = self.addBranches(x,["dropout_1","dropout_2","dropout_3","dropout_4",],newBranch)
        x = self.trainModelTransfer(x,self.loadTrainingData(),epocs = numEpocs, save = True)
        return x
    
    def Run_alexNet(self, numEpocs = 2, modelName="", saveName ="",transfer = True):
        if modelName =="":
            x = tf.keras.models.load_model("models/alexNetv3_new.hdf5")
        else:
            x = tf.keras.models.load_model("models/{}".format(modelName))

        x.summary()
        if saveName =="":
            pass
        else:
            x._name = saveName

        # funcModel = models.Model([input_layer], [prev_layer])
        funcModel = self.addBranches(x,["dense","conv2d","max_pooling2d","batch_normalization","dense","dropout"],newBranch_oneLayer)
        # funcModel = branchy.addBranches(x,["dense_1"],newBranch)

        funcModel.summary()
        funcModel = self.trainModelTransfer(funcModel,tf.keras.datasets.cifar10.load_data(),epocs = numEpocs, save = False, transfer = transfer)
        if saveName == "":
            funcModel.save("models/alexnet_branch_pooling.hdf5")
        else: 
            funcModel.save("models/{}.hdf5".format(saveName))

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
        x = self.anchy.trainModelTransfer(x,dataset,epocs = numEpocs, save = False)


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
    
    def GetResultsCSV(self,model,dataset):
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
        for j in range(1000):

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
        predClasses = pd.DataFrame(predClasses)
        labels = pd.DataFrame(labels)
        predEntropy = pd.DataFrame(predEntropy)

        predClasses.to_csv("results/predClasses.csv", sep=',', mode='a')
        labels.to_csv("results/labels.csv", sep=',', mode='a')
        predEntropy.to_csv("results/predEntropy.csv", sep=',', mode='a')


        # results = KneeGraph(predClasses, labels,predEntropy, num_outputs,labelClasses,output_names)
        # results.to_csv("logs_entropy/{}_{}_entropyStats.csv".format(model.name,time.strftime("%Y%m%d_%H%M%S")), sep=',', mode='a')
        return

    def BranchKneeGraph(self,model,dataset):
        num_outputs = len(model.outputs) # the number of output layers for the purpose of providing labels
        
        output_names = [i.name for i in model.outputs]
        (train_images, train_labels), (test_images, test_labels) = dataset
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
        results = KneeGraph(predClasses, labels,predEntropy, num_outputs,labelClasses,output_names)
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
        (train_images, train_labels), (test_images, test_labels) = dataset
        train_ds, test_ds, validation_ds = self.prepareAlexNetDataset(dataset,1)
        
        predictions = []
        labels = []
        model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(lr=0.001), metrics=['accuracy'])
        iterator = iter(test_ds)
        indices = []
        # for j in range(len(test_ds)):
        for j in range(10):
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
        results = KneeGraphClasses(predClasses, labels,predEntropy, num_outputs,labelClasses,output_names)
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
        results = KneeGraphPredictedClasses(predClasses, labels,predEntropy, num_outputs,labelClasses,output_names)
        # f = open("logs_entropy/{}_{}_entropyStats.txt".format(model.name,time.strftime("%Y%m%d_%H%M%S")), "w")
        # f.write(json.dumps(results))
        # results.to_csv("logs_entropy/{}_{}_entropyClassesStats.csv".format(model.name,time.strftime("%Y%m%d_%H%M%S")), sep=',', mode='a')
        # f.close()
        # print(results)
        # print(pd.DataFrame(results).T)
        return


    def BranchEntropyConfusionMatrix(self, model, dataset):
        num_outputs = len(model.outputs) # the number of output layers for the purpose of providing labels
        
        output_names = [i.name for i in model.outputs]
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
        results = entropyConfusionMatrix(predClasses, labels,predEntropy, num_outputs,labelClasses,output_names)
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
        results = entropyMatrix(predEntropy, labels, num_outputs,labelClasses,output_names)
        print(results)
        # print(pd.DataFrame(results).T)
        return

    def evalBranchMatrix(self, model, dataset):
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
                pred_entropy = []
                label_classes = []
                print("image: {} of {}".format(i,len(predictions)),end='\r')
                for l, branch in enumerate(pred):
                    # print(l)
                    Pclass = np.argmax(branch[0])
                    pred_classes.append(Pclass)     
                predClasses.append(pred_classes)
                


        # labels = list(map(expandlabels,labels,num_outputs))
        labelClasses = [0,1,2,3,4,5,6,7,8,9]
        results = throughputMatrix(predClasses, labels, num_outputs,labelClasses,output_names)
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