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

root_logdir = os.path.join(os.curdir, "logs\\fit\\")
def get_run_logdir():
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)

run_logdir = get_run_logdir()
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)

def augment_images(image, label,label2=""):
            # Normalize images to have a mean of 0 and standard deviation of 1
            image = tf.image.per_image_standardization(image)
            # Resize images from 32x32 to 277x277
            image = tf.image.resize(image, (227,227))
            return image, label,label

class BranchyNet:
    def mainBranch(self):
        model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(784,)),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
        return model

        
    

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
        
        # model = keras.Model([model_old.input], [model_old.output], name="{}_branched".format(model_old.name))
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


    def loadTrainingData(self):
        """ load dataset and configure it. 
        """
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        x_train = x_train.reshape(60000, 784).astype("float32") / 255
        x_test = x_test.reshape(10000, 784).astype("float32") / 255

        return (x_train, y_train), (x_test, y_test)


    def trainModel(self, model, dataset, epocs = 2,save = False):
        """
        Train the model that is passed through. This function works for both single and multiple branch models.
        """
        logs = []
        (x_train, y_train), (x_test, y_test) = dataset
        num_outputs = len(model.outputs) # the number of output layers for the purpose of providing labels
        model.compile(loss="sparse_categorical_crossentropy", optimizer=keras.optimizers.Adam(),metrics=["accuracy"])

        for j in range(epocs):
            results = [j]           
            history = model.fit(x_train, list(itertools.repeat(y_train,num_outputs)), batch_size=64, epochs=1, validation_split=0.2)
            print(history)
            test_scores = model.evaluate(x_test, list(itertools.repeat(y_test,num_outputs)), verbose=2)
            print("overall loss: {}".format(test_scores[0]))
            if num_outputs > 1:
                for i in range(num_outputs):
                    print("Output {}: Test loss: {}, Test accuracy {}".format(i, test_scores[i+1], test_scores[i+1+num_outputs]))
                    results.append("Output {}: Test loss: {}, Test accuracy {}".format(i, test_scores[i+1], test_scores[i+1+num_outputs]))
            else:
                print("Test loss:", test_scores[0])
                print("Test accuracy:", test_scores[1])
            logs.append(results)
        # fullprint(model.predict(x_test[:1]))
        if save:
            saveModel(model,"mnist_trained")

        results = model.predict(x_test[:1])
        fullprint(results)
        S = entropy(results)
        print(S)
        # print(logs)
        return model

    def trainModelTransfer(self, model, dataset, resetBranches = False, epocs = 2,save = False):
        """
        Train the model that is passed using transfer learning. This function expects a model with trained main branches and untrained (or randomized) side branches.
        """
        logs = []
        num_outputs = len(model.outputs) # the number of output layers for the purpose of providing labels



        (train_images, train_labels), (test_images, test_labels) = dataset 
        CLASS_NAMES= ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        validation_images, validation_labels = train_images[:5000], train_labels[:5000]
        train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels,train_labels))
        test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels, test_labels))
        validation_ds = tf.data.Dataset.from_tensor_slices((validation_images, validation_labels,validation_labels))

        train_ds_size = len(list(train_ds))
        train_ds_size = len(list(test_ds))
        validation_ds_size = len(list(validation_ds))

        train_ds = (train_ds
                        .map(augment_images)
                        .shuffle(buffer_size=train_ds_size)
                        .batch(batch_size=32, drop_remainder=True))

        test_ds = (test_ds
                        .map(augment_images)
                        .shuffle(buffer_size=train_ds_size)
                        .batch(batch_size=32, drop_remainder=True))

        validation_ds = (validation_ds
                        .map(augment_images)
                        .shuffle(buffer_size=train_ds_size)
                        .batch(batch_size=32, drop_remainder=True))



        # print(np.repeat([y_train],1))
        # print("before reset:")
        # model.compile(loss="sparse_categorical_crossentropy", optimizer=keras.optimizers.Adam(),metrics=["accuracy"])
        # test_scores = model.evaluate(x_test, list(itertools.repeat(y_test,num_outputs)), verbose=2)
        # printTestScores(test_scores,num_outputs)

        #Freeze main branch layers
        #how to iterate through layers and find main branch ones?
        #simple fix for now: all branch nodes get branch in name.
        if resetBranches: 
            reset_branch_weights(model)
        for i in range(len(model.layers)):
            print(model.layers[i].name)
            # print(model.layers[i].initial_weights)

            if "branch" in model.layers[i].name:
                print("setting branch layer training to true")
                model.layers[i].trainable = True
            else: 
                print("setting main layer training to false")
                model.layers[i].trainable = False
            # for j in range(len(model.layers[i].inbound_nodes)):
            #     print(dir(model.layers[i].inbound_nodes[j]))
            #     print("inboundNode: " + model.layers[i].inbound_nodes[j].name)
            #     print("outboundNode: " + model.layers[i].outbound_nodes[j].name)

        # model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=keras.optimizers.Adam(),metrics=["accuracy"])
        # model.compile(loss="sparse_categorical_crossentropy", optimizer=keras.optimizers.Adam(),metrics=["accuracy"])
        model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(lr=0.001), metrics=['accuracy'])

        print("after reset:")
        test_scores = model.evaluate(test_ds, verbose=2)
        printTestScores(test_scores,num_outputs)
        checkpoint = keras.callbacks.ModelCheckpoint("models/", monitor='val_acc', verbose=1, save_best_only=True, mode='max')


        for j in range(epocs):
            print("epoc: {}".format(j))
            results = [j]           
            # if (ds_check):
                # history = model.fit(train_ds, batch_size=64, epochs=1, validation_data=validation_ds, validation_freq=1, callbacks=[tensorboard_cb,checkpoint])
            # else:
            history = model.fit(train_ds, batch_size=64, epochs=1, callbacks=[tensorboard_cb,checkpoint])
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
        # fullprint(model.predict(x_test[:1]))
        if save:
            saveModel(model,"model_transfer_trained")

        # results = model.predict(x_test[:1])
        # fullprint(results)
        # S = entropy(results)
        # print(S)


        # print(logs)
        return model


        ###### RUN MODEL SHORTCUTS ######

  



    def Run_mnistNormal(self, numEpocs = 2):
        """ load a mnist model, add branches to it and train using transfer learning function
        """
        x = branchy.mnistNormal()
        x = branchy.trainModel(x,self.loadTrainingData(), epocs = numEpocs,save = False)
        return x


    def Run_mnistTransfer(self, numEpocs = 2):
        """ load a mnist model, add branches to it and train using transfer learning function
        """
        x = tf.keras.models.load_model("models/mnist2_trained_.tf")
        x = branchy.addBranches(x,["dropout"],newBranch)
        x = branchy.trainModelTransfer(x,self.loadTrainingData(),epocs = numEpocs, save = False)
        return x
    
    def Run_alexNet(self, numEpocs = 2):
        
        x = tf.keras.models.load_model("models/alexNet_v3.hdf5")
        # x.summary()
        # train_ds, test_ds, validation_ds = loadDataPipeline()
        # print(train_generator.class_indices)
        # print(validation_generator.class_indices)
        # x.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'] )
        # predict = x.evaluate(validation_generator,steps = 32)
        # print(predict)
        

        # input_layer = layers.Input(batch_shape=x.layers[0].input_shape)
        # prev_layer = input_layer
        # for layer in x.layers:
        #     layer._inbound_nodes = []
        #     prev_layer = layer(prev_layer)
        
        # input_layer = layers.Input(batch_shape=x.layers[0].input_shape)
        # prev_layer = input_layer
        # for layer in x.layers:
            # prev_layer = layer(prev_layer)
        
        

        # funcModel = models.Model([input_layer], [prev_layer])
        funcModel = branchy.addBranches(x,["dense1"],newBranch)
        # funcModel = branchy.addBranches(x,["dense_1"],newBranch)

        funcModel.summary()
        funcModel = branchy.trainModelTransfer(funcModel,tf.keras.datasets.cifar10.load_data(),epocs = numEpocs, save = False)
        funcModel.save("models/alexnet_branched.hdf5")


        # x = keras.Model(inputs=x.inputs, outputs=x.outputs, name="{}_normal".format(x.name))



        return x



def newBranchCustom(prevLayer, outputs=[]):
    """ example of a custom branching layer, used as a drop in replacement of "newBranch"
    """                 
    branchLayer = layers.Dense(10, name=tf.compat.v1.get_default_graph().unique_name("branch_output"))(prevLayer)
    outputs.append(layers.Softmax(name=tf.compat.v1.get_default_graph().unique_name("branch_softmax"))(branchLayer))

    return outputs




if __name__ == "__main__":
    branchy = BranchyNet()
    # x = branchy.Run_mnistNormal(1)
    # x = branchy.Run_mnistTransfer(1)
    x = branchy.Run_alexNet(1)

    # x = branchy.mnistBranchy()
    

    # x = branchy.loadModel("models/mnist_trained_20-12-15_112434.hdf5")
    # x = tf.keras.models.load_model("models/mnist2_transfer_trained_.tf")

    # x.save("models/mnistNormal2_trained.hdf5")
    # saveModel(x,"mnist2_transfer_trained_final",includeDate=False)
    pass

