# import the necessary packages
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
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


def newBranch(prevLayer, outputs):
    branchLayer = layers.Dense(124, activation="relu",name=tf.compat.v1.get_default_graph().unique_name("branch124"))(prevLayer)
    branchLayer = layers.Dense(64, activation="relu",name=tf.compat.v1.get_default_graph().unique_name("branch64"))(branchLayer)
    branchLayer = layers.Dense(10, name=tf.compat.v1.get_default_graph().unique_name("branch_output"))(branchLayer)
    outputs.append(layers.Softmax(name=tf.compat.v1.get_default_graph().unique_name("branch_softmax"))(branchLayer))

    return outputs

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



def saveModel(model,name,overwrite = True, includeDate= True, folder ="models", fileFormat = "tf"):
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

        outputs = []
        outputs.append(model.outputs) #get model outputs that already exist 

        if type(identifier) != list:
            identifier = [identifier]
        

        if type(customBranch) != list:
            customBranch = [customBranch]
        if len(customBranch) == 0:
            customBranch = [newBranch]

        if len(identifier) > 0:
            print(">0")
            if type(identifier[0]) == int:
                print("int")
                for i in identifier: 
                    print(model.layers[i].name)
                    try:
                        outputs = customBranch[min(i, len(customBranch))-1](model.layers[i].output,outputs)
                        # outputs = newBranch(model.layers[i].output,outputs)
                    except:
                        pass
            else:
                print("abc")
                for i in range(len(model.layers)):
                    print(model.layers[i].name)
                    if any(id in model.layers[i].name for id in identifier):
                        outputs = customBranch[min(i, len(customBranch))-1](model.layers[i].output,outputs)
                        # outputs = newBranch(model.layers[i].output,outputs)
        else: #if identifier is blank or empty
            print("nothing")
            for i in range(1-len(model.layers)-1):
                print(model.layers[i].name)
                # if "dense" in model.layers[i].name:
                # outputs = newBranch(model.layers[i].output,outputs)
                outputs = customBranch[max(i, len(customBranch))](model.layers[i].output,outputs)
            # for j in range(len(model.layers[i].inbound_nodes)):
            #     print(dir(model.layers[i].inbound_nodes[j]))
            #     print("inboundNode: " + model.layers[i].inbound_nodes[j].name)
            #     print("outboundNode: " + model.layers[i].outbound_nodes[j].name)

        model = keras.Model(inputs=model.inputs, outputs=outputs, name="{}_branched".format(model.name))
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




    def trainModel(self, model, epocs = 2,save = False):
        """
        Train the model that is passed through. This function works for both single and multiple branch models.
        """
        logs = []
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        num_outputs = len(model.outputs) # the number of output layers for the purpose of providing labels
        # print(np.repeat([y_train],1))
        x_train = x_train.reshape(60000, 784).astype("float32") / 255
        x_test = x_test.reshape(10000, 784).astype("float32") / 255
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

    def trainModelTransfer(self, model, resetBranches = False, epocs = 2,save = False):
        """
        Train the model that is passed using transfer learning. This function expects a model with trained main branches and untrained (or randomized) side branches.
        """
        logs = []
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        num_outputs = len(model.outputs) # the number of output layers for the purpose of providing labels
        # print(np.repeat([y_train],1))
        x_train = x_train.reshape(60000, 784).astype("float32") / 255
        x_test = x_test.reshape(10000, 784).astype("float32") / 255


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
        model.compile(loss="sparse_categorical_crossentropy", optimizer=keras.optimizers.Adam(),metrics=["accuracy"])

        print("after reset:")
        test_scores = model.evaluate(x_test, list(itertools.repeat(y_test,num_outputs)), verbose=2)
        printTestScores(test_scores,num_outputs)


        for j in range(epocs):
            print("epoc: {}".format(j))
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
            saveModel(model,"mnist_transfer_trained")

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
        x = branchy.trainModel(x,epocs = numEpocs,save = False)
        return x


    def Run_mnistTransfer(self, numEpocs = 2):
        """ load a mnist model, add branches to it and train using transfer learning function
        """
        x = tf.keras.models.load_model("models/mnist2_trained_.tf")
        x.summary()
        x = branchy.addBranches(x,["dropout"],)
        x.summary()
        # x = branchy.trainModelTransfer(x,epocs = numEpocs, save = False)
        return x






if __name__ == "__main__":
    x = [[-4.535223 , -1.5143484,  3.982851 ,  2.1995668, -5.4335203,
        -5.476383 , -8.685219 , 12.729579 , -4.2230687,  0.6178443]]
    s =entropy(x)
    print(s)


    if True:        
        branchy = BranchyNet()
        # x = branchy.Run_mnistNormal(1)
        x = branchy.Run_mnistTransfer(1)

        # x = branchy.mnistBranchy()
        

        # x = branchy.loadModel("models/mnist_trained_20-12-15_112434.hdf5")
        # x = tf.keras.models.load_model("models/mnist2_transfer_trained_.tf")

        # x.save("models/mnistNormal2_trained.hdf5")
        # saveModel(x,"mnist2_transfer_trained_final",includeDate=False)

    pass

