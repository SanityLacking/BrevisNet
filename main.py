# import the necessary packages
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
import itertools

from keras.models import load_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform

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
        print(prevLayer)
        print(prevLayer.shape)
        branchLayer = layers.Dense(124, activation="relu")(prevLayer)
        branchLayer = layers.Dense(64, activation="relu")(branchLayer)
        outputs.append(layers.Dense(10, name="branch_output_{}".format(len(outputs)+1))(branchLayer))
        return outputs

def calcEntropy(y_hat):
        #entropy is the sum of y * log(y) for all possible labels.
        results =[]
        entropy = 0
        for i in range(len(y_hat)):
            entropy += y_hat[i] * math.log(y_hat(i))
            print(entropy)

        return results


def saveModel(model,name,overwrite = True, folder ="models", fileFormat = "hdf5"):
    from datetime import datetime
    import os
    now = datetime.now() # current date and time
    
    if not os.path.exists(folder):
        try:
            os.mkdir(folder)
        except FileExistsError:
            pass
    try:
        model.save("{}{}_{}.{}".format(folder+"\\",name,now.strftime("%y-%m-%d_%H%M%S"),fileFormat))
    except OSError:
        pass
    return True

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
        outputs.append(output1)
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
        # x = layers.Dense(64, activation="relu")(x)
        # output2 = layers.Dense(10, name="output2")(x)
        outputs.append(output1)
        print(len(outputs))
        model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model_branched")
        model.summary()
        visualize_model(model,"mnist_branched")
        print(len(model.outputs))

        return model
  
    def fullprint(*args, **kwargs):
        from pprint import pprint
        import numpy
        opt = numpy.get_printoptions()
        numpy.set_printoptions(threshold=numpy.inf)
        pprint(*args, **kwargs)
        numpy.set_printoptions(**opt)

    def loadModel(self,modelName):
        with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
            KerasModel = load_model(modelName, {'optimizer':'adam',
              'loss':'sparse_categorical_crossentropy',
              'metrics':['accuracy']})
            KerasModel.summary()
            config = KerasModel.get_config()   
            return KerasModel

    def trainMnist(self, model, epocs = 2,save = False):
        logs = []
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        num_outputs = len(model.outputs) # the number of output layers for the purpose of providing labels
        # print(np.repeat([y_train],1))
        x_train = x_train.reshape(60000, 784).astype("float32") / 255
        x_test = x_test.reshape(10000, 784).astype("float32") / 255
        model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=keras.optimizers.Adam(),metrics=["accuracy"])

        for j in range(epocs):
            results = [j]           
            history = model.fit(x_train, list(itertools.repeat(y_train,num_outputs)), batch_size=64, epochs=1, validation_split=0.2)
            print(history)
            test_scores = model.evaluate(x_test, list(itertools.repeat(y_test,num_outputs)), verbose=2)
            # print(test_scores)
            print("overall loss: {}".format(test_scores[0]))
            # results.append("Output {}: Test loss: {}, Test accuracy {}".format(0, test_scores[1], test_scores[2]))
            if num_outputs > 1:
                for i in range(num_outputs):
                    print("Output {}: Test loss: {}, Test accuracy {}".format(i, test_scores[i+1], test_scores[i+1+num_outputs]))
                    results.append("Output {}: Test loss: {}, Test accuracy {}".format(i, test_scores[i+1], test_scores[i+1+num_outputs]))
            else:
                # results.append("Output {}: Test loss: {}, Test accuracy {}".format(0, test_scores[1], test_scores[2]))
                print("Test loss:", test_scores[0])
                print("Test accuracy:", test_scores[1])
            logs.append(results)
        # fullprint(model.predict(x_test[:1]))
        if save:
            # with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
            model.save("models/mnistNormal2.hdf5")
            new_model = keras.models.load_model('models/mnistNormal2.hdf5')
            # new_model = load_model("models/mnistNormal2.hdf5")
            new_model.summary()

        results = model.predict(x_test[:1])
        fullprint(results)
        S = entropy(results)
        print(S)
        # print(logs)
        return model



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


if __name__ == "__main__":
    x = [[-4.535223 , -1.5143484,  3.982851 ,  2.1995668, -5.4335203,
        -5.476383 , -8.685219 , 12.729579 , -4.2230687,  0.6178443]]
    s =entropy(x)
    print(s)


    if True:        
        branchy = BranchyNet()
        # x = branchy.mainBranch()
        # x = branchy.mnistNormal()
        x = branchy.mnistBranchy()
        # inputs = keras.Input(shape=(784,))
        # x = layers.Flatten(input_shape=(28,28))(inputs)
        # x = layers.Dense(512, activation="relu")(x)
        # x= layers.Dropout(0.2)(x)
        # #exit 2
        # x = layers.Dense(512, activation="relu")(x)
        # x= layers.Dropout(0.2)(x)
        # #exit 3
        # x = layers.Dense(512, activation="relu")(x)
        # x= layers.Dropout(0.2)(x)
        # #exit 4
        # x = layers.Dense(512, activation="relu")(x)
        # x= layers.Dropout(0.2)(x)
        # #exit 5
        # x = layers.Dense(512, activation="relu")(x)
        # x= layers.Dropout(0.2)(x)
        # #exit 1 The main branch exit is refered to as "exit 1" or "main exit" to avoid confusion when adding addtional exits
        # output1 = layers.Dense(10, name="output1")(x)
        # # outputs.append(output1)
        # model = keras.Model(inputs=inputs, outputs=output1, name="mnist_model")

        # model.save("models/mnistNormal2.hdf5")

        # new_model = keras.models.load_model('models/mnistNormal2.hdf5')
        # x = new_model
        # x = branchy.mnistExample()
        # x = branchy.loadModel("models/mnistNormal2.hdf5")

        x = branchy.trainMnist(x, 1,save = False)
        # x.save("models/mnistNormal2_trained.hdf5")
        saveModel(x,"mnist2_trained")

        new_model = keras.models.load_model('models/mnistNormal2_trained.hdf5')
        new_model.summary()

    pass