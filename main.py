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
os.environ["PATH"] += os.pathsep + 'C:/Program Files/graphviz/bin/'
from tensorflow.keras.utils import plot_model

#Visualize Model

def visualize_model(model):
    # tf.keras.utils.plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

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

        return results


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
        model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")
        model.summary()
        visualize_model(model)
        print(len(model.outputs))

        return model

    def mnistExample(self):

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
        model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")
        model.summary()
        visualize_model(model)
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

        print(logs)
        return model



if __name__ == "__main__":

    branchy = BranchyNet()
    # x = branchy.mainBranch()
    # x = branchy.mnistNormal()

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
    # outputs.append(output1)
    model = keras.Model(inputs=inputs, outputs=output1, name="mnist_model")

    model.save("models/mnistNormal2.hdf5")

    new_model = keras.models.load_model('models/mnistNormal2.hdf5')
    x = new_model
    # x = branchy.mnistExample()
    # x = branchy.loadModel("models/mnistNormal2.hdf5")

    x = branchy.trainMnist(x, 1,save = False)
    x.save("models/mnistNormal2_trained.hdf5",)

    new_model = keras.models.load_model('models/mnistNormal2_trained.hdf5')
    new_model.summary()

    pass