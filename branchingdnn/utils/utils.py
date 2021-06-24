import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
import itertools
import glob
import os
import pandas as pd
# from keras.models import load_model
# from keras.utils import CustomObjectScope
# from keras.initializers import glorot_uniform

import pandas as pd
import math
import pydot
import os
import math
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
#os.environ["PATH"] += os.pathsep + "C:\Program Files\Graphviz\bin"
#from tensorflow.keras.utils import plot_model

MODEL_DIR = "models/"

def expandlabels(label,num_outputs):
    flattened = [val for sublist in label for val in sublist]
    label = flattened * num_outputs
    return label
    
def fullprint(*args, **kwargs):
        from pprint import pprint
        import numpy
        opt = numpy.get_printoptions()
        numpy.set_printoptions(threshold=numpy.inf)
        pprint(*args, **kwargs)
        numpy.set_printoptions(**opt)



def calcEntropy(y_hat):
        #entropy is the sum of y * log(y) for all possible labels.
        if isinstance(y_hat, list):
            y_hat = np.array(y_hat)
        sum_entropy = 0
        if y_hat.ndim >1:
            return list(map(calcEntropy,y_hat))
        for i in range(len(y_hat)):
            if y_hat[i] != 0: # log of zero is undefined, see MacKay's book "Information Theory, Inference, and Learning Algorithms"  for more info on this workaround reasoning.
                entropy =y_hat[i] * math.log(y_hat[i],2)
                sum_entropy +=  entropy

        return -sum_entropy

def calcEntropy_Tensors(y_hat):
        #entropy is the sum of y * log(y) for all possible labels.
        #log(0) is evaulated as NAN and then clipped to approaching zero
        #rank is used to reduce multi-dim arrays but leave alone 1d arrays.
        rank = tf.rank(y_hat)
        def calc_E(y_hat):
            results = tf.clip_by_value((tf.math.log(y_hat)/tf.math.log(tf.constant(2, dtype=y_hat.dtype))), -1e12, 1e12)
#             results = tf.clip_by_value(results, -1e12, 1e12)
#             print("res ", results)
            return tf.reduce_sum(y_hat * results)

        sumEntropies = (tf.map_fn(calc_E,tf.cast(y_hat,'float')))
        
        if rank == 1:
            sumEntropies = tf.reduce_sum(sumEntropies)
        return -sumEntropies

def calcEntropy_Tensors2(y_hat):
    #entropy is the sum of y * log(y) for all possible labels.
    #doesn't deal with cases of log(0)
    val = y_hat * tf.math.log(y_hat)/tf.math.log(tf.constant(2, dtype=y_hat.dtype))
    sumEntropies =  tf.reduce_mean(tf.boolean_mask(val,tf.math.is_finite(val)))
    return -sumEntropies

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



def saveModel(model,name,overwrite = True, includeDate= True, folder ="models", fileFormat = "hdf5"):
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


def newestModelPath(modelNames):
    """ returns the path of the newest model with modelname in the filename.
        does not check the actual model name allocated in the file.
        janky, but works
    """
    # if modelNames is not list:
        # modelNames = [modelNames]
    full_list = os.scandir(MODEL_DIR)
    # print(modelNames)
    # print(full_list)
    items = []
    for i in full_list:
        if modelNames in i.name:
            # print(i.name)
            # print(os.path.getmtime(i.path))
            items.append(i)
    
    items.sort(key=lambda x: os.path.getmtime(x.path), reverse=True)
    result = items[0].path

    return result


def augment_images(image, label):
    # Normalize images to have a mean of 0 and standard deviation of 1
    # image = tf.image.per_image_standardization(image)
    # Resize images from 32x32 to 277x277
    image = tf.image.resize(image, (227,227))
    return image, label
    
def resize(image):
    # Normalize images to have a mean of 0 and standard deviation of 1
    image = tf.image.per_image_standardization(image)
    # Resize images from 32x32 to 277x277
    image = tf.image.resize(image, (227,227))
    return image

def fullprint(*args, **kwargs):
    from pprint import pprint
    import numpy
    opt = numpy.get_printoptions()
    numpy.set_printoptions(threshold=numpy.inf)
    pprint(*args, **kwargs)
    numpy.set_printoptions(**opt)

def exitAccuracy(results, labels, classes=[]):
    """ find the accuracy scores of the main network exit for each class
            if classes is empty, return the average accuracy for all labels
    """
    classAcc = {}
    for i, labelClass in enumerate(classes):
        classAcc[labelClass] = results[np.where(labels==labelClass)].sum()/len(labels[np.where(labels == labelClass)])
    return classAcc

class ConfusionMatrixMetric(tf.keras.metrics.Metric):

            
    def update_state(self, y_true, y_pred,sample_weight=None):
        self.total_cm.assign_add(self.confusion_matrix(y_true,y_pred))
        return self.total_cm
        
    def result(self):
        return self.process_confusion_matrix()
    
    def confusion_matrix(self,y_true, y_pred):
        """
        Make a confusion matrix
        """
        y_pred=tf.argmax(y_pred,1)
        cm=tf.math.confusion_matrix(y_true,y_pred,dtype=tf.float32,num_classes=self.num_classes)
        return cm
    
    def process_confusion_matrix(self):
        "returns precision, recall and f1 along with overall accuracy"
        cm=self.total_cm
        diag_part=tf.linalg.diag_part(cm)
        precision=diag_part/(tf.reduce_sum(cm,0)+tf.constant(1e-15))
        recall=diag_part/(tf.reduce_sum(cm,1)+tf.constant(1e-15))
        f1=2*precision*recall/(precision+recall+tf.constant(1e-15))
        return precision,recall,f1


class  confidenceConditional(tf.keras.metrics.Metric):
    def __init__(self, **kwargs):
        # Initialise as normal and add flag variable for when to run computation
        super(confidenceConditional, self).__init__(**kwargs)
        self.metric_variable = self.add_weight(name='metric_variable', initializer='zeros')
        self.update_metric = tf.Variable(False)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Use conditional to determine if computation is done
        if self.update_metric:
            # run computation
            computation_result = confidenceScore_numpy(y_true,y_pred)
            self.metric_variable.assign_add(computation_result)

    def result(self):
        return self.metric_variable

    def reset_states(self):
        self.metric_variable.assign(0.)

class  unconfidenceConditional(tf.keras.metrics.Metric):
    def __init__(self, **kwargs):
        # Initialise as normal and add flag variable for when to run computation
        super(unconfidenceConditional, self).__init__(**kwargs)
        self.metric_variable = self.add_weight(name='metric_variable', initializer='zeros')
        self.update_metric = tf.Variable(False)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Use conditional to determine if computation is done
        if self.update_metric:
            # run computation
            computation_result = unconfidence(y_true,y_pred)
            self.metric_variable.assign_add(computation_result)

    def result(self):
        return self.metric_variable

    def reset_states(self):
        self.metric_variable.assign(0.)

class ToggleMetrics(tf.keras.callbacks.Callback):
    '''On test begin (i.e. when evaluate() is called or 
     validation data is run during fit()) toggle metric flag '''
    def on_test_begin(self, logs):
        for metric in self.model.metrics:
            if 'confidenceConditional' or 'unconfidenceConditional' in metric.name:
                metric.on.assign(True)
    def on_test_end(self,  logs):
        for metric in self.model.metrics:
            if 'confidenceConditional' or 'unconfidenceConditional' in metric.name:
                metric.on.assign(False)


def confidenceScore_numpy(y_true, y_pred):
    # Numpy confidence metric version
    y_true =np.array(y_true)
    y_pred = np.array(y_pred)
    def argmax(x):
        return [np.argmax(x)]
    pred_labels = list(map(argmax,np.array(y_pred)))
    x = np.where(np.equal(y_true,pred_labels) ==True)
    y = y_pred[x[0]]
    results = calcEntropy(y)
    # print(results)
    if not results:
        return 1e-8
    else:
        return np.median(results)

def confidenceScore(y_true, y_pred):
    # print(y_pred)
    # print(tf.keras.backend.get_value(y_pred))
    
    pred_labels = tf.math.argmax(y_pred,1)
    indexes = tf.where(tf.math.equal(pred_labels, tf.cast(tf.reshape(y_true,pred_labels.shape),'int64')))
    indexes = tf.reshape(indexes,[-1])
    entropies = tf.gather(y_pred,indexes)
    if tf.equal(tf.size(entropies), 0):
        correctEntropies = tf.cast(1e-8,'float')
    else:
        correctEntropies = calcEntropy_Tensors2(entropies)

    return correctEntropies


def unconfidence(y_true, y_pred):
        #avg confidence of incorrect items.
        # print(y_pred)
        # print(tf.keras.backend.get_value(y_pred))
        
        pred_labels = tf.math.argmax(y_pred,1)
        indexes = tf.where(tf.math.not_equal(pred_labels, tf.cast(tf.reshape(y_true,pred_labels.shape),'int64')))
        indexes = tf.reshape(indexes,[-1])
        entropies = tf.gather(y_pred,indexes)
        if tf.equal(tf.size(entropies), 0):
            incorrectEntropies = tf.cast(1e-8,'float')
        else:
            incorrectEntropies = calcEntropy_Tensors2(entropies)
        
        return incorrectEntropies


class EntropyConfidenceMetric(tf.keras.metrics.Metric):
    #metric of average confidence for correct answers          
    def update_state(self, y_true, y_pred,sample_weight=None):
        self.confidenceScore(y_true,y_pred)

        return self.AvgConfidence
        
    def confidenceScore(self, y_true, y_pred):
        self.AvgConfidence = -1
        pred_label = list(map(np.argmax,np.array(y_pred)))
        countCorrect=0
        for i in range(len(y_pred)):
            if pred_label[i] == y_true[i]:
                countCorrect += 1
                self.AvgConfidence += calcEntropy_Tensors(y_pred[i])
        
        if countCorrect == 0: #hack so i don't divide by zero
            countCorrect = 1

        self.AvgConfidence = self.AvgConfidence/countCorrect

def crossE_test(y_true, y_pred):
    crossE = tf.keras.losses.SparseCategoricalCrossentropy()
    scce = tf.math.add(crossE(y_true, y_pred), 10)
    # print("crossE",scce)
    return scce
def entropyAddition_loss():
    #create a wrapper function that returns a function

    crossE = tf.keras.losses.SparseCategoricalCrossentropy()
    def entropyAddition(y_true, y_pred):
        # print(y_true)
        # print(y_pred)
        #Entropy is added to the CrossE divided by the len of inputs
        pred_labels = tf.math.argmax(y_pred,1)
        indexes = tf.where(tf.math.equal(pred_labels, tf.cast(tf.reshape(y_true,pred_labels.shape),'int64')))
        indexes = tf.reshape(indexes,[-1])
        entropies = tf.gather(y_pred,indexes)
        if tf.equal(tf.size(entropies), 0):
            correctEntropies = tf.cast(1e-8,'float')
        else:
            # correctEntropies=0
            # correctEntropies = tf.reduce_mean(tf.map_fn(calcEntropy_Tensors,tf.cast(entropies,'float')))
            correctEntropies = calcEntropy_Tensors2(entropies)
        scce = crossE(y_true, y_pred)
        # print("scce",scce)
        # print("loss",correctEntropies)
        loss = scce + (correctEntropies * scce) + 1e-8
        # loss = correctEntropies
        # loss = scce 
        # print(loss)
        return loss
    return entropyAddition
        
    # return entropyAddition

def entropyMultiplication(y_true, y_pred):
    crossE = tf.keras.losses.SparseCategoricalCrossentropy()
    #Entropy is added to the CrossE divided by the len of inputs
    pred_labels = tf.math.argmax(y_pred,1)
    indexes = tf.where(tf.math.equal(pred_labels, tf.cast(tf.reshape(y_true,pred_labels.shape),'int64')))
    indexes = tf.reshape(indexes,[-1])
    entropies = tf.gather(y_pred,indexes)
    if tf.equal(tf.size(entropies), 0):
        correctEntropies = tf.cast(0,'float')
    else:
        correctEntropies = tf.reduce_mean(tf.map_fn(calcEntropy_Tensors,tf.cast(entropies,'float')))
    scce = crossE(y_true, y_pred)
    #note: this may cause issues if no correct entropies are found.
    if correctEntropies == 0:
        correctEntropies = 1
    loss = scce + (correctEntropies * scce)
    return loss

def custom_loss_addition(y_true, y_pred):
    #Entropy is added to the CrossE divided by the len of inputs
    pred_label = list(map(np.argmax,np.array(y_pred)))
    crossE = tf.keras.losses.SparseCategoricalCrossentropy()
    sumEntropy = 0
    for i in range(len(y_pred)):
        # print("Entropy : ",calcEntropy(y_pred[i]))
        if pred_label[i] == y_true[i]:
            sumEntropy += calcEntropy(y_pred[i])
    sumEntropy = sumEntropy / len(y_pred)         
    loss = crossE(y_true, y_pred)
    
    loss +=sumEntropy
    return loss

def custom_loss_multi(y_true, y_pred):
    #CrossE is multiplied by the Entropy
    pred_label = list(map(np.argmax,np.array(y_pred)))
    crossE = tf.keras.losses.SparseCategoricalCrossentropy()
    sumLoss = 0
    
    for i in range(len(y_pred)):
        loss = crossE(y_true[i], y_pred[i])
#         print('crossE: ',loss)
        if pred_label[i] == y_true[i]:
#             print('calcEntropy ',calcEntropy(y_pred[i]))
            loss = loss * calcEntropy(y_pred[i])
        sumLoss += loss
    sumLoss = sumLoss / len(y_pred)         
    
#     loss = crossE(y_true, y_pred)
#     print("CrossE : ",loss.numpy())
#     print("Loss : ",sumLoss)
    return sumLoss

def get_run_logdir(name=""):
    run_id = time.strftime("run_{}_%Y_%m_%d-%H_%M_%S".format(name))
    return os.path.join(root_logdir, run_id)