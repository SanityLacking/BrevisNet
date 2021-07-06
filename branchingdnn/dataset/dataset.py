# import the necessary packages
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model



class prepare:
    def augment_images(image, label,input_size):
            # Normalize images to have a mean of 0 and standard deviation of 1
            # image = tf.image.per_image_standardization(image)
            # Resize images from 32x32 to 277x277
            image = tf.image.resize(image,input_size)
            return image, label

    def dataset(dataset,batch_size=32, validation_size = 0, shuffle_size = 0, input_size=()):
        (train_images, train_labels), (test_images, test_labels) = dataset

        #hack to get around the limitation of providing additional parameters to the map function for the datasets below 
        def augment_images(image, label,input_size=input_size):
            return prepare.augment_images(image, label, input_size)
        
        validation_images, validation_labels = train_images[:validation_size], train_labels[:validation_size] #get the first 5k training samples as validation set
        train_images, train_labels = train_images[validation_size:], train_labels[validation_size:] # now remove the validation set from the training set.
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
                        .shuffle(buffer_size=tf.cast(shuffle_size,'int64'))
                        .batch(batch_size=batch_size, drop_remainder=True))

        test_ds = (test_ds
                        .map(augment_images)
                        #   .shuffle(buffer_size=train_ds_size)
                        .batch(batch_size=batch_size, drop_remainder=True))

        validation_ds = (validation_ds
                        .map(augment_images)
                        #   .shuffle(buffer_size=validation_ds_size)
                        .batch(batch_size=batch_size, drop_remainder=True))

        return train_ds, test_ds, validation_ds

    def prepareMnistDataset(dataset,batch_size=32):
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
                # .map(prepare.augment_images)
                .shuffle(buffer_size=int(train_ds_size),reshuffle_each_iteration=True)
                .batch(batch_size=batch_size, drop_remainder=True))
            test_ds = (test_ds
                # .map(prepare.augment_images)
                .shuffle(buffer_size=int(test_ds_size)) ##why would you shuffle the test set?
                .batch(batch_size=batch_size, drop_remainder=True))

            validation_ds = (validation_ds
                # .map(prepare.augment_images)
                .shuffle(buffer_size=int(validation_ds_size))
                .batch(batch_size=batch_size, drop_remainder=True))
            return train_ds, test_ds, validation_ds

    def prepareAlexNetDataset_alt(dataset,batch_size=32):
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
            .map(prepare.augment_images)
            .shuffle(buffer_size=int(train_ds_size),reshuffle_each_iteration=True)
            .batch(batch_size=batch_size, drop_remainder=True))
        test_ds = (test_ds
            .map(prepare.augment_images)
            .shuffle(buffer_size=int(test_ds_size)) ##why would you shuffle the test set?
            .batch(batch_size=batch_size, drop_remainder=True))

        validation_ds = (validation_ds
            .map(prepare.augment_images)
            .shuffle(buffer_size=int(validation_ds_size))
            .batch(batch_size=batch_size, drop_remainder=True))
        return train_ds, test_ds, validation_ds


    def prepareAlexNetDataset( dataset, batch_size =32):
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
            .map(prepare.augment_images)
            .shuffle(buffer_size=int(train_ds_size))
            # .shuffle(buffer_size=int(train_ds_size),reshuffle_each_iteration=True)
            .batch(batch_size=batch_size, drop_remainder=True))
        test_ds = (test_ds
            .map(prepare.augment_images)
            # .shuffle(buffer_size=int(train_ds_size)) ##why would you shuffle the test set?
            .batch(batch_size=batch_size, drop_remainder=True))

        validation_ds = (validation_ds
            .map(prepare.augment_images)
            # .shuffle(buffer_size=int(train_ds_size))
            .batch(batch_size=batch_size, drop_remainder=True))
        return train_ds, test_ds, validation_ds

    def prepareAlexNetDataset_old( batchsize=32):
        # tf.debugging.experimental.enable_dump_debug_info(logdir, tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1)
        (train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()

        CLASS_NAMES= ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        # validation_images, validation_labels = train_images[:5000], alt_trainLabels[:5000]
        # train_ds = tf.data.Dataset.from_tensor_slices((train_images, alt_trainLabels))
        # test_ds = tf.data.Dataset.from_tensor_slices((test_images, alt_testLabels))

        ###normal method
        validation_images, validation_labels = train_images[:5000], train_labels[:5000] #get the first 5k training samples as validation set
        train_images, train_labels = train_images[5000:], train_labels[5000:] # now remove the validation set from the training set.
        train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
        validation_ds = tf.data.Dataset.from_tensor_slices((validation_images, validation_labels))

        def augment_images(image, label):
            # Normalize images to have a mean of 0 and standard deviation of 1
            # image = tf.image.per_image_standardization(image)
            # Resize images from 32x32 to 277x277
            image = tf.image.resize(image, (227,227))
            return image, label
        

        train_ds_size = len(list(train_ds))
        test_ds_size = len(list(test_ds))
        validation_ds_size = len(list(validation_ds))

        print("trainSize {}".format(train_ds_size))
        print("testSize {}".format(test_ds_size))

        train_ds = (train_ds
                        .map(prepare.augment_images)
                        .shuffle(buffer_size=tf.cast(train_ds_size/2,'int64'))
                        .batch(batch_size=batchsize, drop_remainder=True))

        test_ds = (test_ds
                        .map(prepare.augment_images)
                        #   .shuffle(buffer_size=train_ds_size)
                        .batch(batch_size=batchsize, drop_remainder=True))

        validation_ds = (validation_ds
                        .map(prepare.augment_images)
                        #   .shuffle(buffer_size=validation_ds_size)
                        .batch(batch_size=batchsize, drop_remainder=True))

        print("testSize2 {}".format(len(list(test_ds))))
        return train_ds, test_ds, validation_ds

    def prepareInceptionDataset( dataset, batch_size=32):
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
            .map(prepare.augment_images)
            .shuffle(buffer_size=int(train_ds_size))
            # .shuffle(buffer_size=int(train_ds_size),reshuffle_each_iteration=True)
            .batch(batch_size=batch_size, drop_remainder=True))
        test_ds = (test_ds
            .map(prepare.augment_images)
            # .shuffle(buffer_size=int(train_ds_size)) ##why would you shuffle the test set?
            .batch(batch_size=batch_size, drop_remainder=True))

        validation_ds = (validation_ds
            .map(prepare.augment_images)
            # .shuffle(buffer_size=int(train_ds_size))
            .batch(batch_size=batch_size, drop_remainder=True))
        return train_ds, test_ds, validation_ds


    def datasetStats( dataset):
        (train_images, train_labels), (test_images, test_labels) = dataset
        train_ds, test_ds, validation_ds = prepare.prepareAlexNetDataset(dataset, batch_size = 1)
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