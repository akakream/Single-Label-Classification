import tensorflow as tf    
import os
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, AvgPool3D, BatchNormalization, Activation
import keras.backend as K
import numpy as np

class Model:
   
    save_dir = f'{os.getcwd()}/../saved_models'

    def __init__(self, name, classes, batch_size, epochs):
        self.model_name = f'dct_cifar10_trained_{name}.h5'
        self.classes = classes
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None

    def __repr__(self):
        return f'Model(self.model_name)'

    def __str__(self):
        return self.model_name

    def custom_loss(self, y_batch_train, logits):

        #loss_val = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_batch_train, logits=logits)
        loss_object = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=keras.losses.Reduction.NONE)
        loss_array = loss_object(y_batch_train, logits)
        
        print(f'This is loss_array: {loss_array}')
            
        #Select the 8 examples within the minibatch that produces the lowest losses
        low_loss_samples = tf.sort(loss_array)[:8]
        print(f'This is low_loss_samples : {low_loss_samples}')

        return tf.nn.compute_average_loss(loss_array, global_batch_size=self.batch_size)

    # This is called L1 in the paper, aka Classification Loss
    def buildClassLoss(self):
        loss = 0
        print('Classification Loss')
        return loss    

    # This is called L2 in the paper, aka Discrepancy Loss 
    def buildDiscLoss(self):
        loss = 0
        print('Discrepancy Loss')
        return loss    

    # This is called L3 in the paper, aka Consistency Loss 
    def buildConsLoss(self):
        loss = 0
        print('Consistency Loss')
        return loss

    # Loss = L1 + (lambda3)*L3 - (lambda2)*L2  ---> Do this for both F and G, which are the two complementary networks
    def buildFinalLoss(self, L1, L2, L3):
        print('Loss = L1 + (lambda3)*L3 - (lambda2)*L2')

    def useTfData(self, x_train, x_test, y_train, y_test):

        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

        SHUFFLE_BUFFER_SIZE = 100

        train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(self.batch_size)
        test_dataset = test_dataset.batch(self.batch_size)

        return train_dataset, test_dataset

    def cust_training_loop(self, train_dataset, test_dataset):
        
        if self.model == None:
            raise Exception('The model is not built, call buildModel() method first')
            exit()

        optimizer = keras.optimizers.Adam()
        train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
        for epoch in range(self.epochs):
            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    logits = self.model(x_batch_train, training=True)
                    loss_value = self.custom_loss(y_batch_train, logits)
                grads = tape.gradient(loss_value, self.model.trainable_weights)
                optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
                
                # Log loss
                train_acc_metric(y_batch_train, logits)
                if step % 200 == 0:
                    print('Training loss (for one batch) at step %s: %s' % (step, float(loss_value)))
                    print('Seen so far: %s samples' % ((step + 1) * 64))
            
            # Display metrics at the end of each epoch.
            train_acc = train_acc_metric.result()
            print('Training acc over epoch: %s' % (float(train_acc),))
            # Reset training metrics at the end of each epoch
            train_acc_metric.reset_states()

    def buildModel(self, inputShape):

        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=inputShape))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        #TODO: Add here the L2 layer
        self.buildDiscLoss()

        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.classes))
        #model.add(Activation('softmax'))

        # TODO: Add here the L3 layer
        self.buildConsLoss()

        self.model = model

    def eval(self, X_TEST, Y_TEST):
        
        if self.model == None:
            raise Exception('The model is not built, call buildModel() method first')
            exit()
        
        val_loss, val_acc = self.model.evaluate(X_TEST, Y_TEST, batch_size=self.batch_size, verbose=1)
        print(f'val_loss is {val_loss} and val_acc is {val_acc}')    

    def saveModel(self):
        
        if self.model == None:
            raise Exception('The model is not built, call buildModel() method first')
            exit()

        # Save model and weights
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)
        model_path = os.path.join(self.save_dir, self.model_name)
        self.model.save(model_path)
        print('Saved trained model at %s ' % model_path)
