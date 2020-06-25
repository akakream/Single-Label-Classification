import os
import numpy as np
import tensorflow as tf    
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization, Activation, LeakyReLU
import tensorflow.keras.backend as K
from mmd import mmd
import time

class Model:
   
    save_dir = f'{os.getcwd()}/../saved_models'

    def __init__(self, name, classes, batch_size, epochs):
        self.model_name = f'dct_trained_{name}.h5'
        self.classes = classes
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None

    def __repr__(self):
        return f'Model(self.model_name)'

    def __str__(self):
        return self.model_name

    def custom_loss(self, y_batch_train, logits):

        loss_object = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=keras.losses.Reduction.NONE)
        loss_array = loss_object(y_batch_train, logits)
           
        low_loss_args = tf.argsort(loss_array)[:int(self.batch_size*3/4)]

        low_loss_samples = tf.gather(loss_array, low_loss_args)

        return tf.nn.compute_average_loss(low_loss_samples, global_batch_size=int(self.batch_size*3/4))
    
    # FIXME: although x_train and y_train are not shuffled, val sets created 
    def useTfData(self, x_train, x_test, y_train, y_test):

        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        SHUFFLE_BUFFER_SIZE = 100
        train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(self.batch_size)
        
        x_val = x_train[-10000:]
        y_val = y_train[-10000:]
        x_train = x_train[:-10000]
        y_train = y_train[:-10000]

        val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        val_dataset = val_dataset.batch(self.batch_size)

        test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        test_dataset = test_dataset.batch(self.batch_size)

        return train_dataset, test_dataset, val_dataset

    def cust_training_loop(self, train_dataset, test_dataset, val_dataset):
        
        if self.model == None:
            raise Exception('The model is not built, call buildModel() method first')
            exit()

        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
        val_acc_metric = keras.metrics.SparseCategoricalAccuracy()

        for epoch in range(self.epochs):

            print(f'Start of epoch {epoch}')
            start_time = time.time()

            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    
                    logits, _ = self.model(x_batch_train, training=True)
                    loss_value = self.custom_loss(y_batch_train, logits)
                
                grads = tape.gradient(loss_value, self.model.trainable_weights)
                optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
                
                train_acc_metric.update_state(y_batch_train, logits)

                if step % 200 == 0:
                    print(f"Training loss (for one batch) at step {step}: {float(loss_value)}")  
                    print(f"Seen so far: {(step+1)*self.batch_size} samples")
            
            # Display metrics at the end of each epoch.
            train_acc = train_acc_metric.result()
            print(f'Training acc 1 over epoch: {float(train_acc)}')
            train_acc_metric.reset_states()

            for x_batch_val, y_batch_val in val_dataset:
                val_logits, _ = self.model(x_batch_val, training=False)
                val_acc_metric.update_state(y_batch_val, val_logits)

            val_acc = val_acc_metric.result()
            print(f'Validation acc: {float(val_acc)}')
            print(f'Time taken: {time.time() - start_time}') 

    def build_paper_model(self, inputShape):

        inputs = keras.Input(shape=inputShape)
        x = Conv2D(128, (3,3), padding='same')(inputs)
        x = LeakyReLU(alpha=0.01)(x)
        x = Conv2D(128, (3,3), padding='same')(x)
        x = LeakyReLU(alpha=0.01)(x)
        x = Conv2D(128, (3,3), padding='same')(x)
        x = LeakyReLU(alpha=0.01)(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(x)
        x = Dropout(0.25)(x)

        x = Conv2D(256, (3,3), padding='same')(x)
        x = LeakyReLU(alpha=0.01)(x)
        x = Conv2D(256, (3,3), padding='same', name='l2-layer')(x)
        x = LeakyReLU(alpha=0.01)(x)
        l2_logits = x
        x = Conv2D(256, (3,3), padding='same')(x)
        x = LeakyReLU(alpha=0.01)(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(x)
        x = Dropout(0.25)(x)

        x = Conv2D(512, (3,3), padding='same')(x)
        x = LeakyReLU(alpha=0.01)(x)
        x = Conv2D(256, (3,3), padding='same')(x)
        x = LeakyReLU(alpha=0.01)(x)
        x = Conv2D(128, (3,3), padding='same')(x)
        x = LeakyReLU(alpha=0.01)(x)
        x = AveragePooling2D(pool_size=(2,2), strides=(2,2))(x)
        
        x = Flatten()(x)
        x = Dense(128)(x)
        outputs = Dense(self.classes)(x)
        # There is no softmax activation layer because it is applied by SparseCategoricalCrossentropy in training loop

        model = keras.Model(inputs=inputs, outputs=[outputs, l2_logits], name='dct_model')
    
        self.model = model

    def build_keras_model(self, inputShape):

        inputs = keras.Input(shape=inputShape)
        x = Conv2D(32, (3,3), padding='same', activation='relu')(inputs)
        x = Conv2D(32, (3,3), padding='same', activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(x)
        x = Dropout(0.25)(x)

        x = Conv2D(64, (3,3), padding='same', activation='relu')(x)
        x = Conv2D(64, (3,3), padding='same', activation='relu', name='l2-layer')(x)
        l2_logits = x
        x = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(x)
        x = Dropout(0.25)(x)
        
        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        outputs = Dense(self.classes)(x)
        # There is no softmax activation layer because it is applied by SparseCategoricalCrossentropy in training loop

        model = keras.Model(inputs=inputs, outputs=[outputs, l2_logits], name='dct_model')
    
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
