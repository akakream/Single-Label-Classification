import tensorflow as tf    
import os
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, AvgPool3D, BatchNormalization, Activation
import keras.backend as K

save_dir = f'{os.getcwd()}/../saved_models'
model_name = 'dct_cifar10_trained_model.h5'   

# TODO: Build a second model to collaborate

# TODO: Choose a selection strategy and apply

def custom_loss(y_batch_train, logits):
    '''
    the input dimensions are (batch_size, num_of_classes)
    the loss must be a vector of length 32
    '''
    
    #loss_val = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_batch_train, logits=logits)
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)  
    loss_val = loss_fn(y_batch_train, logits)

    '''
    print(loss_val.shape)
    Select here the low loss samples!!!!!!!
    '''

    return loss_val

# This is called L1 in the paper, aka Classification Loss
def buildClassLoss():
    loss = 0
    print('Classification Loss')
    return loss    

# This is called L2 in the paper, aka Discrepancy Loss 
def buildDiscLoss():
    loss = 0
    print('Discrepancy Loss')
    return loss    

# This is called L3 in the paper, aka Consistency Loss 
def buildConsLoss():
    loss = 0
    print('Consistency Loss')
    return loss

# Loss = L1 + (lambda3)*L3 - (lambda2)*L2  ---> Do this for both F and G, which are the two complementary networks
def buildFinalLoss(L1, L2, L3):
    print('Loss = L1 + (lambda3)*L3 - (lambda2)*L2')

def useTfData(x_train, x_test, y_train, y_test, batch_size):

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    SHUFFLE_BUFFER_SIZE = 100

    train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(batch_size)
    test_dataset = test_dataset.batch(batch_size)

    return train_dataset, test_dataset

def cust_training_loop(train_dataset, test_dataset, model, epochs):
    optimizer = keras.optimizers.Adam()
    train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
    for epoch in range(epochs):
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                logits = model(x_batch_train, training=True)
                loss_value = custom_loss(y_batch_train, logits)
            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            
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

def buildModel(num_of_classes, inputShape):

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
    buildDiscLoss()

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_of_classes))
    #model.add(layers.Activation('softmax'))

    # TODO: Add here the L3 layer
    buildConsLoss()

    return model

def eval(model, X_TEST, Y_TEST, batch_size):
    val_loss, val_acc = model.evaluate(X_TEST, Y_TEST, batch_size=batch_size, verbose=1)
    print(f'val_loss is {val_loss} and val_acc is {val_acc}')    

def saveModel(model):
    # Save model and weights
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)
