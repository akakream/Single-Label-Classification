import tensorflow as tf
from tensorflow import keras
from keras.datasets import cifar10
from keras.datasets import cifar100
from keras.datasets import mnist
from noisifier import Noisifier

def prep_data(dataset, batch_size, noise_type, noise_rate):
    '''
    Creates numpy arrays that are ready to be fed into model
    '''

    NUM_OF_CLASSES = None
    
    if dataset == 'cifar10':
        '''
        32x32
        50k train
        10k test
        '''
        NUM_OF_CLASSES = 10
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    elif dataset == 'cifar100_fine':
        '''
        32x32
        50k train
        10k test
        '''
        NUM_OF_CLASSES = 100
        (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
    
    elif dataset == 'cifar100_coarse': 
        '''
        32x32
        50k train
        10k test
        '''
        NUM_OF_CLASSES = 20
        (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='coarse')
    
    elif dataset == 'mnist': 
        '''
        28x28
        60k train
        10k test
        '''
        NUM_OF_CLASSES = 10
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255.0
    x_test /= 255.0

    # Uncomment this part when the tests are done, otherwise takes too long
    '''
    x_val = x_train[-10000:]
    y_val = y_train[-10000:]
    x_train = x_train[:-10000]
    y_train = y_train[:-10000]
    '''

    x_val = x_train[-10:]
    y_val = y_train[-10:]
    x_train = x_train[:20]
    y_train = y_train[:20]
    
    if noise_type != 'none':

        # Add noise to the training data. The last zero is the random seed
        noisifier = Noisifier()
        noised_y_train = noisifier.noisify(y_train, noise_type, noise_rate, NUM_OF_CLASSES, 0)
        y_train = noised_y_train

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    SHUFFLE_BUFFER_SIZE = 100
    train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(batch_size)
    
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_dataset = val_dataset.batch(batch_size)
                                                                                                                                                                    
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.batch(batch_size)
    
    return train_dataset, test_dataset, val_dataset, x_train, x_test, y_train, y_test, NUM_OF_CLASSES
