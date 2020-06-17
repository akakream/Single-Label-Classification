import tensorflow as tf
from tensorflow import keras
from keras.datasets import cifar10

def prep_data():

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255.0
    x_test /= 255.0

    return x_train, y_train, x_test, y_test

def useTfData(x_train, y_train, x_test, y_test, batch_size):
    
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))                        
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))                           
    
    SHUFFLE_BUFFER_SIZE = 100                                                                     
    
    train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(batch_size)             
    test_dataset = test_dataset.batch(batch_size)

    print(type(train_dataset))
    print(type(test_dataset))
    
    return train_dataset, test_dataset

a,b,c,d = prep_data()
e,f = useTfData(a,b,c,d,32)
