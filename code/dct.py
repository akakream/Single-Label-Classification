import tensorflow as tf
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import model
import pickle
import datetime
import argparse
import keras
from keras.datasets import cifar10

def add_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--data_destination', required=True, help='relative destination to the folder where pickle files are located')
    ap.add_argument('-c', '--classes', required=True, help='Number of classes. This is going to be added to the last layer of the model')
    ap.add_argument('-b', '--batch_size', default=64, help='Batch size, default is 64')
    ap.add_argument('-e', '--epochs', default=10, help='Number of epochs, default is 10')
    args = vars(ap.parse_args())

    return args

def plot():
    print('--plot--')

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict        

def prep_data():
    '''
    Creates numpy arrays that are ready to be fed into model
    '''
    # The data, split between train and test sets:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    print(x_train.shape[1:])

    '''
    For now the SparseCategoricalCrossEntropy is used which takes integer inputs
    If you want to use CategoricalCrossENtropy which tajes one-hot vectors, uncomment below
    '''
    # Convert class vectors to binary class matrices.
    #y_train = keras.utils.to_categorical(y_train, 10)
    #y_test = keras.utils.to_categorical(y_test, 10)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    return x_train, x_test, y_train, y_test

# data_destination, shape, classes, batch_size, epochs 
def main(args):
    
    x_train, x_test, y_train, y_test = prep_data()

    train_dataset, test_dataset = model.useTfData(x_train, x_test, y_train, y_test, int(args['batch_size']))

    dct_model = model.buildModel(int(args['classes']), x_train.shape[1:])
    
    model.cust_training_loop(train_dataset, test_dataset, dct_model, int(args['epochs']), int(args['batch_size']))

    model_sum = dct_model.summary()
    print(f'model summary: {model_sum}')
    # model.eval(dct_model, X_TEST, Y_TEST, int(args['batch_size']))
    try:
        model.predict(dct_model, x_test)
    except:
        print("model.predict gave an error")
    model.saveModel(dct_model)  
    
if __name__ == '__main__':
    args = add_arguments()
    main(args)
