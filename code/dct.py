import tensorflow as tf
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from model import Model
import pickle
import datetime
import argparse
from tensorflow import keras
from keras.datasets import cifar10
from keras.datasets import cifar100
from keras.datasets import mnist
from run_together import run_together

NUM_OF_CLASSES = None

def add_arguments():
    ap = argparse.ArgumentParser(prog='discrepant collaborative training', 
            description='This is a modified implementation of th paper Learning from Noisy Labels via Discrepant Collaborative Training', 
            epilog='-- Float like a butterfly, sting like a bee --')
    ap.add_argument('-b', '--batch_size', type=int, default=32, help='Batch size, default is 32')
    ap.add_argument('-e', '--epochs', type=int, default=10, help='Number of epochs, default is 10')
    ap.add_argument('-m', '--models', type=int, default=2, help='Number of models to run, default is 2')
    ap.add_argument('-d', '--dataset', default=cifar10, help='cifar10 or cifar100')
    args = vars(ap.parse_args())

    return args

def plot():
    print('--plot--')

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict        

def prep_data(dataset):
    '''
    Creates numpy arrays that are ready to be fed into model
    '''

    global NUM_OF_CLASSES
    
    # The data, split between train and test sets:
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
    #print('x_train shape:', x_train.shape)
    #print(x_train.shape[0], 'train samples')
    #print(x_test.shape[0], 'test samples')
    #print(x_train.shape[1:])

    '''
    For now the SparseCategoricalCrossEntropy is used which takes integer inputs
    If you want to use CategoricalCrossEntropy which takes one-hot vectors, uncomment below
    '''
    # Convert class vectors to binary class matrices.
    #y_train = keras.utils.to_categorical(y_train, 10)
    #y_test = keras.utils.to_categorical(y_test, 10)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255.0
    x_test /= 255.0

    return x_train, x_test, y_train, y_test

def main(args):

    if args['models'] != 1 and args['models'] != 2: 
        print('Enter 1 for a single model, enter 2 for collaborative model')
        exit()

    x_train, x_test, y_train, y_test = prep_data(args['dataset'])
    
    model1 = Model('model1', NUM_OF_CLASSES, args['batch_size'], args['epochs'])
    
    train_dataset, test_dataset, val_dataset = model1.useTfData(x_train, x_test, y_train, y_test)

    model1.buildModel(x_train.shape[1:])

    if args['models'] == 1:
        model1.cust_training_loop(train_dataset, test_dataset, val_dataset)
        #model2.cust_training_loop(train_dataset, test_dataset, val_dataset)
    elif args['models'] == 2:
        model2 = Model('model2', NUM_OF_CLASSES, args['batch_size'], args['epochs'])
        model2.buildModel(x_train.shape[1:])
        run_together(model1.model, model2.model, train_dataset, test_dataset, val_dataset, args['epochs'], args['batch_size'])

    model_sum_1 = model1.model.summary()
    print(f'model1 summary: {model_sum_1}')
    #keras.utils.plot_model(model1.model, 'model1.png', show_shapes=True)
    try:
        model1.model.predict(x_test)
    except:
        print("model.predict gave an error")
    model1.saveModel()  
    
    if args['models'] == 2:
        model_sum_2 = model2.model.summary()
        print(f'model2 summary: {model_sum_2}')
        #keras.utils.plot_model(model2.model, 'model2.png', show_shapes=True)
        try:
            model2.model.predict(x_test)
        except:
            print("model.predict gave an error")
        model2.saveModel()  

if __name__ == '__main__':
    args = add_arguments()
    main(args)
