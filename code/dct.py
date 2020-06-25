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
    ap.add_argument('-f', '--framework', default='co', help='collaborative training or training with single model: co and single')
    ap.add_argument('-a', '--architecture', default='paper_model', help='paper_model or keras_model')
    ap.add_argument('-d', '--dataset', default='cifar10', help='cifar10, cifar100_fine, cifar100_coarse or mnist')
    args = vars(ap.parse_args())

    return args

def prep_data(dataset):
    '''
    Creates numpy arrays that are ready to be fed into model
    '''

    global NUM_OF_CLASSES
    
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

    if args['framework'] not in ('co', 'single'): 
        raise ValueError('Enter 1 for a single model, enter 2 for collaborative model')

    # Load the given dataset by user
    if args['dataset'] in ('cifar10', 'cifar100_fine', 'cifar100_coarse', 'mnist'):  
        x_train, x_test, y_train, y_test = prep_data(args['dataset'])
    else:
        raise ValueError('Argument Error: Legal arguments are cifar10, cifar100_fine, cifar100_fine, mnist')
    
    # Set model1 parameters
    model1 = Model('model1', NUM_OF_CLASSES, args['batch_size'], args['epochs'])
    
    # Create validation arrays and by-custom-training-loop-consumable datasets
    train_dataset, test_dataset, val_dataset = model1.useTfData(x_train, x_test, y_train, y_test)
    
    # model1 is going to be created no matter what, as long as the argument is right
    if args['architecture'] == 'paper_model':
        model1.build_paper_model(x_train.shape[1:])
    elif args['architecture'] == 'keras_model':
        model1.build_keras_model(x_train.shape[1:])
    else:
        raise ValueError('Argument Error: Legal arguments are paper_model and keras_model') 
    
    # If single model framework is chosen, runs the models own custom training loop
    # If co model framework is chosen, creates a second model and runs them simultaneously
    if args['framework'] == 'single':
        model1.cust_training_loop(train_dataset, test_dataset, val_dataset)
    elif args['framework'] == 'co':

        # Set model2 parameters
        model2 = Model('model2', NUM_OF_CLASSES, args['batch_size'], args['epochs'])

        if args['architecture'] == 'paper_model':
            model2.build_paper_model(x_train.shape[1:])
        elif args['architecture'] == 'keras_model':
            model2.build_keras_model(x_train.shape[1:])
        else:
            raise ValueError('Argument Error: Did you mean keras_model?') 
        
        run_together(model1.model, model2.model, train_dataset, test_dataset, val_dataset, args['epochs'], args['batch_size'])

    # Model 1 summary
    model_sum_1 = model1.model.summary()
    print(f'model1 summary: {model_sum_1}')
    #keras.utils.plot_model(model1.model, 'model1.png', show_shapes=True)
    try:
        model1.model.predict(x_test)
    except:
        print("model.predict gave an error")
    model1.saveModel()  
    
    # Model 2 summary
    if args['framework'] == 'co':
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
