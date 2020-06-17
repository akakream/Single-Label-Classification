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
from run_together import run_together

def add_arguments():
    #TODO: Integer arguments are casted in the program, fix it
    ap = argparse.ArgumentParser(prog='discrepant collaborative training', 
            description='This is a modified implementation of th paper Learning from Noisy Labels via Discrepant Collaborative Training', 
            epilog='-- Float like a butterfly, sting like a bee --')
    ap.add_argument('-c', '--classes', required=True, help='Number of classes. This is going to be added to the last layer of the model')
    ap.add_argument('-b', '--batch_size', default=64, help='Batch size, default is 64')
    ap.add_argument('-e', '--epochs', default=10, help='Number of epochs, default is 10')
    ap.add_argument('-m', '--models', default=2, help='Number of models to run, default is 2')
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
    #print('x_train shape:', x_train.shape)
    #print(x_train.shape[0], 'train samples')
    #print(x_test.shape[0], 'test samples')
    #print(x_train.shape[1:])

    '''
    For now the SparseCategoricalCrossEntropy is used which takes integer inputs
    If you want to use CategoricalCrossENtropy which tajes one-hot vectors, uncomment below
    '''
    # Convert class vectors to binary class matrices.
    #y_train = keras.utils.to_categorical(y_train, 10)
    #y_test = keras.utils.to_categorical(y_test, 10)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255.0
    x_test /= 255.0

    return x_train, x_test, y_train, y_test

# classes, batch_size, epochs, models 
def main(args):

    if int(args['models']) != 1 and int(args['models']) != 2: 
        print('Enter 1 for a single model, enter 2 for collaborative model')
        exit()

    x_train, x_test, y_train, y_test = prep_data()
    
    model1 = Model('model1', int(args['classes']), int(args['batch_size']), int(args['epochs']))
    
    train_dataset, test_dataset = model1.useTfData(x_train, x_test, y_train, y_test)

    model1.buildModel(x_train.shape[1:])

    if int(args['models']) == 1:
        model1.cust_training_loop(train_dataset, test_dataset)
        #model2.cust_training_loop(train_dataset, test_dataset)
    elif int(args['models']) == 2:
        model2 = Model('model2', int(args['classes']), int(args['batch_size']), int(args['epochs']))
        model2.buildModel(x_train.shape[1:])
        run_together(model1.model, model2.model, train_dataset, test_dataset, int(args['epochs']), int(args['batch_size']))

    model_sum_1 = model1.model.summary()
    print(f'model1 summary: {model_sum_1}')
    #keras.utils.plot_model(model1.model, 'model1.png', show_shapes=True)
    # model1.eval(X_TEST, Y_TEST)
    try:
        model1.model.predict(x_test)
    except:
        print("model.predict gave an error")
    model1.saveModel()  
    
    if int(args['models']) == 2:
        model_sum_2 = model2.model.summary()
        print(f'model2 summary: {model_sum_2}')
        #keras.utils.plot_model(model2.model, 'model2.png', show_shapes=True)
        # model2.eval(X_TEST, Y_TEST)
        try:
            model2.model.predict(x_test)
        except:
            print("model.predict gave an error")
        model2.saveModel()  

if __name__ == '__main__':
    args = add_arguments()
    main(args)
