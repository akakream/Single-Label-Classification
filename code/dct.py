import tensorflow as tf
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from model import Model
import pickle
from datetime import datetime
import argparse
from tensorflow import keras
from run_together import run_together
from prep_data import prep_data


def add_arguments():
    ap = argparse.ArgumentParser(prog='discrepant collaborative training', 
            description='This is a modified implementation of th paper Learning from Noisy Labels via Discrepant Collaborative Training', 
            epilog='-- Float like a butterfly, sting like a bee --')
    ap.add_argument('-b', '--batch_size', type=int, default=32, help='Batch size, default is 32')
    ap.add_argument('-e', '--epochs', type=int, default=10, help='Number of epochs, default is 10')
    ap.add_argument('-f', '--framework', default='co', help='collaborative training or training with single model: co and single')
    ap.add_argument('-a', '--architecture', default='paper_model', help='paper_model or keras_model')
    ap.add_argument('-d', '--dataset', default='cifar10', help='cifar10, cifar100_fine, cifar100_coarse or mnist')
    ap.add_argument('-nt', '--noise_type', help='type of label noise to be added: symmetry, flip or none')
    ap.add_argument('-nr', '--noise_rate', type=float, help='rate of label noise to be added, use float: between 0. and 1.')
    ap.add_argument('-dm', '--divergence_metric', help='Divergence metric to be used to diverge and converge the predictions of the models. Options: "mmd" or "jensen_shannon".')
    ap.add_argument('-si', '--sigma', type=float, help='The value of the sigma for the gaussian kernel.')
    ap.add_argument('-sr', '--swap_rate', type=float, help='The percentage of the swap between the two models.')
    ap.add_argument('-lto', '--lambda_two', type=float, help='Lambda two for the L2.')
    ap.add_argument('-ltr', '--lambda_three', type=float, help='Lambda three for the L3.')
    args = vars(ap.parse_args())

    return args

def main(args):
    
    if args['framework'] not in ('co', 'single'): 
        raise ValueError('Enter 1 for a single model, enter 2 for collaborative model')

    # Load the given dataset by user
    if args['dataset'] in ('cifar10', 'cifar100_fine', 'cifar100_coarse', 'mnist'):  
        train_dataset, test_dataset, val_dataset, x_train, x_test, y_train, y_test, NUM_OF_CLASSES = prep_data(args['dataset'], args['batch_size'], args['noise_type'], args['noise_rate'])
    else:
        raise ValueError('Argument Error: Legal arguments are cifar10, cifar100_fine, cifar100_fine, mnist')
    
    # Set model1 parameters
    model1 = Model('model1', NUM_OF_CLASSES, args['batch_size'], args['epochs'])
    
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
        
        run_together(model1.model, model2.model, train_dataset, test_dataset, val_dataset, args['epochs'], args['batch_size'], args['divergence_metric'], args['sigma'], args['swap_rate'], args['lambda_two'], args['lambda_three'])

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
