import tensorflow as tf
from tensorflow import keras
from mmd import mmd2
import time
from datetime import datetime
from tensorboard.plugins.hparams import api as hp

def normalize_01(tensor):
    min_val = tf.reduce_min(tensor)
    return tf.math.divide(tf.math.subtract(tensor, min_val), tf.math.subtract(tf.reduce_max(tensor), min_val))

def loss_fun(y_batch_train, logits_1, logits_2, batch_size, l2_logits_m1, l2_logits_m2, divergence_metric, sigma, swap_rate, lambda2, lambda3):
    
    loss_object_1 = keras.losses.CategoricalCrossentropy(from_logits=True, reduction=keras.losses.Reduction.NONE)
    loss_object_2 = keras.losses.CategoricalCrossentropy(from_logits=True, reduction=keras.losses.Reduction.NONE)
    loss_array_1 = loss_object_1(y_batch_train, logits_1)
    loss_array_2 = loss_object_2(y_batch_train, logits_2)
    
    # One hot labels
    #loss_array_1 = tf.nn.softmax_cross_entropy_with_logits(y_batch_train, logits_1)
    #loss_array_2 = tf.nn.softmax_cross_entropy_with_logits(y_batch_train, logits_2)
    
    if divergence_metric == 'mmd':
        L2 = mmd2(l2_logits_m1, l2_logits_m2, sigma)
        L2 = L2 * lambda2
        # The model does not have a softmax layer. Thus we perform it.
        # softed_logits_1_3 = tf.nn.softmax(logits_1)
        # softed_logits_2_3 = tf.nn.softmax(logits_2)
        L3 = mmd2(logits_1, logits_2, sigma)
        L3 = L3 * lambda3
    
    elif divergence_metric == 'jensen_shannon':
        kl = keras.lossses.KLDivergence()
        M2 = (0.5) * (l2_logits_m1 + l2_logits_m2)
        M3 = (0.5) * (logits_1 + logits_2)
        L2 = lambda_2 * (0.5 * kl(l2_logits_m1, M2) + 0.5 * kl(l2_logits_m2, M2))
        L3 = lambda_3 * (0.5 * kl(logits_1, M3) + 0.5 * kl(logits_2, M3))

    # Chooses the args of the (batch_size*1/4) low loss samples in the corresponding low_loss arrays
    low_loss_args_1 = tf.argsort(loss_array_1)[:int(batch_size * swap_rate)]
    low_loss_args_2 = tf.argsort(loss_array_2)[:int(batch_size * swap_rate)]
    # Gets the low_loss_samples as conducted by the peer network
    low_loss_samples_1 = tf.gather(loss_array_1, low_loss_args_2)
    low_loss_samples_2 = tf.gather(loss_array_2, low_loss_args_1)

    loss_1 = tf.nn.compute_average_loss(low_loss_samples_1, global_batch_size=int(batch_size * swap_rate))
    loss_2 = tf.nn.compute_average_loss(low_loss_samples_2, global_batch_size=int(batch_size * swap_rate))

    return loss_1+L3-L2, loss_2+L3-L2, L3, L2

#@tf.function
def run_together(model_1, model_2, train_dataset, test_dataset, val_dataset, epochs, batch_size, divergence_metric, sigma, swap_rate, lambda2, lambda3):

    if model_1 == None or model_2 == None:
        raise Exception('Models are not built properly')

    '''
    HYPERPARAMETER TUNING
    '''
    '''
    HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([2, 32]))
    HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.2))
    HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))

    METRIC_ACCURACY = 'accuracy'

    with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
          hp.hparams_config(
                  hparams=[HP_NUM_UNITS, HP_DROPOUT, HP_OPTIMIZER],
                  metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],)
    '''
    '''
    HYPERPARAMETER TUNING
    '''

    optimizer_1 = keras.optimizers.Adam(learning_rate=0.001)
    optimizer_2 = keras.optimizers.Adam(learning_rate=0.001)

    train_acc_metric_1 = keras.metrics.CategoricalAccuracy('train_acc_for_model_1')
    train_loss_metric_1 = keras.metrics.Mean('train_loss_for_model_1', dtype=tf.float32)
    
    train_acc_metric_2 = keras.metrics.CategoricalAccuracy('train_acc_for_model_2')
    train_loss_metric_2 = keras.metrics.Mean('train_loss_for_model_2', dtype=tf.float32)
    
    val_acc_metric_1 = keras.metrics.CategoricalAccuracy('val_acc_for_model_1')
    val_loss_metric_1 = keras.metrics.Mean('val_loss_for_model_1', dtype=tf.float32)
    
    val_acc_metric_2 = keras.metrics.CategoricalAccuracy('val_acc_for_model_2')
    val_loss_metric_2 = keras.metrics.Mean('val_loss_for_model_2', dtype=tf.float32)

    logdir = '../output/logs/scalars/' + datetime.now().strftime("%Y%m%d-%H%M%S")
    
    train_summary_writer_model1 = tf.summary.create_file_writer(logdir + '/model1/train')
    train_summary_writer_model2 = tf.summary.create_file_writer(logdir + '/model2/train')

    val_summary_writer_model1 = tf.summary.create_file_writer(logdir + '/model1/val')
    val_summary_writer_model2 = tf.summary.create_file_writer(logdir + '/model2/val')
    
    for epoch in range(epochs):

        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            with tf.GradientTape(persistent=True) as tape:
                
                logits_1, l2_logits_m1 = model_1(x_batch_train, training=True)
                logits_2, l2_logits_m2 = model_2(x_batch_train, training=True)
                loss_value_1, loss_value_2, L3, L2 = loss_fun(y_batch_train, logits_1, logits_2, batch_size,
                        l2_logits_m1, l2_logits_m2, divergence_metric, sigma, swap_rate, lambda2, lambda3)
            
            grads_1 = tape.gradient(loss_value_1, model_1.trainable_variables)
            grads_2 = tape.gradient(loss_value_2, model_2.trainable_variables)
            
            optimizer_1.apply_gradients(zip(grads_1, model_1.trainable_variables))
            optimizer_2.apply_gradients(zip(grads_2, model_2.trainable_variables))    

        # Update metrics for visualization
        train_loss_metric_1.update_state(loss_value_1)
        train_loss_metric_2.update_state(loss_value_2)
        train_acc_metric_1.update_state(y_batch_train, logits_1)
        train_acc_metric_2.update_state(y_batch_train, logits_2)
        
        with train_summary_writer_model1.as_default():
            tf.summary.scalar('loss', train_loss_metric_1.result(), step=epoch)
            tf.summary.scalar('accuracy', train_acc_metric_1.result(), step=epoch)
            tf.summary.scalar('L2', float(L2), step=epoch)
            tf.summary.scalar('L3', float(L3), step=epoch)
        with train_summary_writer_model2.as_default():
            tf.summary.scalar('loss', train_loss_metric_2.result(), step=epoch)
            tf.summary.scalar('accuracy', train_acc_metric_2.result(), step=epoch)
            tf.summary.scalar('L2', float(L2), step=epoch)
            tf.summary.scalar('L3', float(L3), step=epoch)
       
        print(f'Training acc 1 over epoch: {train_acc_metric_1.result()}')
        print(f'Training acc 2 over epoch: {train_acc_metric_2.result()}')
        print(f"L2 to be maximized: {L2}")
        print(f"L3 to be minimized: {L3}")
        
        train_acc_metric_1.reset_states()
        train_acc_metric_2.reset_states()
        train_loss_metric_1.reset_states()
        train_loss_metric_2.reset_states()

        for x_batch_val, y_batch_val in val_dataset:
            val_logits_1, _ = model_1(x_batch_val, training=False)
            val_logits_2, _ = model_2(x_batch_val, training=False)

        # Update metrics for visualization
        val_acc_metric_1.update_state(y_batch_val, val_logits_1)
        val_acc_metric_2.update_state(y_batch_val, val_logits_2)

        with train_summary_writer_model1.as_default():
            # tf.summary.scalar('loss', val_loss_metric_1.result(), step=epoch)
            tf.summary.scalar('accuracy', val_acc_metric_1.result(), step=epoch)
        with train_summary_writer_model2.as_default():
            # tf.summary.scalar('loss', val_loss_metric_2.result(), step=epoch)
            tf.summary.scalar('accuracy', val_acc_metric_2.result(), step=epoch)

        print(f'Validation acc for model 1: {val_acc_metric_1.result()}')
        print(f'Validation acc for model 2: {val_acc_metric_2.result()}')
        val_acc_metric_1.reset_states()
        val_acc_metric_2.reset_states()
        # val_loss_metric_1.reset_states()
        # val_loss_metric_2.reset_states()

