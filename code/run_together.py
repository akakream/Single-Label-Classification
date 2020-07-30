import tensorflow as tf
from tensorflow import keras
from mmd import mmd2
import time
from datetime import datetime

def loss_fun(y_batch_train, logits_1, logits_2, batch_size, l2_logits_m1, l2_logits_m2):
    
    loss_object_1 = keras.losses.CategoricalCrossentropy(from_logits=True, reduction=keras.losses.Reduction.NONE)
    loss_object_2 = keras.losses.CategoricalCrossentropy(from_logits=True, reduction=keras.losses.Reduction.NONE)
    loss_array_1 = loss_object_1(y_batch_train, logits_1)
    loss_array_2 = loss_object_2(y_batch_train, logits_2)
    
    # One hot labels
    #loss_array_1 = tf.nn.softmax_cross_entropy_with_logits(y_batch_train, logits_1)
    #loss_array_2 = tf.nn.softmax_cross_entropy_with_logits(y_batch_train, logits_2)
    
    lamb_2 = 1
    L2 = mmd2(l2_logits_m1, l2_logits_m2) * lamb_2 

    # The model does not have a softmax layer. Thus we perform it.
    lamb_3 = 1
    #softed_logits_1 = tf.nn.softmax(logits_1)
    #softed_logits_2 = tf.nn.softmax(logits_2)
    L3 = mmd2(logits_1, logits_2) * lamb_3
    
    # Chooses the args of the (batch_size*1/4) low loss samples in the corresponding low_loss arrays
    low_loss_args_1 = tf.argsort(loss_array_1)[:int(batch_size*1/4)]
    low_loss_args_2 = tf.argsort(loss_array_2)[:int(batch_size*1/4)]
    # Gets the low_loss_samples as conducted by the peer network
    low_loss_samples_1 = tf.gather(loss_array_1, low_loss_args_2)
    low_loss_samples_2 = tf.gather(loss_array_2, low_loss_args_1)

    loss_1 = tf.nn.compute_average_loss(low_loss_samples_1, global_batch_size=int(batch_size*1/4))
    loss_2 = tf.nn.compute_average_loss(low_loss_samples_2, global_batch_size=int(batch_size*1/4))

    return loss_1+L3-L2, loss_2+L3-L2, L3, L2

# @tf.function
def run_together(model_1, model_2, train_dataset, test_dataset, val_dataset, epochs, batch_size):

    if model_1 == None or model_2 == None:
        raise Exception('Models are not built properly')

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

    # TODO: NOT SURE IF THIS IS THE RIGHT PLACE TO PUT THIS, FIX LATER
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
                        l2_logits_m1, l2_logits_m2)
            
            grads_1 = tape.gradient(loss_value_1, model_1.trainable_weights)
            grads_2 = tape.gradient(loss_value_2, model_2.trainable_weights)
            
            optimizer_1.apply_gradients(zip(grads_1, model_1.trainable_weights))
            optimizer_2.apply_gradients(zip(grads_2, model_2.trainable_weights))
            
            # Update metrics for visualization
            train_loss_metric_1.update_state(loss_value_1)
            train_loss_metric_2.update_state(loss_value_2)
            train_acc_metric_1.update_state(y_batch_train, logits_1)
            train_acc_metric_2.update_state(y_batch_train, logits_2)

            if step % 200 == 0:
                print(f"Training loss (for one batch) at step {step}: {train_loss_metric_1.result()}")
                print(f"Seen so far: {(step+1)*batch_size} samples")
                print(f"Training loss (for one batch) at step {step}: {train_loss_metric_1.result()}")
                print(f"Seen so far: {(step+1)*batch_size} samples")
                print(f"L2 to be maximized: {L2}")
                print(f"L3 to be minimized: {L3}")
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
        print(f'Validation acc for model 1: {val_acc_metric_2.result()}')
        val_acc_metric_1.reset_states()
        val_acc_metric_2.reset_states()
        # val_loss_metric_1.reset_states()
        # val_loss_metric_2.reset_states()

