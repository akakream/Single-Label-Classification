import tensorflow as tf
from tensorflow import keras
from mmd import mmd

def loss_fun(y_batch_train, logits_1, logits_2, batch_size):
    
    loss_object_1 = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=keras.losses.Reduction.NONE)
    loss_object_2 = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=keras.losses.Reduction.NONE)
    loss_array_1 = loss_object_1(y_batch_train, logits_1)
    loss_array_2 = loss_object_2(y_batch_train, logits_2)
    
    lamb_3 = 1
    L3 = mmd(logits_1, logits_2) * lamb_3

    try:
        print(f'L3: {L3}')
    except:
        print('L3 could not be printed')

    # batch_size/4 lowest loss samples are being used
    low_loss_samples_1 = tf.sort(loss_array_1)[:int(batch_size/4)]
    low_loss_samples_2 = tf.sort(loss_array_2)[:int(batch_size/4)]

    #TODO: Low loss samples must be exchanged
    loss_1 = tf.nn.compute_average_loss(low_loss_samples_1, global_batch_size=int(batch_size/4))
    loss_2 = tf.nn.compute_average_loss(low_loss_samples_2, global_batch_size=int(batch_size/4))

    return loss_1+L3, loss_2+L3

def run_together(model_1, model_2, train_dataset, test_dataset, epochs, batch_size):

    if model_1 == None or model_2 == None:
        raise Exception('Models are not built properly')
        exit()

    optimizer_1 = keras.optimizers.Adam()
    optimizer_2 = keras.optimizers.Adam()
    train_acc_metric_1 = keras.metrics.SparseCategoricalAccuracy()
    train_acc_metric_2 = keras.metrics.SparseCategoricalAccuracy()
    
    for epoch in range(epochs):
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            with tf.GradientTape(persistent=True) as tape:

                #TODO: Get L2 here and give it as input to the loss_fun 
                
                logits_1 = model_1(x_batch_train, training=True)
                logits_2 = model_2(x_batch_train, training=True)
                loss_value_1, loss_value_2 = loss_fun(y_batch_train, logits_1, logits_2, batch_size)
            
            grads_1 = tape.gradient(loss_value_1, model_1.trainable_weights)
            grads_2 = tape.gradient(loss_value_2, model_2.trainable_weights)
            
            optimizer_1.apply_gradients(zip(grads_1, model_1.trainable_weights))
            optimizer_2.apply_gradients(zip(grads_2, model_2.trainable_weights))

            train_acc_metric_1(y_batch_train, logits_1)
            train_acc_metric_2(y_batch_train, logits_2)
            if step % 200 == 0:
                print("Training loss (for one batch) at step %d: %.4f" % (step, float(loss_value_1))                                       )
                print("Seen so far: %s samples" % ((step + 1) * 64))
                print("Training loss (for one batch) at step %d: %.4f" % (step, float(loss_value_2))                                       )
                print("Seen so far: %s samples" % ((step + 1) * 64))
                                               
        train_acc_1 = train_acc_metric_1.result()
        train_acc_2 = train_acc_metric_2.result()
        print('Training acc 1 over epoch: %s' % (float(train_acc_1),))
        print('Training acc 2 over epoch: %s' % (float(train_acc_2),))
        train_acc_metric_1.reset_states()
        train_acc_metric_2.reset_states()





