import tensorflow as tf
from functools import reduce

def partial_mmd(X,Y):
    '''
    Pairwise dsitance
    uses gaussian kernel
    input: tensor with shape(batch_size,a,b,c,...)
    output: scalar
    '''

    # Allows pairwise subtraction by adding additional dimensions and broadcasting
    X_ = tf.expand_dims(X,1)
    Y_ = tf.expand_dims(Y,0)
    k = tf.math.subtract(X_,Y_)

    # tf.norm needs float32
    k = tf.cast(k, dtype=tf.float32)

    # To get the norms of the tensors that come from intermediate layers, the tensors are reshaped into vectors. Afterwards, the norms of the vectors are taken
    dims = list(k.shape)
    dims_to_reduce = dims[2:] # all dimension except the first two (they are batch_sizes) will be reshaped into vectors
    reduced_dim = reduce(lambda x,y: x*y, dims_to_reduce)

    reshaped_k = tf.reshape(k, dims[:2]+[reduced_dim]) # result: (batch_size, batch_size, N)

    k = tf.norm(reshaped_k, axis=2) # result: (batch_size, batch_size)
    
    k = tf.math.square(k)
    k = -k/2.0
    k = tf.math.exp(k)
    
    kernel = tf.math.reduce_mean(k)
    
    return kernel

def mmd_helper(X,Y):
    
    kernel = partial_mmd(X,Y)

    return kernel

def mmd(X,Y):
    '''
    Calculates the maximum mean discrepancy of two distributions.
    input: tensor
    output: scalar
    '''

    L1 = mmd_helper(X,X)
    L2 = mmd_helper(X,Y)
    L3 = mmd_helper(Y,Y)
        
    disc_loss = L1 - L2 + L3

    return disc_loss

def test_mmd():

    a = tf.constant([[0,34,2,3,4],[5,34,76,82,9],[32,342,532,23,1]])
    b = tf.constant([[5,2,7,238,9],[0,1,25,33,4],[32,54,15,78,4]])

    c = tf.constant([0,1,2,3,4])
    d = tf.constant([5,6,7,8,9])

    e = tf.constant([[[[0,34,2,3,4],[5,34,76,82,9],[32,342,532,23,1]], [[3,23,432,342,43],[51,2,72,4,9],[23,3,5323,22,32]]]])
    f = tf.constant([[[[5,2,7,238,9],[0,1,25,33,4],[32,54,15,78,4]], [[6,345,56,76,78],[556,4,766,2,95],[2,32,578,268,657]]]])

    t = mmd(e,f)
    print(t)

#test_mmd()
