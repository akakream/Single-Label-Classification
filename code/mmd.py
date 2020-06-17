import tensorflow as tf

def gaussian_kernel(X,Y):
    '''
    Gaussian kernel
    input: tensor with shape(None,)
    output: scalar
    '''

    #print(f'THIS IS X: {X}')
    k = tf.math.subtract(X,Y)
    #print(f'This is after subtract: {k}')
    k = tf.cast(k, dtype=tf.float32)
    #print(f'This is after cast: {k}')
    k = tf.norm(k, axis=1)
    #print(f'This is after norm: {k}')
    k = tf.math.square(k)
    #print(f'This is after square: {k}')
    k = -k/2
    #print(f'This is after -k/2: {k}')
    k = tf.math.exp(k)
    #print(f'This is after exp: {k}')
    kernel = tf.math.reduce_sum(k)
    #print(f'This is after reduce_sum: {kernel}')

    return kernel

def mmd_helper(X,Y):
    
    kernel = gaussian_kernel(X,Y)

    return kernel / (len(X) * len(Y))

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

    e = tf.constant([[[0,34,2,3,4],[5,34,76,82,9],[32,342,532,23,1]], [[3,23,432,342,43],[51,2,72,4,9],[23,3,5323,22,32]]])
    f = tf.constant([[[5,2,7,238,9],[0,1,25,33,4],[32,54,15,78,4]], [[6,345,56,76,78],[556,4,766,2,95],[2,32,578,268,657]]])

    t = mmd(e,f)
    print(t)

#test_mmd()
