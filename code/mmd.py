import tensorflow as tf

def gaussian_kernel(X,Y):
    '''
    Gaussian kernel
    input: tensor with shape(None,)
    output: scalar
    '''
    k = tf.math.subtract(X,Y)
    k = tf.cast(k, dtype=tf.float32)
    k = tf.norm(k)
    k = tf.math.square(k)
    k = -k/2
    kernel = tf.math.exp(k)

    return kernel

def mmd_helper(X,Y):

    X_ = tf.keras.backend.eval(X)
    Y_ = tf.keras.backend.eval(Y)

    loss = 0
    
    for i in X_:
        for t in Y_:
            l = gaussian_kernel(i,t)
            loss += l

    return loss / (len(X_) * len(Y_))

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
    
    t = mmd(a,b)
    print(t)

#test_mmd()
