import tensorflow as tf

def gaussian_kernel(X,Y):
    '''
    Gaussian kernel
    input: tensor with shape(None,)
    output: tensor with shape(None,)
    '''
    kernel = tf.math.exp(-tf.math.square((tf.math.subtract(X,Y)))/2)

    return kernel

def mmd_helper(X,Y):
    '''
    input: tensor (logit)
    output: tensor with shape(None,)
    '''  

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
    output: tensor with shape(None,)
    '''

    L1 = mmd_helper(X,X)
    L2 = mmd_helper(X,Y)
    L3 = mmd_helper(Y,Y)
        
    disc_loss = L1 - L2 + L3

    return disc_loss

def test_mmd():

    a = tf.constant([[0,1,2,3,4],[5,6,7,8,9]])
    b = tf.constant([[5,6,7,8,9],[0,1,2,3,4]])
    
    t = mmd(a,b)
    print(t)

#test_mmd()
