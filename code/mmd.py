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


def mmd2(X,Y, sigma):

    dimsX = list(X.shape)
    dims_to_reduceX = dimsX[1:] # all dimension except the first (it is the batch_sizes) will be reshaped into a vector
    reduced_dimX = reduce(lambda x,y: x*y, dims_to_reduceX)
   
    dimsY = list(Y.shape)
    dims_to_reduceY = dimsY[1:] # all dimension except the first (it is the batch_sizes) will be reshaped into a vector
    reduced_dimY = reduce(lambda x,y: x*y, dims_to_reduceY)
   
    X = tf.reshape(X, dimsX[:1]+[reduced_dimX])
    Y = tf.reshape(Y, dimsY[:1]+[reduced_dimY])

    XX = tf.matmul(X, X, transpose_b=True)
    XY = tf.matmul(X, Y, transpose_b=True)
    YY = tf.matmul(Y, Y, transpose_b=True)

    X_sqnorms = tf.linalg.tensor_diag_part(XX)
    Y_sqnorms = tf.linalg.tensor_diag_part(YY)

    r = lambda x: tf.expand_dims(x, 0)
    c = lambda x: tf.expand_dims(x, 1)
    
    K_XX = tf.exp(-(1./sigma) * (-2. * XX + c(X_sqnorms) + r(X_sqnorms)))
    K_XY = tf.exp(-(1./sigma) * (-2. * XY + c(X_sqnorms) + r(Y_sqnorms)))
    K_YY = tf.exp(-(1./sigma) * (-2. * YY + c(Y_sqnorms) + r(Y_sqnorms))) 
   
    return tf.reduce_mean(K_XX) -2. * tf.reduce_mean(K_XY) + tf.reduce_mean(K_YY)

def test_mmd():

    a = tf.constant([[0.,34.,2.,3.,4.],[5.,34.,76.,82.,9.],[32.,342.,532.,23.,1.]])
    b = tf.constant([[5.,2.,7.,238.,9.],[0.,1.,25.,33.,4.],[32.,54.,15.,78.,4.]])

    c = tf.constant([[.5,.1,.1,.1,.2],[.3,.2,.1,.0,.4]])
    d = tf.constant([[.2,.1,.1,.4,.2],[.3,.2,.1,.0,.4]])

    c_2 = tf.constant([5.,1.,0.3,5.2,0.1,1.,5.2,5.1,5.1,0.5])
    d_2 = tf.constant([1.,0.7,1.2,1.9,1.2,0.95,0.1,0.2,0.1,0.4])
    
    e = tf.constant([[[[0.,34.,2.,3.,4.],[5.,34.,76.,82.,9.],[32.,342.,532.,23.,1.]], [[3.,23.,432.,342.,43.],[51.,2.,72.,4.,9.],[23.,3.,5323.,22.,32.]]]])
    f = tf.constant([[[[5.,2.,7.,238.,9.],[0.,1.,25.,33.,4.],[32.,54.,15.,78.,4.]], [[6.,345.,56.,76.,78.],[556.,4.,766.,2.,95.],[2.,32.,578.,268.,657.]]]])
    
    c_1 = tf.constant([[0.,1.,2.,3.,4.],[0.,1.,2.,3.,4.],[0.,1.,2.,3.,4.],[0.,1.,2.,3.,4.],[0.,1.,2.,3.,4.]])
    d_1 = tf.constant([[5.,6.,7.,8.,9.],[5.,6.,7.,8.,9.],[5.,6.,7.,8.,9.],[5.,6.,7.,8.,9.],[5.,6.,7.,8.,9.]])
    
    e_1 = tf.constant([[[[0.,34.,2.,3.,4.],[5.,34.,76.,82.,9.],[32.,342.,532.,23.,1.]]], [[[3.,23.,432.,342.,43.],[51.,2.,72.,4.,9.],[23.,3.,5323.,22.,32.]]]])
    f_1 = tf.constant([[[[5.,2.,7.,238.,9.],[0.,1.,25.,33.,4.],[32.,54.,15.,78.,4.]]], [[[6.,345.,56.,76.,78.],[556.,4.,766.,2.,95.],[2.,32.,578.,268.,657.]]]])

    #t_1 = mmd(e,f)
    #t_2 = mmd2(e_1,f_1)
    #t_3 = mmd(a,b)
    #t_4 = mmd2(a,b)
    #t_5 = mmd(c,d)
    #t_6 = mmd2(c,d)

    #t_7 = mmd2(c_1,d_1)
    
    #print(f'mmd(e,f): {t_1}')
    #print(f'mmd2(e_1,f_1): {t_2}')
    #print(f'mmd(a,b): {t_3}')
    #print(f'mmd2(a,b): {t_4}')
    #print(f'mmd(c,d): {t_5}')
    #print(f'mmd2(c,d): {t_6}')
    #print(f'mmd2(c_1,d_1): {t_7}')

    kl = tf.keras.losses.KLDivergence()
    m = (1/2) * (c + d)
    print(f'kl result_cd: {kl(c,m).numpy}')
    print(f'kl result_dc: {kl(d,m).numpy}') 
    print(f'kl result_cdm: {0.5 * kl(c,m) + 0.5 * kl(d,m)}')


#test_mmd()
