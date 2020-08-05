import tensorflow as tf
from functools import reduce
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics.pairwise import rbf_kernel

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

    '''
    What if I use the scikit function?
    Caveat: This applies the gaussian kernel pairwise,
            which means for each pair of rows x in X and y in Y. 
    '''
    '''
    K_XX = rbf_kernel(X,X,0.0001) 
    K_XY = rbf_kernel(X,Y,0.0001) 
    K_YY = rbf_kernel(Y,Y,0.0001) 
    '''
    '''
    What if I use the scikit function?
    '''
    
    return tf.reduce_mean(K_XX) -2. * tf.reduce_mean(K_XY) + tf.reduce_mean(K_YY)

def test_mmd():

    a = tf.constant([[0.,34.,2.,3.,4.],[5.,34.,76.,82.,9.],[32.,342.,532.,23.,1.]])
    b = tf.constant([[5.,2.,7.,238.,9.],[0.,1.,25.,33.,4.],[32.,54.,15.,78.,4.]])

    c = tf.constant([[.5,.1,.1,.1,.2],[.3,.2,.1,.0,.4]])
    d = tf.constant([[.2,.1,.1,.4,.2],[.3,.2,.1,.0,.4]])

    c_2 = tf.constant([[5.,1.,0.3,5.2,0.1,1.,5.2,5.1,5.1,0.5]])
    d_2 = tf.constant([[1.,0.7,1.2,1.9,1.2,0.95,0.1,0.2,0.1,0.4]])
    
    e = tf.constant([[[[0.,34.,2.,3.,4.],[5.,34.,76.,82.,9.],[32.,342.,532.,23.,1.]], [[3.,23.,432.,342.,43.],[51.,2.,72.,4.,9.],[23.,3.,5323.,22.,32.]]]])
    f = tf.constant([[[[5.,2.,7.,238.,9.],[0.,1.,25.,33.,4.],[32.,54.,15.,78.,4.]], [[6.,345.,56.,76.,78.],[556.,4.,766.,2.,95.],[2.,32.,578.,268.,657.]]]])
    
    c_1 = tf.constant([[0.,1.,2.,3.,4.],[0.,1.,2.,3.,4.],[0.,1.,2.,3.,4.],[0.,1.,2.,3.,4.],[0.,1.,2.,3.,4.]])
    d_1 = tf.constant([[5.,6.,7.,8.,9.],[5.,6.,7.,8.,9.],[5.,6.,7.,8.,9.],[5.,6.,7.,8.,9.],[5.,6.,7.,8.,9.]])
    
    e_1 = tf.constant([[[[0.,34.,2.,3.,4.],[5.,34.,76.,82.,9.],[32.,342.,532.,23.,1.]]], [[[3.,23.,432.,342.,43.],[51.,2.,72.,4.,9.],[23.,3.,5323.,22.,32.]]]])
    f_1 = tf.constant([[[[5.,2.,7.,238.,9.],[0.,1.,25.,33.,4.],[32.,54.,15.,78.,4.]]], [[[6.,345.,56.,76.,78.],[556.,4.,766.,2.,95.],[2.,32.,578.,268.,657.]]]])
    
    aa = tf.reshape(e, [1,30])
    bb = tf.reshape(f, [1,30])

    print(f"aa: {aa}")
    print(f"bb: {bb}")
    
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

    '''
    kl = tf.keras.losses.KLDivergence()
    m = (1/2) * (c + d)
    print(f'kl result_cd: {kl(c,m).numpy}')
    print(f'kl result_dc: {kl(d,m).numpy}') 
    print(f'kl result_cdm: {0.5 * kl(c,m) + 0.5 * kl(d,m)}')
    '''
    
    kernel = 1.0 * RBF(1.0)
    res_of_RBF = kernel(c_2, d_2)
    print(f"res_of_RBF: {res_of_RBF}")

    print(f"rbf_kernel(c_2,d_2,0.5): {rbf_kernel(c_2,d_2,0.5)}")

    print(f"rbf_kernel(c_2,c_2,0.5): {rbf_kernel(c_2,c_2,0.5)}")

    mmd2(c_2,d_2,2.)

#test_mmd()
