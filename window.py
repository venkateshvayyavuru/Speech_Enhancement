#############################################################
#                                                           #
#   Author:  Dr K. Sri Rama Murty                           #
#   Co-Author:Sivaganesh Andhavarapu                        #
#   Institute: Indian Institute of Techonolgy Hyderabad     #
#                                                           #
#############################################################

import numpy as np
import tensorflow as tf


def win(signal):
    print("---signal.shape -------",signal.shape)
    k1=tf.zeros(20,dtype=tf.float32)
    k2=tf.ones(100,dtype=tf.float32)
    k3=tf.zeros(137,dtype=tf.float32)
    w=tf.concat([k1,k2,k3],0)
    matrix=tf.ones([1,1])*w
    X=tf.multiply(signal,matrix)
    print("---X.shape----",X.shape)   
    return X

